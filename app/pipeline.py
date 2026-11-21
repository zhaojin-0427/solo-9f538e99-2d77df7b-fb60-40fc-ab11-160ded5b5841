"""主平场（master flat）制备管线。

设计要点
========

* 输入为 3~12 张已扣偏置、无压缩、二维 16 位整数单图 FITS。
* 用 ``do_not_scale_image_data=True`` 读头部（不实体化像元），自行按
  ``BSCALE`` / ``BZERO`` 换算，显式按 ``BLANK`` 排除缺测像元。
* 有效像元：换算后 ``0 < value < saturation``（且不等于 BLANK）。
* 每帧有效值先除以该帧有效像元中位数；再对同一位置的有效归一值取中位数；
  参与帧数少于 3 的位置输出 NaN。
* 16 位整数只有 65536 个取值，第一趟用直方图精确求每帧中位数，
  避免一次性存放整帧有效值。

磁盘 I/O 预算（单帧在暂存上至多一份）
------------------------------------

每帧只通过 :class:`FrameReader` 持有**一个**裸文件句柄，从 FITS 数据块
起始偏移按需读取原始像元字节，全程不复制文件、不产生临时文件：

* 结构校验只读头部卡片（2880 字节块），按约定不计入像元读取预算；
* 第一趟（直方图中位数）顺序读完整帧像元，恰好 1 倍帧像元字节数；
* 第二趟（条带聚合）再顺序分块读完整帧像元，再 1 倍；
* 累计像元读取字节数硬上限为 ``2 × 帧像元字节数``，命中操作系统缓存
  的重复读取同样计数，超限直接整批失败。

内存预算（256 MiB 容器）
-----------------------

第二趟按行带（band）聚合：同一行带内依次读入各帧的部分像元，
行带高度由 ``BAND_BUDGET_BYTES`` 反推，使常驻 float32 缓冲不超过 8 MiB；
帧轴中位数用原地排序 + 索引选择，避免 ``np.nanmedian`` 的整份复制与
类型提升；输出用 ``ndarray.tofile`` 流式直写 FITS 数据块。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits

MIN_FRAMES = 3
MAX_FRAMES = 12
MAX_DIM = 4096
MIN_FRAMES_PER_PIXEL = 3
# 行带聚合缓冲上限（float32）。取值需为排序/花式索引等瞬时分配、
# 常驻结果图与解释器基线留出余量，使总峰值稳在 256 MiB 容器预算内。
BAND_BUDGET_BYTES = 8 * 2 ** 20
# 单次行块的行数，限制换算时的临时数组大小。
ROW_CHUNK = 256
RAW_BIN_COUNT = 65536  # int16 全部取值数
RAW_OFFSET = 32768
FITS_BLOCK = 2880  # FITS 头部与数据块的最小对齐单位
# 每帧允许从暂存读取的像元字节数相对帧像元数据量的倍数。
MAX_READ_MULTIPLIER = 2.0


class PreparationError(Exception):
    """整批制备失败；status_code 决定 HTTP 响应码。"""

    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass
class FrameReader:
    """单帧暂存文件的唯一读取通道。

    整个制备过程中每帧只打开这一个文件句柄；像元通过数据块偏移按需
    部分读取，``pixel_bytes_read`` 累计像元字节数（含缓存命中的读取），
    结构头部读取不计入该计数。
    """

    path: Path
    image_index: int
    data_offset: int
    pixel_bytes: int
    pixel_bytes_read: int = 0
    _handle: object = field(default=None, repr=False)

    def open(self) -> "FrameReader":
        try:
            self._handle = open(self.path, "rb")
        except OSError as exc:
            raise PreparationError(f"暂存文件无法读取：{self.path.name}：{exc}") from exc
        return self

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None

    def __enter__(self) -> "FrameReader":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def read_rows(self, row0: int, row1: int, width: int) -> np.ndarray:
        """读取第 ``[row0, row1)`` 行的原始 16 位整数像元（native dtype）。"""
        rows = row1 - row0
        count = rows * width
        nbytes = count * 2
        budget = int(math.ceil(self.pixel_bytes * MAX_READ_MULTIPLIER))
        if self.pixel_bytes_read + nbytes > budget:
            raise PreparationError(
                f"帧 {self.path.name} 的暂存像元读取量超过帧像元数据量的 "
                f"{MAX_READ_MULTIPLIER:g} 倍上限，整批制备失败",
                status_code=500,
            )
        offset = self.data_offset + row0 * width * 2
        try:
            self._handle.seek(offset)
            blob = self._handle.read(nbytes)
        except OSError as exc:
            raise PreparationError(f"FITS 文件截断或数据损坏：{self.path.name}：{exc}") from exc
        if len(blob) != nbytes:
            raise PreparationError(
                f"FITS 文件截断或数据损坏：{self.path.name}："
                f"期望 {nbytes} 字节像元，实际读到 {len(blob)} 字节"
            )
        self.pixel_bytes_read += nbytes
        # FITS 数据块固定大端；转成机器字节序供 numpy 向量化计算。
        return np.frombuffer(blob, dtype=">i2").astype(np.int16, copy=True).reshape(
            rows, width
        )


@dataclass(frozen=True)
class FrameInfo:
    """单帧的元数据与第一趟统计结果。"""

    reader: FrameReader
    source_name: str
    shape: tuple[int, int]
    bscale: float
    bzero: float
    blank: int | None
    median: float

    @property
    def path(self) -> Path:
        return self.reader.path


def _open_header_only(path: Path) -> fits.HDUList:
    """只打开 FITS 头部：不实体化像元，结构校验不产生像元读取量。"""
    try:
        return fits.open(path, do_not_scale_image_data=True, memmap=False)
    except (OSError, EOFError, ValueError, fits.VerifyError) as exc:
        raise PreparationError(f"FITS 文件截断或无法解析：{path.name}：{exc}") from exc


def _data_offset(hdu: fits.ImageHDU) -> int:
    """返回图像 HDU 数据块在文件中的字节偏移（含历史兼容回退）。"""
    try:
        offset = hdu.fileinfo()["datLoc"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise PreparationError("无法定位 FITS 数据块偏移，整批制备失败") from exc
    return int(offset)


def _expected_file_size(hdu: fits.ImageHDU, offset: int) -> int:
    return offset + int(hdu.data.size) * abs(int(hdu.header["BITPIX"])) // 8


def _validate_hdus(hdul: fits.HDUList, name: str) -> tuple[int, fits.ImageHDU]:
    """校验每张图恰好包含一幅无压缩二维 16 位整数主图，返回其 HDU 序号。

    只访问头部与形状元数据，不实体化任何像元数据。
    """
    image_hdus: list[tuple[int, fits.ImageHDU]] = []
    for index, hdu in enumerate(hdul):
        if isinstance(hdu, fits.CompImageHDU):
            raise PreparationError(f"帧 {name} 含压缩图像 HDU，仅接受无压缩 FITS")
        if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
            raise PreparationError(f"帧 {name} 含表 HDU，每张 FITS 只能有一幅主图")
        try:
            has_data = hdu.data is not None
        except (OSError, EOFError, ValueError) as exc:
            raise PreparationError(f"FITS 文件截断或数据损坏：{name}：{exc}") from exc
        if hdu.is_image and has_data:
            image_hdus.append((index, hdu))

    if len(image_hdus) != 1:
        raise PreparationError(
            f"帧 {name} 含 {len(image_hdus)} 幅图像，要求每张 FITS 恰好一幅主图"
        )

    index, hdu = image_hdus[0]
    bitpix = hdu.header.get("BITPIX")
    if bitpix != 16:
        raise PreparationError(f"帧 {name} 的 BITPIX={bitpix}，要求为 16 位整数（BITPIX=16）")
    if int(hdu.header.get("NAXIS", 0)) != 2:
        raise PreparationError(f"帧 {name} 的 NAXIS={hdu.header.get('NAXIS')}，要求二维图像")

    shape = tuple(int(dim) for dim in hdu.shape)
    if len(shape) != 2:
        raise PreparationError(f"帧 {name} 主图不是二维图像")
    height, width = shape
    if height == 0 or width == 0 or height > MAX_DIM or width > MAX_DIM:
        raise PreparationError(
            f"帧 {name} 尺寸 {width}x{height} 非法，宽高均须在 1~{MAX_DIM} 之间"
        )
    return index, hdu


def _camera_filter_keys(header: fits.Header) -> dict[str, str]:
    """提取相机/滤镜标识；缺失关键字不参与比对（不强制要求这些头存在）。"""
    result: dict[str, str] = {}
    for key in ("INSTRUME", "FILTER"):
        if key in header and header[key] is not None:
            value = str(header[key]).strip()
            if value:
                result[key] = value
    return result


def _read_scaling(header: fits.Header, name: str) -> tuple[float, float, int | None]:
    try:
        bscale = float(header.get("BSCALE", 1.0))
        bzero = float(header.get("BZERO", 0.0))
    except (TypeError, ValueError) as exc:
        raise PreparationError(f"帧 {name} 的 BSCALE/BZERO 关键字非法：{exc}") from exc
    if not (math.isfinite(bscale) and math.isfinite(bzero)) or bscale == 0.0:
        raise PreparationError(f"帧 {name} 的 BSCALE/BZERO 非有限值或 BSCALE 为 0")

    blank: int | None = None
    if "BLANK" in header:
        try:
            blank = int(header["BLANK"])
        except (TypeError, ValueError) as exc:
            raise PreparationError(f"帧 {name} 的 BLANK 关键字不是整数：{exc}") from exc
        if not (-32768 <= blank <= 32767):
            raise PreparationError(f"帧 {name} 的 BLANK={blank} 超出 16 位整数范围")
    return bscale, bzero, blank


def _frame_median(
    reader: FrameReader,
    height: int,
    width: int,
    bscale: float,
    bzero: float,
    blank: int | None,
    saturation: float,
) -> tuple[int, np.ndarray]:
    """第一趟：基于原始值直方图求有效像元中位数（整帧顺序读 1 倍像元）。

    返回 ``(有效像元总数, 65536 桶直方图)``，由调用方换算精确中位数，
    避免一次性存放整帧有效值。
    """
    hist = np.zeros(RAW_BIN_COUNT, dtype=np.int64)

    for r0 in range(0, height, ROW_CHUNK):
        r1 = min(r0 + ROW_CHUNK, height)
        block = reader.read_rows(r0, r1, width).astype(np.int32, copy=False)
        scaled = block.astype(np.float64) * bscale + bzero
        valid = (scaled > 0.0) & (scaled < saturation)
        if blank is not None:
            valid &= block != blank
        if not valid.any():
            continue
        indices = block + RAW_OFFSET
        hist += np.bincount(indices[valid], minlength=RAW_BIN_COUNT)

    total = int(hist.sum())
    if total == 0:
        raise PreparationError(f"帧 {reader.path.name} 不存在任何有效像元，整批制备失败")
    return total, hist


def _median_scaled_from_hist(
    hist: np.ndarray, total: int, bscale: float, bzero: float
) -> float:
    """由原始整数直方图精确换算有效像元（缩放后）的中位数。"""
    k1 = (total - 1) // 2
    k2 = total // 2
    ordered = hist if bscale > 0 else hist[::-1]
    cumulative = np.cumsum(ordered)
    p1 = int(np.searchsorted(cumulative, k1 + 1, side="left"))
    p2 = int(np.searchsorted(cumulative, k2 + 1, side="left"))
    if bscale > 0:
        raw1, raw2 = p1 - RAW_OFFSET, p2 - RAW_OFFSET
    else:
        raw1, raw2 = RAW_OFFSET - 1 - p1, RAW_OFFSET - 1 - p2
    return ((raw1 * bscale + bzero) + (raw2 * bscale + bzero)) / 2.0


def _band_height(n_frames: int, width: int) -> int:
    """根据内存预算计算行带高度（float32 聚合同步缓冲）。"""
    per_row = n_frames * width * np.dtype(np.float32).itemsize
    height = BAND_BUDGET_BYTES // max(per_row, 1)
    return max(1, min(height, MAX_DIM))


def _aggregate_buffer(
    buffer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """对帧轴求有效归一值中位数与有效帧数。

    不用 ``np.nanmedian``：它会对整个行带再复制一份并做类型提升，
    在 256 MiB 预算下峰值超标。这里原地排序（NaN 按浮点规则排到末尾），
    再按每像素的有效计数用索引直接取中位数。
    """
    counts = np.isfinite(buffer).sum(axis=0, dtype=np.uint8)

    buffer.sort(axis=0)  # 原地，避免整份行带的复制与 float64 提升
    median_band = np.full(buffer.shape[1:], np.nan, dtype=np.float32)
    valid_pixels = counts > 0
    rows, cols = np.nonzero(valid_pixels)
    k1 = ((counts[valid_pixels] - 1) // 2).astype(np.int64)
    k2 = (counts[valid_pixels] // 2).astype(np.int64)
    # 偶数计数取中间两个均值；奇数计数两索引相同，等价直接取值。
    median_band[valid_pixels] = (
        buffer[k1, rows, cols].astype(np.float64)
        + buffer[k2, rows, cols].astype(np.float64)
    ) / 2.0
    median_band[counts < MIN_FRAMES_PER_PIXEL] = np.nan
    return median_band, counts


def _fill_band(
    frames: list[FrameInfo],
    row0: int,
    row1: int,
    width: int,
    saturation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """聚合一个行带：各帧仅按需读取该行带的像元，返回中位数带与参与帧数带。"""
    band_h = row1 - row0
    buffer = np.full((len(frames), band_h, width), np.nan, dtype=np.float32)

    for f_index, frame in enumerate(frames):
        block = frame.reader.read_rows(row0, row1, width).astype(np.int32, copy=False)
        scaled = block.astype(np.float64) * frame.bscale + frame.bzero
        valid = (scaled > 0.0) & (scaled < saturation)
        if frame.blank is not None:
            valid &= block != frame.blank
        normalized = (scaled / frame.median).astype(np.float32)
        np.copyto(buffer[f_index], normalized, where=valid)

    return _aggregate_buffer(buffer)


def _sanitize_ascii(text: str, limit: int = 60) -> str:
    cleaned = "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in text).strip()
    return cleaned[:limit] or "unknown"


def build_master_flat(
    frame_paths: list[Path],
    source_names: list[str],
    saturation: float,
    output_dir: Path,
    io_stats: dict[str, int] | None = None,
) -> Path:
    """制备主平场，成功后返回成品 FITS 的最终路径，失败则整体报错。

    成品先写到 ``output_dir`` 内的隐藏临时文件（与成品同目录、同卷，
    不额外复制输入），回读校验通过后原子改名；任何一步失败都不会留下
    半成品。``io_stats`` 若非空，将填入每帧累计像元读取字节数等统计。
    """
    if not (math.isfinite(saturation) and saturation > 0.0):
        raise PreparationError("饱和阈值必须为正且有限的数值")
    if not (MIN_FRAMES <= len(frame_paths) <= MAX_FRAMES):
        raise PreparationError(
            f"平场帧数须在 {MIN_FRAMES}~{MAX_FRAMES} 张之间，收到 {len(frame_paths)} 张"
        )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PreparationError(f"输出目录不可用，整批制备失败：{exc}", status_code=500) from exc

    frames: list[FrameInfo] = []
    readers: list[FrameReader] = []
    common_shape: tuple[int, int] | None = None
    common_camera_filter: dict[str, str] | None = None

    try:
        # 第一趟准备：头部结构校验（不读像元）并打开每帧唯一读取句柄。
        for path, source_name in zip(frame_paths, source_names):
            with _open_header_only(path) as hdul:
                image_index, hdu = _validate_hdus(hdul, path.name)
                shape = (int(hdu.shape[0]), int(hdu.shape[1]))
                bscale, bzero, blank = _read_scaling(hdu.header, path.name)
                camera_filter = _camera_filter_keys(hdu.header)
                data_offset = _data_offset(hdu)
                pixel_bytes = shape[0] * shape[1] * 2
                expected_size = _expected_file_size(hdu, data_offset)

            try:
                actual_size = path.stat().st_size
            except OSError as exc:
                raise PreparationError(f"暂存文件无法访问：{path.name}：{exc}") from exc
            if actual_size < expected_size:
                raise PreparationError(
                    f"FITS 文件截断或数据损坏：{path.name}："
                    f"文件长度 {actual_size} 小于所需 {expected_size} 字节"
                )

            if common_shape is None:
                common_shape = shape
            elif shape != common_shape:
                raise PreparationError(
                    f"帧 {path.name} 尺寸 {shape[1]}x{shape[0]} 与首帧 "
                    f"{common_shape[1]}x{common_shape[0]} 不一致，同批尺寸须一致"
                )

            if common_camera_filter is None:
                common_camera_filter = camera_filter
            elif camera_filter != common_camera_filter:
                raise PreparationError(
                    f"帧 {path.name} 的相机/滤镜标识 {camera_filter} 与首帧 "
                    f"{common_camera_filter} 不一致，同批相机与滤镜须一致"
                )

            reader = FrameReader(
                path=path,
                image_index=image_index,
                data_offset=data_offset,
                pixel_bytes=pixel_bytes,
            ).open()
            readers.append(reader)

            total, hist = _frame_median(
                reader, shape[0], shape[1], bscale, bzero, blank, saturation
            )
            median = _median_scaled_from_hist(hist, total, bscale, bzero)
            if not math.isfinite(median) or median <= 0.0:
                raise PreparationError(f"帧 {path.name} 的有效中位数非法（{median}）")
            frames.append(
                FrameInfo(reader, source_name, shape, bscale, bzero, blank, median)
            )

        assert common_shape is not None
        height, width = common_shape
        n_frames = len(frames)

        master = np.full((height, width), np.nan, dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.uint8)

        # 第二趟：自适应行带聚合，每帧再顺序读 1 倍像元，总读取量 ≤ 2 倍。
        band_h = _band_height(n_frames, width)
        for r0 in range(0, height, band_h):
            r1 = min(r0 + band_h, height)
            median_band, count_band = _fill_band(frames, r0, r1, width, saturation)
            master[r0:r1, :] = median_band
            counts[r0:r1, :] = count_band

        product = _write_output(frames, saturation, master, counts, output_dir)

        if io_stats is not None:
            io_stats["frames"] = n_frames
            io_stats["pixel_bytes_per_frame"] = height * width * 2
            io_stats["max_pixel_bytes_read"] = max(
                rd.pixel_bytes_read for rd in readers
            )
            io_stats["total_pixel_bytes_read"] = sum(
                rd.pixel_bytes_read for rd in readers
            )
        return product
    finally:
        for reader in readers:
            reader.close()


def _write_fits_low_memory(
    master: np.ndarray,
    counts: np.ndarray,
    header0: fits.Header,
    header1: fits.Header,
    tmp_path: Path,
) -> None:
    """流式写出双 HDU FITS，避免 ``HDUList.writeto`` 的整图内部缓冲。

    主图 float32（BITPIX=-32），扩展图 uint8（BITPIX=8，帧数最多 12）。
    头部仍由 astropy 生成以保证关键字合规；数据块用 ``ndarray.tofile``
    直接按大端字节追加，并补齐到 2880 字节块。
    """

    def header_block(header: fits.Header) -> bytes:
        # 直接序列化调用方基于真实数组建好的头部；切勿再用
        # PrimaryHDU(header=...) 重建，那会把 BITPIX/NAXIS 重置。
        # padding=True 保证输出为整 2880 字节块且 END 卡对齐卡边界。
        return header.tostring(sep="", padding=True).encode("ascii")

    blocks = (header_block(header0), header_block(header1))

    with tmp_path.open("wb") as handle:
        handle.write(blocks[0])
        np.ascontiguousarray(master, dtype=">f4").tofile(handle)
        pad = (FITS_BLOCK - master.nbytes % FITS_BLOCK) % FITS_BLOCK
        if pad:
            handle.write(b"\0" * pad)
        handle.write(blocks[1])
        np.ascontiguousarray(counts, dtype=np.uint8).tofile(handle)
        pad = (FITS_BLOCK - counts.nbytes % FITS_BLOCK) % FITS_BLOCK
        if pad:
            handle.write(b"\0" * pad)
        handle.flush()
        os.fsync(handle.fileno())


def _write_output(
    frames: list[FrameInfo],
    saturation: float,
    master: np.ndarray,
    counts: np.ndarray,
    output_dir: Path,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    token = os.urandom(3).hex()
    final_name = f"master_flat_{timestamp}_{token}.fits"
    # 临时文件与成品同目录同卷：原子改名即可，不产生跨卷复制，
    # 也不在输入暂存盘上留下任何临时文件。
    tmp_path = output_dir / f".{final_name}.tmp"
    final_path = output_dir / final_name

    primary_hdu = fits.PrimaryHDU(data=master)
    header = primary_hdu.header
    header["IMAGTYPE"] = ("MASTERFLAT", "Master flat field correction coefficients")
    header["NFRAMES"] = (len(frames), "Number of input flat frames")
    header["SATURATE"] = (saturation, "Saturation threshold in data units")
    header["MINCNT"] = (MIN_FRAMES_PER_PIXEL, "Minimum frames for a valid pixel")
    header["CREATED"] = (
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "Creation time (UTC)",
    )
    for idx, frame in enumerate(frames):
        header[f"MED{idx + 1:02d}"] = (
            frame.median,
            f"Valid-pixel median of frame {idx + 1}",
        )
    header.add_history(
        "Pixels scaled by BSCALE/BZERO; BLANK, non-positive and saturated "
        "pixels excluded; per-frame median normalization followed by pixelwise "
        "median; pixels seen by fewer than 3 frames are NaN."
    )
    for idx, frame in enumerate(frames):
        header.add_history(f"Input {idx + 1:02d}: {_sanitize_ascii(frame.source_name)}")

    counts_hdu = fits.ImageHDU(data=counts, name="COUNTS")
    # 取头部副本后立即释放 HDU，避免 HDU 额外持有数据引用。
    header0, header1 = header.copy(), counts_hdu.header.copy()
    del primary_hdu, counts_hdu, header

    try:
        _write_fits_low_memory(master, counts, header0, header1, tmp_path)
        # 头部回读校验结构与尺寸（不实体化数据，避免额外内存），
        # 再用 mmap 抽样确认数据块可解析。
        with fits.open(tmp_path, memmap=False) as verify:
            if len(verify) != 2:
                raise PreparationError("输出 FITS 结构校验失败", status_code=500)
            if verify[0].shape != master.shape or verify[1].shape != counts.shape:
                raise PreparationError("输出 FITS 尺寸校验失败", status_code=500)
            if verify[1].name != "COUNTS":
                raise PreparationError("输出 FITS 扩展名校验失败", status_code=500)
        with fits.open(tmp_path, memmap=True) as verify:
            sample0 = np.asarray(verify[0].data)
            sample1 = np.asarray(verify[1].data)
            if sample0.shape != master.shape or sample1.shape != counts.shape:
                raise PreparationError("输出 FITS 数据块校验失败", status_code=500)
            # 帧数图取值必须落在 0~NFRAMES；确认数据块整体可读。
            if int(sample1.min()) < 0 or int(sample1.max()) > len(frames):
                raise PreparationError("输出 FITS 数据块校验失败", status_code=500)
        os.replace(tmp_path, final_path)
    except PreparationError:
        tmp_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise PreparationError(f"主平场写出失败：{exc}", status_code=500) from exc

    return final_path
