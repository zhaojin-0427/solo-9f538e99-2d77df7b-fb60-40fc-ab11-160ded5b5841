"""主平场制备 FastAPI 接口。

* ``GET  /docs``    Swagger 交互文档，可直接上传 FITS。
* ``GET  /health``  存活探针。
* ``POST /prepare`` 上传 3~12 张平场 FITS 与饱和阈值，返回成品 FITS。

磁盘暂存约束
============

每张原始曝光在服务磁盘上**至多暂存一份**：multipart 报文由
python-multipart 的推送式解析器逐块解析，文件像元字节直接落到该帧唯一的
暂存文件，不经过 Starlette/SpooledTemporaryFile 的二次落盘，也不产生任何
上传解析临时文件。无论成功失败，请求结束即清理本批输入暂存。
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
import python_multipart.multipart as multipart_parser
from python_multipart.exceptions import FormParserError, MultipartParseError

from .pipeline import (
    MAX_FRAMES,
    MIN_FRAMES,
    PreparationError,
    build_master_flat,
)

logger = logging.getLogger("master_flat")

app = FastAPI(
    title="天文测光主平场制备接口",
    version="1.0.0",
    description=(
        "上传 3~12 张已扣偏置的无压缩二维 16 位整数平场 FITS，"
        "按 BLANK / BSCALE / BZERO 处理后进行每帧中位数归一与像素级中位数聚合，"
        "输出含校正系数主图与参与帧数扩展图的成品 FITS。"
    ),
)

# 单容器内存预算 256 MiB，串行处理避免并发制备叠加内存。
_preparation_lock = threading.Lock()

# 单张上传上限：4096x4096 的 16 位主图数据块约 32 MiB，加上头部余量给 64 MiB。
MAX_UPLOAD_BYTES = 64 * 2 ** 20
# 单个表单字段（saturation）文本上限。
MAX_FIELD_BYTES = 256


def _data_dir(env_name: str, default: str) -> Path:
    path = Path(__file__).resolve().parent.parent / os.environ.get(env_name, default)
    path.mkdir(parents=True, exist_ok=True)
    return path


WORK_DIR = _data_dir("FLAT_WORK_DIR", "work")
OUTPUT_DIR = _data_dir("FLAT_OUTPUT_DIR", "products")


@app.exception_handler(PreparationError)
async def preparation_error_handler(
    request: Request, exc: PreparationError
) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


class _MultipartStreamer:
    """推送式 multipart 回调：文件部分直写唯一暂存文件，字段部分驻留内存。

    全程不使用 SpooledTemporaryFile：除该帧唯一暂存文件外，磁盘上不会
    出现任何上传解析临时文件。
    """

    def __init__(self, batch_dir: Path) -> None:
        self._batch_dir = batch_dir
        self._header_field = bytearray()
        self._header_value = bytearray()
        self._part_name: bytes | None = None
        self._part_size = 0
        self._handle = None
        self._target: Path | None = None
        self._field_chunks: dict[str, list[bytes]] = {}
        self.frame_paths: list[Path] = []
        self.source_names: list[str] = []
        self.fields: dict[str, str] = {}

    # ---- parser callbacks ----
    def on_part_begin(self) -> None:
        self._header_field = bytearray()
        self._header_value = bytearray()
        self._part_name = None
        self._part_size = 0
        self._handle = None
        self._target = None

    def on_header_begin(self) -> None:
        self._header_field = bytearray()
        self._header_value = bytearray()

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_field += data[start:end]

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._header_value += data[start:end]

    def on_header_end(self) -> None:
        if self._header_field.strip().lower() == b"content-disposition":
            _, params = multipart_parser.parse_options_header(
                self._header_value.decode("latin-1")
            )
            self._part_name = params.get(b"name")
            self._part_filename = params.get(b"filename")

    def on_headers_finished(self) -> None:
        if getattr(self, "_part_filename", None) is not None:
            if len(self.frame_paths) >= MAX_FRAMES:
                raise PreparationError(
                    f"平场帧数须在 {MIN_FRAMES}~{MAX_FRAMES} 张之间，"
                    f"收到超过 {MAX_FRAMES} 个文件部分"
                )
            raw_name = self._part_filename.decode("utf-8", errors="replace")
            safe_name = (
                Path(raw_name).name or f"frame_{len(self.frame_paths) + 1:02d}.fits"
            )
            index = len(self.frame_paths) + 1
            # 每个上传文件放在独立的新空目录中，隔离不可信数据。
            frame_dir = self._batch_dir / f"frame_{index:02d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            self._target = frame_dir / safe_name
            try:
                self._handle = self._target.open("wb")
            except OSError as exc:
                raise PreparationError(
                    f"上传文件保存失败：{safe_name}：{exc}"
                ) from exc
            self.source_names.append(raw_name)

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        chunk = data[start:end]
        if not chunk:
            return
        self._part_size += len(chunk)
        if self._handle is not None:
            if self._part_size > MAX_UPLOAD_BYTES:
                raise PreparationError(
                    f"上传文件超过单帧上限 {MAX_UPLOAD_BYTES // 2 ** 20} MiB"
                )
            try:
                self._handle.write(chunk)
                self._handle.flush()
            except OSError as exc:
                # ENOSPC 等工作盘写满场景：整批报错，由外层统一清理暂存。
                raise PreparationError(
                    f"工作盘写入失败，整批制备中止：{exc}", status_code=500
                ) from exc
        else:
            if self._part_size > MAX_FIELD_BYTES:
                raise PreparationError("表单字段内容过长")
            if self._part_name is not None:
                name = self._part_name.decode("latin-1")
                self._field_chunks.setdefault(name, []).append(chunk)

    def on_part_end(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError as exc:
                raise PreparationError(
                    f"上传文件保存失败：{self._target.name}：{exc}"
                ) from exc
            self._handle = None
            if self._part_size == 0:
                raise PreparationError(f"上传文件为空：{self._target.name}")
            self.frame_paths.append(self._target)
        elif self._part_name is not None:
            name = self._part_name.decode("latin-1")
            self.fields[name] = b"".join(
                self._field_chunks.pop(name, [])
            ).decode("utf-8", errors="replace")
        self._target = None

    def on_end(self) -> None:
        return None

    def close_handles(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None


@app.post("/prepare", response_class=FileResponse, include_in_schema=False)
async def prepare_master_flat(request: Request) -> FileResponse:
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type.lower():
        raise PreparationError("请求必须为 multipart/form-data 报文")
    try:
        _, params = multipart_parser.parse_options_header(content_type)
        boundary = params.get(b"boundary")
    except (MultipartParseError, FormParserError) as exc:
        raise PreparationError(f"Content-Type 无法解析：{exc}") from exc
    if not boundary:
        raise PreparationError("multipart 报文缺少 boundary")

    batch_dir = Path(tempfile.mkdtemp(prefix="batch_", dir=WORK_DIR))
    streamer = _MultipartStreamer(batch_dir)
    callbacks = {
        "on_part_begin": streamer.on_part_begin,
        "on_part_data": streamer.on_part_data,
        "on_part_end": streamer.on_part_end,
        "on_header_begin": streamer.on_header_begin,
        "on_header_field": streamer.on_header_field,
        "on_header_value": streamer.on_header_value,
        "on_header_end": streamer.on_header_end,
        "on_headers_finished": streamer.on_headers_finished,
        "on_end": streamer.on_end,
    }
    try:
        try:
            parser = multipart_parser.MultipartParser(boundary, callbacks)
            async for chunk in request.stream():
                if chunk:
                    parser.write(chunk)
            parser.finalize()
        except (MultipartParseError, FormParserError) as exc:
            raise PreparationError(f"multipart 报文解析失败：{exc}") from exc

        if not (MIN_FRAMES <= len(streamer.frame_paths) <= MAX_FRAMES):
            raise PreparationError(
                f"平场帧数须在 {MIN_FRAMES}~{MAX_FRAMES} 张之间，"
                f"收到 {len(streamer.frame_paths)} 张"
            )

        raw_saturation = streamer.fields.get("saturation")
        if raw_saturation is None or raw_saturation.strip() == "":
            raise PreparationError("缺少 saturation 表单字段")
        try:
            saturation = float(raw_saturation)
        except ValueError as exc:
            raise PreparationError("饱和阈值必须为数值") from exc
        if not (saturation == saturation and saturation not in (
            float("inf"),
            float("-inf"),
        )):
            raise PreparationError("饱和阈值必须为有限数值")
        if saturation <= 0.0:
            raise PreparationError("饱和阈值必须为正且有限的数值")

        with _preparation_lock:
            product = build_master_flat(
                frame_paths=streamer.frame_paths,
                source_names=streamer.source_names,
                saturation=saturation,
                output_dir=OUTPUT_DIR,
            )

        logger.info(
            "主平场制备成功：%s <- %d 帧", product.name, len(streamer.frame_paths)
        )
        return FileResponse(
            product,
            media_type="application/fits",
            filename=product.name,
            headers={"X-Frame-Count": str(len(streamer.frame_paths))},
        )
    finally:
        streamer.close_handles()
        # 成功或失败都清理本批输入暂存（含上传解析产生的全部文件）。
        shutil.rmtree(batch_dir, ignore_errors=True)


def _prepare_openapi_schema() -> dict:
    """补回 /prepare 的 OpenAPI 描述，使 /docs 上的上传表单可直接使用。"""
    schema = app._original_openapi()
    schema["paths"].setdefault("/prepare", {})["post"] = {
        "summary": "上传平场 FITS 制备主平场",
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["files", "saturation"],
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {
                                    "type": "string", "format": "binary"
                                },
                                "description": (
                                    f"平场 FITS 文件，数量 {MIN_FRAMES}~{MAX_FRAMES} 张"
                                ),
                            },
                            "saturation": {
                                "type": "number",
                                "description": (
                                    "正且有限的饱和阈值"
                                    "（与 BSCALE/BZERO 换算后同单位）"
                                ),
                            },
                        },
                    }
                }
            }
        },
        "responses": {
            "200": {
                "description": "成品 FITS（PRIMARY 校正系数 + COUNTS 参与帧数）",
                "content": {
                    "application/fits": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            },
            "422": {"description": "输入/数据问题"},
            "500": {"description": "写出问题"},
        },
    }
    return schema


app._original_openapi = app.openapi
app.openapi = _prepare_openapi_schema
