"""管线层测试：数值正确性、过滤规则、磁盘读取预算与各类整批失败。"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from astropy.io import fits

from app.pipeline import (
    MAX_DIM,
    MAX_FRAMES,
    MAX_READ_MULTIPLIER,
    MIN_FRAMES,
    FrameReader,
    PreparationError,
    build_master_flat,
    _median_scaled_from_hist,
)
from conftest import make_flat_fits, reference_master


def _run(tmp_path, paths, saturation, raws=None, bscale=1.0, bzero=0.0, blank=None):
    io_stats: dict[str, int] = {}
    product = build_master_flat(
        paths,
        [p.name for p in paths],
        saturation,
        output_dir=tmp_path / "out",
        io_stats=io_stats,
    )
    return product, io_stats


def test_master_matches_reference(tmp_path, batch_factory):
    paths, raws, saturation = batch_factory(n=5)
    product, _ = _run(tmp_path, paths, saturation)
    expected_master, expected_counts = reference_master(
        raws, 1.0, 0.0, saturation, blank=None
    )
    with fits.open(product) as out:
        assert len(out) == 2
        master, counts = out[0].data, out[1].data
        assert master.dtype.kind == "f" and master.dtype.itemsize == 4
        assert counts.dtype == np.uint8
        np.testing.assert_array_equal(counts, expected_counts)
        ok = np.isfinite(expected_master)
        np.testing.assert_allclose(master[ok], expected_master[ok], rtol=1e-5)
        assert np.isnan(master[~ok]).all()
        assert out[0].header["NFRAMES"] == 5
        assert out[1].name == "COUNTS"


def test_blank_bscale_bzero_pipeline(tmp_path, batch_factory):
    blank = -32768
    paths, raws, saturation = batch_factory(
        n=6, bscale=0.5, bzero=10.0, blank=blank
    )
    product, _ = _run(tmp_path, paths, saturation)
    expected_master, expected_counts = reference_master(
        raws, 0.5, 10.0, saturation, blank=blank
    )
    with fits.open(product) as out:
        master, counts = out[0].data, out[1].data
        np.testing.assert_array_equal(counts, expected_counts)
        ok = np.isfinite(expected_master)
        np.testing.assert_allclose(master[ok], expected_master[ok], rtol=1e-4)
        assert np.isnan(master[~ok]).all()


def test_pixels_with_fewer_than_three_frames_are_nan(tmp_path):
    rng = np.random.default_rng(0)
    raws = [rng.integers(500, 800, size=(20, 20)).astype(np.int16) for _ in range(4)]
    # 同一位置只在两帧里有效：另外两帧填 0。
    raws[0][5, 5] = 0
    raws[1][5, 5] = 0
    paths = [
        make_flat_fits(tmp_path / f"f{i}.fits", raw) for i, raw in enumerate(raws)
    ]
    product, _ = _run(tmp_path, paths, 5000.0)
    with fits.open(product) as out:
        assert np.isnan(out[0].data[5, 5])
        assert out[1].data[5, 5] == 2
        assert np.isfinite(out[0].data[0, 0])
        assert out[1].data[0, 0] == 4


def test_saturation_excludes_pixels(tmp_path):
    rng = np.random.default_rng(1)
    raws = [rng.integers(500, 800, size=(16, 16)).astype(np.int16) for _ in range(4)]
    for raw in raws:
        raw[2, 2] = 9000
    paths = [make_flat_fits(tmp_path / f"s{i}.fits", raw) for i, raw in enumerate(raws)]
    product, _ = _run(tmp_path, paths, 4000.0)
    with fits.open(product) as out:
        assert np.isnan(out[0].data[2, 2])
        assert out[1].data[2, 2] == 0


def test_histogram_median_exact_for_even_and_odd_counts():
    rng = np.random.default_rng(7)
    for total in (1, 2, 3, 4, 100, 101):
        values = rng.integers(-1000, 1000, size=total)
        hist = np.bincount(values + 32768, minlength=65536)
        bscale, bzero = 2.0, -5.0
        got = _median_scaled_from_hist(hist, total, bscale, bzero)
        want = np.median(values.astype(np.float64) * bscale + bzero)
        assert got == pytest.approx(want)


def test_histogram_median_negative_bscale():
    values = np.array([-3, 0, 2, 2, 9], dtype=np.int32)
    hist = np.bincount(values + 32768, minlength=65536)
    bscale = -1.5
    got = _median_scaled_from_hist(hist, len(values), bscale, 7.0)
    want = np.median(values.astype(np.float64) * bscale + 7.0)
    assert got == pytest.approx(want)


def test_frame_without_valid_pixels_fails(tmp_path):
    raw = np.full((16, 16), 30000, dtype=np.int16)
    paths = [make_flat_fits(tmp_path / f"dead{i}.fits", raw) for i in range(3)]
    with pytest.raises(PreparationError, match="不存在任何有效像元"):
        build_master_flat(
            paths, [p.name for p in paths], 5000.0,
            output_dir=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


def test_truncated_file_fails_batch(tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    data = paths[1].read_bytes()
    paths[1].write_bytes(data[: len(data) // 2])
    with pytest.raises(PreparationError, match="截断|无法解析"):
        build_master_flat(
            paths, [p.name for p in paths], saturation,
            output_dir=tmp_path / "out",
        )
    out_dir = tmp_path / "out"
    assert not out_dir.exists() or not list(out_dir.iterdir())


def test_non_fits_content_fails(tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    paths[0].write_bytes(b"not a fits file at all" * 32)
    with pytest.raises(PreparationError):
        build_master_flat(
            paths, [p.name for p in paths], saturation,
            output_dir=tmp_path / "out",
        )


def test_wrong_bitpix_fails(tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    data = fits.open(paths[0])[0].data.astype(np.int32)
    fits.PrimaryHDU(data=data).writeto(paths[0], overwrite=True)
    with pytest.raises(PreparationError, match="BITPIX"):
        build_master_flat(
            paths, [p.name for p in paths], saturation,
            output_dir=tmp_path / "out",
        )


def test_compressed_hdu_fails(tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    raw = fits.open(paths[0])[0].data
    comp = fits.CompImageHDU(data=np.asarray(raw, dtype=np.int16))
    fits.HDUList([fits.PrimaryHDU(), comp]).writeto(paths[0], overwrite=True)
    with pytest.raises(PreparationError, match="压缩"):
        build_master_flat(
            paths, [p.name for p in paths], saturation,
            output_dir=tmp_path / "out",
        )


def test_shape_mismatch_fails(tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    other = np.ones((32, 32), dtype=np.int16) * 600
    make_flat_fits(paths[2], other)
    with pytest.raises(PreparationError, match="尺寸"):
        build_master_flat(
            paths, [p.name for p in paths], saturation,
            output_dir=tmp_path / "out",
        )


def test_filter_mismatch_fails(tmp_path, batch_factory):
    from astropy.io import fits as afits

    paths, _, saturation = batch_factory(n=3)
    with afits.open(paths[1], mode="update") as hdul:
        hdul[0].header["FILTER"] = "R"
    with afits.open(paths[0]) as hdul:
        hdul[0].header.setdefault("FILTER", "V")
    # 首帧补一个不同滤镜，制造 V/R 冲突。
    with afits.open(paths[0], mode="update") as hdul:
        hdul[0].header["FILTER"] = "V"
    with pytest.raises(PreparationError, match="滤镜|相机"):
        build_master_flat(
            paths, [p.name for p in paths], saturation,
            output_dir=tmp_path / "out",
        )


def test_instrument_mismatch_fails(tmp_path, batch_factory):
    from astropy.io import fits as afits

    paths, _, saturation = batch_factory(n=3)
    for i, p in enumerate(paths):
        with afits.open(p, mode="update") as hdul:
            hdul[0].header["INSTRUME"] = "CAM-A" if i % 2 == 0 else "CAM-B"
    with pytest.raises(PreparationError, match="相机|滤镜"):
        build_master_flat(
            paths, [p.name for p in paths], saturation,
            output_dir=tmp_path / "out",
        )


def test_matching_camera_filter_passes(tmp_path, batch_factory):
    from astropy.io import fits as afits

    paths, _, saturation = batch_factory(n=3)
    for p in paths:
        with afits.open(p, mode="update") as hdul:
            hdul[0].header["INSTRUME"] = "CAM-X"
            hdul[0].header["FILTER"] = "g"
    product, _ = _run(tmp_path, paths, saturation)
    with afits.open(product) as out:
        assert out[0].header["IMAGTYPE"] == "MASTERFLAT"


def test_two_image_hdus_fails(tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    raw = fits.open(paths[0])[0].data.astype(np.int16)
    hdul = fits.HDUList(
        [fits.PrimaryHDU(data=raw), fits.ImageHDU(data=raw, name="EXTRA")]
    )
    hdul.writeto(paths[0], overwrite=True)
    with pytest.raises(PreparationError, match="幅图像"):
        build_master_flat(
            paths, [p.name for p in paths], saturation,
            output_dir=tmp_path / "out",
        )


def test_frame_count_limits(tmp_path):
    raw = np.ones((8, 8), dtype=np.int16) * 600
    paths = [make_flat_fits(tmp_path / f"x{i}.fits", raw) for i in range(MIN_FRAMES - 1)]
    with pytest.raises(PreparationError, match="帧数"):
        build_master_flat(
            paths, [p.name for p in paths], 5000.0,
            output_dir=tmp_path / "out",
        )

    paths = [make_flat_fits(tmp_path / f"y{i}.fits", raw) for i in range(MAX_FRAMES + 1)]
    with pytest.raises(PreparationError, match="帧数"):
        build_master_flat(
            paths, [p.name for p in paths], 5000.0,
            output_dir=tmp_path / "out",
        )


@pytest.mark.parametrize("bad_saturation", [0.0, -1.0, float("nan"), float("inf")])
def test_bad_saturation(tmp_path, batch_factory, bad_saturation):
    paths, _, _ = batch_factory(n=3)
    with pytest.raises(PreparationError, match="饱和阈值"):
        build_master_flat(
            paths, [p.name for p in paths], bad_saturation,
            output_dir=tmp_path / "out",
        )


def test_write_failure_reports_batch_error(tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    # 输出目录内部被同名普通文件阻塞，写出必然失败；不得留下半成品。
    blocked_output = tmp_path / "blocked"
    blocked_output.mkdir()
    (blocked_output / "child").write_text("not a directory")
    with pytest.raises(PreparationError) as exc_info:
        build_master_flat(
            paths, [p.name for p in paths], saturation,
            output_dir=blocked_output / "child" / "out",
        )
    assert exc_info.value.status_code == 500
    assert not list(blocked_output.rglob("*.tmp"))


# ---------------------------------------------------------------------------
# 磁盘读取预算：每帧从暂存累计读取的像元字节数 ≤ 2 倍帧像元数据量。
# ---------------------------------------------------------------------------

def test_pixel_read_budget_is_two_multiples(tmp_path, batch_factory):
    paths, raws, saturation = batch_factory(n=7, height=128, width=96)
    _, io_stats = _run(tmp_path, paths, saturation)
    per_frame = io_stats["pixel_bytes_per_frame"]
    assert per_frame == 128 * 96 * 2
    # 第一趟整帧 + 第二趟整帧：恰好两倍，任何额外重读都会超预算。
    assert io_stats["max_pixel_bytes_read"] == per_frame * 2
    assert io_stats["total_pixel_bytes_read"] == per_frame * 2 * 7
    assert MAX_READ_MULTIPLIER == 2.0


def test_read_budget_counts_cached_rereads(tmp_path):
    """同一块像元即使命中系统缓存，重复读取也计入预算。"""
    rng = np.random.default_rng(3)
    raw = rng.integers(500, 800, size=(12, 8)).astype(np.int16)
    path = make_flat_fits(tmp_path / "f.fits", raw)
    with fits.open(path, do_not_scale_image_data=True) as hdul:
        offset = hdul[0].fileinfo()["datLoc"]
    reader = FrameReader(
        path=path, image_index=0, data_offset=offset, pixel_bytes=12 * 8 * 2
    ).open()
    try:
        reader.read_rows(0, 12, 8)
        reader.read_rows(0, 12, 8)  # 命中页缓存的重读，照样计数
        assert reader.pixel_bytes_read == 12 * 8 * 2 * 2
        with pytest.raises(PreparationError, match="读取量超过"):
            reader.read_rows(0, 1, 8)  # 第三次读取越过 2 倍上限
    finally:
        reader.close()


def test_partial_row_reads_cover_full_frame_twice(tmp_path, batch_factory):
    """行带分块读取拼起来仍只覆盖每帧两遍，且数值与全量参考一致。"""
    paths, raws, saturation = batch_factory(n=4, height=257, width=65)
    product, io_stats = _run(tmp_path, paths, saturation)
    expected_master, expected_counts = reference_master(
        raws, 1.0, 0.0, saturation, blank=None
    )
    with fits.open(product) as out:
        np.testing.assert_array_equal(out[1].data, expected_counts)
        ok = np.isfinite(expected_master)
        np.testing.assert_allclose(out[0].data[ok], expected_master[ok], rtol=1e-5)
    per_frame = io_stats["pixel_bytes_per_frame"]
    assert io_stats["max_pixel_bytes_read"] == per_frame * 2


def test_pipeline_never_copies_staged_inputs(tmp_path, batch_factory):
    """制备前后暂存文件 inode 不变，且不产生任何额外临时文件。"""
    staging = tmp_path / "staging"
    paths0, _, saturation = batch_factory(n=4)
    # 把夹具文件迁入独立暂存目录，与输出目录隔离后再做快照。
    paths = []
    staging.mkdir()
    for p in paths0:
        target = staging / p.name
        shutil.copy2(p, target)
        paths.append(target)
    before = {p: (p.stat().st_size, p.stat().st_ino) for p in paths}
    snapshot = sorted(p.relative_to(staging) for p in staging.rglob("*"))
    product, _ = _run(tmp_path, paths, saturation)
    for p, (size, ino) in before.items():
        assert p.exists()
        assert (p.stat().st_size, p.stat().st_ino) == (size, ino)
    assert sorted(p.relative_to(staging) for p in staging.rglob("*")) == snapshot
    # 成品位于输出目录，暂存侧没有出现输入副本或临时文件。
    assert product.parent == (tmp_path / "out")
    assert not list(staging.rglob("*.tmp"))


def test_tall_and_wide_shapes_match_reference(tmp_path):
    """窄高/宽矮两种极端形状覆盖行带切分边界。"""
    rng = np.random.default_rng(11)
    for height, width in ((4096, 16), (16, 4096)):
        sub = tmp_path / f"{height}x{width}"
        raws = [
            rng.integers(500, 900, size=(height, width)).astype(np.int16)
            for _ in range(3)
        ]
        paths = [
            make_flat_fits(sub / f"f{i}.fits", raw) for i, raw in enumerate(raws)
        ]
        stats: dict[str, int] = {}
        product = build_master_flat(
            paths, [p.name for p in paths], 5000.0,
            output_dir=sub / "out", io_stats=stats,
        )
        io_stats = stats
        expected_master, expected_counts = reference_master(
            raws, 1.0, 0.0, 5000.0, blank=None
        )
        with fits.open(product) as out:
            np.testing.assert_array_equal(out[1].data, expected_counts)
            ok = np.isfinite(expected_master)
            np.testing.assert_allclose(
                out[0].data[ok], expected_master[ok], rtol=1e-5
            )
        assert io_stats["max_pixel_bytes_read"] == height * width * 2 * 2
        assert MAX_DIM == 4096
