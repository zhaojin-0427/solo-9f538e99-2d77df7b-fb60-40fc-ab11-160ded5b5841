"""/prepare 与 /docs 接口测试。"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from fastapi.testclient import TestClient

from app import main
from conftest import make_flat_fits


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path / "products")
    (tmp_path / "work").mkdir()
    (tmp_path / "products").mkdir()
    with TestClient(main.app) as c:
        yield c


def _upload(client, paths, saturation=5000.0):
    files = [
        ("files", (p.name, p.read_bytes(), "application/fits")) for p in paths
    ]
    return client.post(
        "/prepare", data={"saturation": str(saturation)}, files=files
    )


def test_health_and_docs(client):
    assert client.get("/health").json() == {"status": "ok"}
    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "主平场" in docs.text
    openapi = client.get("/openapi.json").json()
    assert "/prepare" in openapi["paths"]
    post = openapi["paths"]["/prepare"]["post"]
    assert "multipart/form-data" in post["requestBody"]["content"]


def test_prepare_happy_path(client, tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=4)
    resp = _upload(client, paths, saturation)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/fits"
    assert resp.headers["x-frame-count"] == "4"
    with fits.open(io.BytesIO(resp.content)) as out:
        assert out[0].header["IMAGTYPE"] == "MASTERFLAT"
        assert out[0].header["SATURATE"] == saturation
        assert out[1].name == "COUNTS"
        assert out[1].data.shape == out[0].data.shape
    # 成品已落盘且整批输入暂存被清理。
    products = list((tmp_path / "products").glob("master_flat_*.fits"))
    assert len(products) == 1
    assert not list((tmp_path / "work").glob("batch_*"))
    assert not list((tmp_path / "work").rglob("*.tmp"))


def test_no_tempfile_created_during_upload(client, tmp_path, batch_factory, monkeypatch):
    """上传解析不得使用 SpooledTemporaryFile 或在磁盘上留任何解析临时文件。"""
    paths, _, saturation = batch_factory(n=3)
    real_mkdtemp = tempfile.mkdtemp
    mkdtemp_calls: list[str] = []

    def spy_mkdtemp(*args, **kwargs):
        result = real_mkdtemp(*args, **kwargs)
        mkdtemp_calls.append(result)
        return result

    monkeypatch.setattr(main.tempfile, "mkdtemp", spy_mkdtemp)

    class ForbiddenSpool:
        def __init__(self, *args, **kwargs):
            raise AssertionError("上传链路禁止使用 SpooledTemporaryFile 落盘")

    monkeypatch.setattr(main.tempfile, "SpooledTemporaryFile", ForbiddenSpool)

    resp = _upload(client, paths, saturation)
    assert resp.status_code == 200, resp.text

    # 每帧只有一个暂存目录、目录内恰好一个文件；请求结束后全部清除。
    assert len(mkdtemp_calls) == 1
    assert not Path(mkdtemp_calls[0]).exists()
    assert not list((tmp_path / "work").rglob("*"))


def test_each_frame_staged_at_most_once(client, tmp_path, batch_factory):
    """处理期间每帧在磁盘上恰好一份暂存文件，大小与上传内容逐字节一致。"""
    paths, raws, saturation = batch_factory(n=5)
    expected_sizes = {p.name: p.stat().st_size for p in paths}
    seen: dict[str, list[int]] = {name: [] for name in expected_sizes}

    original_build = main.build_master_flat

    def spy_build(frame_paths, source_names, saturation, output_dir, **kwargs):
        work_root = tmp_path / "work"
        staged = list(work_root.rglob("*.fits"))
        # 每帧恰好一个暂存文件，且工作目录下没有任何其他文件（无解析临时件）。
        assert len(staged) == len(frame_paths)
        all_files = [p for p in work_root.rglob("*") if p.is_file()]
        assert sorted(all_files) == sorted(staged)
        for p in staged:
            seen.setdefault(p.name, []).append(p.stat().st_size)
        return original_build(
            frame_paths, source_names, saturation, output_dir, **kwargs
        )

    main.build_master_flat = spy_build
    try:
        resp = _upload(client, paths, saturation)
    finally:
        main.build_master_flat = original_build
    assert resp.status_code == 200, resp.text
    for name, size in expected_sizes.items():
        assert seen[name] == [size]


def test_too_few_files_rejected(client, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    resp = _upload(client, paths[:2], saturation)
    assert resp.status_code == 422
    assert "帧数" in resp.json()["detail"]


def test_too_many_files_rejected(client, tmp_path):
    raw = np.ones((8, 8), dtype=np.int16) * 600
    paths = [make_flat_fits(tmp_path / f"m{i}.fits", raw) for i in range(13)]
    resp = _upload(client, paths)
    assert resp.status_code == 422
    assert "帧数" in resp.json()["detail"]


def test_bad_saturation_form_value(client, batch_factory):
    paths, _, _ = batch_factory(n=3)
    files = [("files", (p.name, p.read_bytes(), "application/fits")) for p in paths]
    resp = client.post(
        "/prepare", data={"saturation": "-3"}, files=files
    )
    assert resp.status_code == 422
    assert "饱和阈值" in resp.json()["detail"]


def test_truncated_upload_returns_422_and_no_product(client, tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    blob = paths[1].read_bytes()
    files = [
        ("files", (p.name, p.read_bytes() if i != 1 else blob[:40], "application/fits"))
        for i, p in enumerate(paths)
    ]
    resp = client.post("/prepare", data={"saturation": str(saturation)}, files=files)
    assert resp.status_code == 422
    assert list((tmp_path / "products").iterdir()) == []
    # 失败也清理本批输入暂存。
    assert not list((tmp_path / "work").rglob("*"))


def test_all_dead_frames_returns_422(client, tmp_path):
    raw = np.full((12, 12), 30000, dtype=np.int16)
    paths = [make_flat_fits(tmp_path / f"d{i}.fits", raw) for i in range(3)]
    resp = _upload(client, paths, saturation=5000.0)
    assert resp.status_code == 422
    assert "有效像元" in resp.json()["detail"]


def test_missing_saturation_field_is_422(client, batch_factory):
    paths, _, _ = batch_factory(n=3)
    files = [("files", (p.name, p.read_bytes(), "application/fits")) for p in paths]
    resp = client.post("/prepare", files=files)
    assert resp.status_code == 422
    assert "saturation" in resp.json()["detail"]


def test_non_numeric_saturation_is_422(client, batch_factory):
    paths, _, _ = batch_factory(n=3)
    files = [("files", (p.name, p.read_bytes(), "application/fits")) for p in paths]
    resp = client.post("/prepare", data={"saturation": "abc"}, files=files)
    assert resp.status_code == 422
    assert "饱和阈值" in resp.json()["detail"]


def test_non_multipart_content_type_rejected(client):
    resp = client.post("/prepare", content=b"x", headers={"content-type": "text/plain"})
    assert resp.status_code == 422
    assert "multipart" in resp.json()["detail"]


def test_malformed_multipart_body_rejected(client, batch_factory):
    paths, _, _ = batch_factory(n=3)
    boundary = "----broken"
    resp = client.post(
        "/prepare",
        content=b"this is not a valid multipart body at all",
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )
    assert resp.status_code == 422
    assert "解析失败" in resp.json()["detail"]


def test_oversized_upload_rejected_and_cleaned(client, tmp_path, batch_factory):
    paths, _, _ = batch_factory(n=3)
    big = tmp_path / "big.fits"
    # 先造一个合法小 FITS，再追加零字节把上传体量撑过单帧上限；
    # 大小拦截发生在管线校验之前，内容是否合法不影响本用例。
    make_flat_fits(big, np.ones((8, 8), dtype=np.int16) * 600)
    with big.open("ab") as handle:
        handle.write(b"\0" * main.MAX_UPLOAD_BYTES)
    files = [
        ("files", (p.name, p.read_bytes(), "application/fits")) for p in paths[:2]
    ]
    files.append(("files", (big.name, big.read_bytes(), "application/fits")))
    resp = client.post("/prepare", data={"saturation": "5000"}, files=files)
    assert resp.status_code == 422
    assert "上限" in resp.json()["detail"]
    assert not list((tmp_path / "work").rglob("*"))
