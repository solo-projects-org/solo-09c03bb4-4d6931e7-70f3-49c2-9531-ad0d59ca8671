"""Flask 路由层测试（含本地样例加载与检查点生成下载）。"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from app.web import SAMPLE_ROOT, app


def _commit_bytes(actions):
    return ("\n".join(json.dumps(a) for a in actions)).encode("utf-8")


def test_index_renders():
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Delta Lake 版本可恢复性分析".encode() in resp.data


def test_upload_analyze_flow():
    client = app.test_client()
    data = {
        "from_version": "0",
        "to_version": "1",
        "log_files": [
            (io.BytesIO(_commit_bytes([{"add": {"path": "data/a.parquet"}}])),
             "00000000000000000000.json"),
            (io.BytesIO(_commit_bytes([{"add": {"path": "data/b.parquet"}}])),
             "00000000000000000001.json"),
        ],
        "manifest_files": [
            (io.BytesIO(b"data/a.parquet\ndata/b.parquet\n"), "existing.txt"),
        ],
    }
    resp = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert "可重建 · 文件齐全".encode() in resp.data


def test_upload_missing_file_reports_400():
    client = app.test_client()
    resp = client.post("/analyze", data={
        "from_version": "0", "to_version": "1",
    }, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "至少上传".encode() in resp.data


def test_sample_api_healthy_and_incident():
    client = app.test_client()
    for name, expected_recoverable in (("healthy_table", 5),
                                       ("incident_table", None)):
        resp = client.post(f"/api/analyze-sample/{name}")
        assert resp.status_code == 200, resp.data
        report = resp.get_json()
        assert report["range"] == {"from": 0, "to": (4 if name == "healthy_table" else 5)}
        if expected_recoverable:
            assert report["summary"]["recoverable"] == expected_recoverable


def test_unknown_sample_404():
    resp = app.test_client().post("/api/analyze-sample/nope")
    assert resp.status_code == 404


def test_sample_api_list():
    resp = app.test_client().get("/api/samples")
    assert resp.status_code == 200
    names = resp.get_json()
    assert "healthy_table" in names and "incident_table" in names


# ---------- 检查点生成与下载 ----------

_PROTOCOL = {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}}
_METADATA = {"metaData": {
    "id": "t", "name": None, "description": None,
    "format": {"provider": "parquet", "options": {}},
    "schemaString": "{}", "partitionColumns": [],
    "configuration": {}, "createdTime": 1700000000000,
}}


def _non_null(table, column):
    return [r for r in table.column(column).combine_chunks().to_pylist() if r]


def test_checkpoint_upload_download():
    client = app.test_client()
    data = {
        "target_version": "1",
        "commit_files": [
            (io.BytesIO(_commit_bytes([_PROTOCOL, _METADATA,
                                       {"add": {"path": "data/a.parquet",
                                                "partitionValues": {},
                                                "size": 1,
                                                "modificationTime": 1,
                                                "dataChange": True}}])),
             "00000000000000000000.json"),
            (io.BytesIO(_commit_bytes([{"remove": {"path": "data/a.parquet",
                                                   "deletionTimestamp": 2,
                                                   "dataChange": True}}])),
             "00000000000000000001.json"),
        ],
    }
    resp = client.post("/checkpoint", data=data,
                       content_type="multipart/form-data")
    assert resp.status_code == 200
    disposition = resp.headers["Content-Disposition"]
    assert "attachment" in disposition
    assert "00000000000000000001.checkpoint.parquet" in disposition
    table = pq.read_table(io.BytesIO(resp.data))
    assert _non_null(table, "add") == []
    assert [r["path"] for r in _non_null(table, "remove")] == ["data/a.parquet"]


def test_checkpoint_upload_gap_returns_400_with_reason():
    client = app.test_client()
    data = {
        "target_version": "2",
        "commit_files": [
            (io.BytesIO(_commit_bytes([_PROTOCOL, _METADATA])),
             "00000000000000000000.json"),
            # 缺 v1，但目标版本是 2
            (io.BytesIO(_commit_bytes([])), "00000000000000000002.json"),
        ],
    }
    resp = client.post("/checkpoint", data=data,
                       content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "断档".encode() in resp.data
    assert "v1".encode() in resp.data


def test_checkpoint_upload_missing_files_returns_400():
    client = app.test_client()
    resp = client.post("/checkpoint", data={"target_version": "0"},
                       content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "没有可识别".encode() in resp.data


def test_checkpoint_sample_api_download():
    client = app.test_client()
    resp = client.post("/api/checkpoint-sample/healthy_table",
                       json={"target_version": 2})
    assert resp.status_code == 200
    assert "00000000000000000002.checkpoint.parquet" in \
        resp.headers["Content-Disposition"]
    table = pq.read_table(io.BytesIO(resp.data))
    adds = {r["path"] for r in _non_null(table, "add")}
    assert adds == {"data/part-00000-healthy.parquet",
                    "data/part-00002-healthy.parquet",
                    "data/part-00003-healthy.parquet"}
    # 样例 v2 的 remove（part-00001）成为墓碑
    tombstones = {r["path"] for r in _non_null(table, "remove")}
    assert tombstones == {"data/part-00001-healthy.parquet"}
    # 保留样例表的 metaData 与 txn
    assert _non_null(table, "metaData")[0]["id"] == "healthy-orders"
    assert _non_null(table, "txn")[0]["appId"] == "ingest"


def test_checkpoint_sample_gap_and_unknown_sample():
    client = app.test_client()
    # 事故样例 v3 提交丢失：目标版 4 必断档
    resp = client.post("/api/checkpoint-sample/incident_table",
                       json={"target_version": 4})
    assert resp.status_code == 400
    assert "v3" in resp.get_json()["error"]
    resp = client.post("/api/checkpoint-sample/nope",
                       json={"target_version": 0})
    assert resp.status_code == 404


def test_sample_checkpoint_roundtrip_through_analyze_page():
    """样例下载的检查点 + 后一版提交交给核查页，结论与从 0 版重放一致。"""
    client = app.test_client()
    resp = client.post("/api/checkpoint-sample/healthy_table",
                       json={"target_version": 2})
    assert resp.status_code == 200
    checkpoint_bytes = resp.data

    log_dir = SAMPLE_ROOT / "healthy_table" / "_delta_log"
    later = [log_dir / f"{v:020d}.json" for v in (3, 4)]
    manifest_path = SAMPLE_ROOT / "healthy_table" / "existing_files.txt"

    data = {
        "from_version": "2",
        "to_version": "4",
        "log_files": [
            (io.BytesIO(checkpoint_bytes),
             "00000000000000000002.checkpoint.parquet"),
            *[(io.BytesIO(p.read_bytes()), p.name) for p in later],
        ],
        "manifest_files": [
            (io.BytesIO(manifest_path.read_bytes()), manifest_path.name),
        ],
    }
    resp = client.post("/analyze", data=data,
                       content_type="multipart/form-data")
    assert resp.status_code == 200
    # 无需补传 v0–v2 提交，v2–v4 全部可重建且文件齐全
    html = resp.data.decode()
    marker = '<script type="application/json" id="report-data">'
    report = json.loads(html.split(marker)[1].split("</script>")[0])
    assert [v["status"] for v in report["versions"]] == ["recoverable"] * 3
    assert all(v["base"] == "checkpoint-2" for v in report["versions"])
    assert all(not v["missing_files"] and not v["missing_commits"]
               for v in report["versions"])
    # 与从 0 版重放的文件引用一致
    assert report["versions"][0]["referenced_files"] == [
        "data/part-00000-healthy.parquet",
        "data/part-00002-healthy.parquet",
        "data/part-00003-healthy.parquet",
    ]
    assert report["versions"][2]["referenced_files"] == [
        "data/part-00000-healthy.parquet",
        "data/part-00003-healthy.parquet",
        "data/part-00004-healthy.parquet",
        "data/part-00005-healthy.parquet",
    ]
