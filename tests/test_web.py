"""Flask 路由层测试（含本地样例加载）。"""

from __future__ import annotations

import io
import json

import pyarrow.parquet as pq

from app.web import app


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


# ---------- 检查点生成入口 ----------

_PROTOCOL = {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}}
_METADATA = {"metaData": {
    "id": "tbl-web", "format": {"provider": "parquet", "options": {}},
    "schemaString": "{}", "partitionColumns": [], "configuration": {},
    "createdTime": 1700000000000}}


def _commit_upload(version, actions):
    return (io.BytesIO(_commit_bytes(actions)), f"{version:020d}.json")


def test_checkpoint_upload_download_flow():
    client = app.test_client()
    data = {
        "target_version": "1",
        "commit_files": [
            _commit_upload(0, [_PROTOCOL, _METADATA,
                               {"add": {"path": "data/a.parquet",
                                        "partitionValues": {}, "size": 1,
                                        "modificationTime": 1,
                                        "dataChange": True}}]),
            _commit_upload(1, [{"add": {"path": "data/b.parquet",
                                        "partitionValues": {}, "size": 2,
                                        "modificationTime": 2,
                                        "dataChange": True}}]),
        ],
    }
    resp = client.post("/checkpoint", data=data,
                       content_type="multipart/form-data")
    assert resp.status_code == 200
    disposition = resp.headers["Content-Disposition"]
    assert "00000000000000000001.checkpoint.parquet" in disposition
    assert resp.headers["X-Checkpoint-Adds"] == "2"
    table = pq.read_table(io.BytesIO(resp.data))
    assert set(table.column_names) == {"txn", "add", "remove",
                                       "metaData", "protocol"}
    paths = sorted(a["path"] for a in
                   table.column("add").combine_chunks().to_pylist()
                   if a is not None)
    assert paths == ["data/a.parquet", "data/b.parquet"]


def test_checkpoint_gap_returns_400_without_download():
    client = app.test_client()
    data = {
        "target_version": "2",
        "commit_files": [
            _commit_upload(0, [_PROTOCOL, _METADATA]),
            _commit_upload(2, [{"add": {"path": "data/c.parquet"}}]),
        ],
    }
    resp = client.post("/checkpoint", data=data,
                       content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "attachment" not in resp.headers.get("Content-Disposition", "")
    body = resp.data
    assert "断档".encode() in body and "v1".encode() in body


def test_checkpoint_unsupported_table_explains_reason():
    client = app.test_client()
    proto_v3 = {"protocol": {"minReaderVersion": 3, "minWriterVersion": 7}}
    data = {
        "target_version": "0",
        "commit_files": [_commit_upload(0, [proto_v3, _METADATA])],
    }
    resp = client.post("/checkpoint", data=data,
                       content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "超出支持范围".encode() in resp.data


def test_sample_checkpoint_api_healthy():
    client = app.test_client()
    resp = client.post("/api/sample-checkpoint/healthy_table",
                       json={"target_version": 2})
    assert resp.status_code == 200
    assert "00000000000000000002.checkpoint.parquet" in \
        resp.headers["Content-Disposition"]
    table = pq.read_table(io.BytesIO(resp.data))
    adds = [a["path"] for a in table.column("add").combine_chunks().to_pylist()
            if a is not None]
    removes = [r["path"] for r in
               table.column("remove").combine_chunks().to_pylist()
               if r is not None]
    assert sorted(adds) == [f"data/part-{i:05d}-healthy.parquet"
                            for i in (0, 2, 3)]
    assert removes == ["data/part-00001-healthy.parquet"]


def test_sample_checkpoint_api_incident_gap_explained():
    client = app.test_client()
    resp = client.post("/api/sample-checkpoint/incident_table",
                       json={"target_version": 4})
    assert resp.status_code == 400
    err = resp.get_json()["error"]
    assert "断档" in err and "v3" in err


def test_sample_checkpoint_api_unknown_sample_404():
    resp = app.test_client().post("/api/sample-checkpoint/nope",
                                  json={"target_version": 0})
    assert resp.status_code == 404


def test_index_offers_checkpoint_generator():
    resp = app.test_client().get("/")
    assert resp.status_code == 200
    assert "生成重放起点检查点".encode() in resp.data
    assert "target_version".encode() in resp.data
