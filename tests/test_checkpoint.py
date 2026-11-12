"""checkpoint 生成逻辑测试（含与 analyzer 的端到端一致性）。

运行（仓库根目录）：
    /tmp/dvenv/bin/python -I -m pytest tests -q
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from app.analyzer import (
    STATUS_RECOVERABLE,
    AnalyzerError,
    InputFile,
    analyze,
    parse_checkpoint,
)
from app.checkpoint import build_checkpoint, checkpoint_filename

SAMPLES = Path(__file__).resolve().parent.parent / "samples"

PROTOCOL = {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}}


def metadata(table_id: str = "tbl-000", **overrides) -> dict:
    md = {
        "id": table_id,
        "format": {"provider": "parquet", "options": {}},
        "schemaString": "{\"type\":\"struct\",\"fields\":[]}",
        "partitionColumns": [],
        "configuration": {"delta.appendOnly": "false"},
        "createdTime": 1700000000000,
    }
    md.update(overrides)
    return {"metaData": md}


def commit_file(version: int, actions: list[dict]) -> InputFile:
    payload = "\n".join(json.dumps(a) for a in actions)
    return InputFile(filename=f"{version:020d}.json",
                     data=payload.encode("utf-8"))


def add(path: str, **kw) -> dict:
    action = {"path": path, "partitionValues": {}, "size": 1024,
              "modificationTime": 1700000000000, "dataChange": True}
    action.update(kw)
    return {"add": action}


def remove(path: str, **kw) -> dict:
    action = {"path": path, "deletionTimestamp": 1700000100000,
              "dataChange": True}
    action.update(kw)
    return {"remove": action}


def txn(app_id: str, version: int, **kw) -> dict:
    action = {"appId": app_id, "version": version}
    action.update(kw)
    return {"txn": action}


def f(n: int) -> str:
    return f"data/part-{n:05d}.parquet"


def read_checkpoint(data: bytes) -> dict:
    table = pq.read_table(io.BytesIO(data))
    rows = table.to_pylist()
    return {
        "adds": [r["add"] for r in rows if r["add"] is not None],
        "removes": [r["remove"] for r in rows if r["remove"] is not None],
        "txns": [r["txn"] for r in rows if r["txn"] is not None],
        "metadatas": [r["metaData"] for r in rows if r["metaData"] is not None],
        "protocols": [r["protocol"] for r in rows if r["protocol"] is not None],
        "columns": table.column_names,
    }


def manifest(paths: list[str]) -> InputFile:
    return InputFile(filename="existing_files.txt",
                     data=("\n".join(paths) + "\n").encode("utf-8"))


# ---------- 基本生成与内容 ----------

def test_filename_follows_target_version():
    assert checkpoint_filename(2) == "00000000000000000002.checkpoint.parquet"
    assert checkpoint_filename(0) == "00000000000000000000.checkpoint.parquet"
    artifact = build_checkpoint(
        [commit_file(0, [PROTOCOL, metadata(), add(f(0))])], 0)
    assert artifact.filename == "00000000000000000000.checkpoint.parquet"


def test_basic_checkpoint_contents():
    commits = [
        commit_file(0, [PROTOCOL, metadata("tbl-abc"), add(f(0)), add(f(1)),
                        txn("app-a", 0)]),
        commit_file(1, [add(f(2)), txn("app-a", 1), txn("app-b", 1)]),
        commit_file(2, [add(f(3)), remove(f(1))]),
    ]
    artifact = build_checkpoint(commits, 2)
    assert (artifact.add_count, artifact.remove_count, artifact.txn_count) == (3, 1, 2)
    assert artifact.table_id == "tbl-abc"

    ck = read_checkpoint(artifact.data)
    assert set(ck["columns"]) == {"txn", "add", "remove", "metaData", "protocol"}

    assert sorted(a["path"] for a in ck["adds"]) == [f(0), f(2), f(3)]
    assert [r["path"] for r in ck["removes"]] == [f(1)]

    # 文件动作 dataChange 一律 false，其余标准字段保留
    for a in ck["adds"]:
        assert a["dataChange"] is False
        assert a["size"] == 1024
        assert a["modificationTime"] == 1700000000000
        assert a["partitionValues"] == []
    assert ck["removes"][0]["dataChange"] is False
    assert ck["removes"][0]["deletionTimestamp"] == 1700000100000

    # 每个 appId 只留最后提交的 txn
    assert sorted((t["appId"], t["version"]) for t in ck["txns"]) == [
        ("app-a", 1), ("app-b", 1)]

    # 保留目标版生效的 protocol 与 metaData
    assert ck["protocols"] == [
        {"minReaderVersion": 1, "minWriterVersion": 2}]
    md = ck["metadatas"][0]
    assert md["id"] == "tbl-abc"
    assert md["format"]["provider"] == "parquet"
    assert md["partitionColumns"] == []
    assert md["configuration"] == [("delta.appendOnly", "false")]
    assert md["createdTime"] == 1700000000000


def test_effective_action_per_path():
    commits = [
        commit_file(0, [PROTOCOL, metadata(), add(f(0)), add(f(1)), add(f(2))]),
        commit_file(1, [remove(f(0)), remove(f(2))]),
        commit_file(2, [add(f(0))]),  # f0 重新加入：生效 add，不留墓碑
    ]
    ck = read_checkpoint(build_checkpoint(commits, 2).data)
    assert sorted(a["path"] for a in ck["adds"]) == [f(0), f(1)]
    assert [r["path"] for r in ck["removes"]] == [f(2)]


def test_tombstones_not_expired_by_time():
    # deletionTimestamp 再古老也不淘汰墓碑
    old_remove = remove(f(0), deletionTimestamp=1)
    commits = [
        commit_file(0, [PROTOCOL, metadata(), add(f(0)), add(f(1))]),
        commit_file(1, [old_remove]),
    ]
    ck = read_checkpoint(build_checkpoint(commits, 1).data)
    assert [r["path"] for r in ck["removes"]] == [f(0)]
    assert ck["removes"][0]["deletionTimestamp"] == 1


def test_optional_standard_fields_preserved():
    commits = [
        commit_file(0, [PROTOCOL, metadata(),
                        add(f(0), tags={"owner": "ops"},
                            stats="{\"numRecords\": 10}")]),
        commit_file(1, [remove(f(0), extendedFileMetadata=True, size=1024,
                               tags={"reason": "compact"})]),
    ]
    ck = read_checkpoint(build_checkpoint(commits, 1).data)
    assert ck["adds"] == []
    r = ck["removes"][0]
    assert r["extendedFileMetadata"] is True
    assert r["size"] == 1024
    assert r["tags"] == [("reason", "compact")]
    assert r["dataChange"] is False


def test_latest_protocol_and_metadata_win():
    commits = [
        commit_file(0, [PROTOCOL, metadata("tbl-old"), add(f(0))]),
        commit_file(1, [metadata("tbl-new")]),  # 同协议下元数据演进
    ]
    ck = read_checkpoint(build_checkpoint(commits, 1).data)
    assert ck["metadatas"][0]["id"] == "tbl-new"
    assert len(ck["metadatas"]) == 1 and len(ck["protocols"]) == 1


def test_empty_table_checkpoint_has_metadata_and_protocol_only():
    artifact = build_checkpoint(
        [commit_file(0, [PROTOCOL, metadata()])], 0)
    ck = read_checkpoint(artifact.data)
    assert ck["adds"] == [] and ck["removes"] == [] and ck["txns"] == []
    assert len(ck["metadatas"]) == 1 and len(ck["protocols"]) == 1


# ---------- 断档 / 无效输入 ----------

def test_commit_gap_rejected_with_missing_versions():
    commits = [
        commit_file(0, [PROTOCOL, metadata(), add(f(0))]),
        commit_file(1, [add(f(1))]),
        commit_file(3, [add(f(3))]),
    ]
    with pytest.raises(AnalyzerError, match="断档") as excinfo:
        build_checkpoint(commits, 3)
    assert "v2" in str(excinfo.value)


def test_missing_commit_zero_is_gap():
    with pytest.raises(AnalyzerError, match="断档"):
        build_checkpoint([commit_file(1, [add(f(0))])], 1)


def test_commit_beyond_target_rejected():
    commits = [
        commit_file(0, [PROTOCOL, metadata(), add(f(0))]),
        commit_file(1, [add(f(1))]),
    ]
    with pytest.raises(AnalyzerError, match="高于目标版号"):
        build_checkpoint(commits, 0)


def test_duplicate_version_rejected():
    c = commit_file(0, [PROTOCOL, metadata(), add(f(0))])
    with pytest.raises(AnalyzerError, match="重复上传"):
        build_checkpoint([c, c], 0)


def test_non_commit_file_rejected():
    weird = InputFile(filename="00000000000000000000.checkpoint.parquet",
                      data=b"x")
    with pytest.raises(AnalyzerError, match="JSON 提交"):
        build_checkpoint([weird], 0)
    with pytest.raises(AnalyzerError, match="JSON 提交"):
        build_checkpoint([InputFile(filename="notes.txt", data=b"x")], 0)


def test_invalid_json_rejected():
    bad = InputFile(filename=f"{0:020d}.json", data=b"{not json")
    with pytest.raises(AnalyzerError, match="合法 JSON"):
        build_checkpoint([bad], 0)


def test_no_files_and_bad_target_rejected():
    with pytest.raises(AnalyzerError, match="至少上传"):
        build_checkpoint([], 0)
    with pytest.raises(AnalyzerError, match="负数"):
        build_checkpoint([commit_file(0, [])], -1)


# ---------- 支持范围 ----------

def test_partitioned_table_rejected_two_ways():
    # add 带非空 partitionValues
    commits = [commit_file(0, [PROTOCOL, metadata(),
                               add("date=2024-01-01/" + f(0),
                                   partitionValues={"date": "2024-01-01"})])]
    with pytest.raises(AnalyzerError, match="无分区"):
        build_checkpoint(commits, 0)
    # metaData 声明分区列
    commits = [commit_file(0, [PROTOCOL, metadata(partitionColumns=["date"]),
                               add(f(0))])]
    with pytest.raises(AnalyzerError, match="无分区"):
        build_checkpoint(commits, 0)


def test_deletion_vector_rejected():
    dv = {"storageType": "u", "pathOrInlineDv": "abc"}
    with pytest.raises(AnalyzerError, match="删除向量"):
        build_checkpoint(
            [commit_file(0, [PROTOCOL, metadata(), add(f(0), deletionVector=dv)])],
            0)
    with pytest.raises(AnalyzerError, match="删除向量"):
        build_checkpoint(
            [commit_file(0, [PROTOCOL, metadata(), add(f(0))]),
             commit_file(1, [remove(f(0), deletionVector=dv)])], 1)


def test_dv_protocol_feature_rejected():
    proto_dv = {"protocol": {"minReaderVersion": 3, "minWriterVersion": 7,
                             "readerFeatures": ["deletionVectors"],
                             "writerFeatures": ["deletionVectors"]}}
    with pytest.raises(AnalyzerError, match="deletionVectors"):
        build_checkpoint(
            [commit_file(0, [proto_dv, metadata(), add(f(0))])], 0)


def test_unsupported_protocol_version_rejected():
    proto_new = {"protocol": {"minReaderVersion": 1, "minWriterVersion": 3}}
    with pytest.raises(AnalyzerError, match="minReaderVersion=1"):
        build_checkpoint(
            [commit_file(0, [proto_new, metadata(), add(f(0))])], 0)
    # 目标版生效协议才作数：v0 是 (1,2)，v1 升级到 (3,7) 同样拒绝
    proto_v3 = {"protocol": {"minReaderVersion": 3, "minWriterVersion": 7}}
    with pytest.raises(AnalyzerError, match="超出支持范围"):
        build_checkpoint(
            [commit_file(0, [PROTOCOL, metadata(), add(f(0))]),
             commit_file(1, [proto_v3])], 1)


def test_missing_protocol_or_metadata_rejected():
    with pytest.raises(AnalyzerError, match="protocol"):
        build_checkpoint([commit_file(0, [metadata(), add(f(0))])], 0)
    with pytest.raises(AnalyzerError, match="metaData"):
        build_checkpoint([commit_file(0, [PROTOCOL, add(f(0))])], 0)


# ---------- 与 analyzer 的端到端一致性 ----------

def _healthy_sample_commits() -> list[InputFile]:
    log_dir = SAMPLES / "healthy_table" / "_delta_log"
    return [
        InputFile(filename=p.name, data=p.read_bytes())
        for p in sorted(log_dir.glob("*.json"))
    ]


def test_generated_checkpoint_matches_repo_checkpoint():
    """生成的 v2 检查点与仓库自带 v2 检查点的快照内容一致。"""
    commits = _healthy_sample_commits()
    artifact = build_checkpoint(commits[:3], 2)

    generated = parse_checkpoint(2, artifact.data, artifact.filename)
    assert generated.dv_detected is False
    repo_ckpt = (SAMPLES / "healthy_table" / "_delta_log"
                 / "00000000000000000002.checkpoint.parquet")
    reference = parse_checkpoint(2, repo_ckpt.read_bytes(), repo_ckpt.name)
    assert sorted(generated.adds) == sorted(reference.adds)
    assert sorted(generated.tombstones) == sorted(reference.tombstones)


def test_checkpoint_plus_later_commits_matches_replay_from_zero():
    """核心验收：检查点 + 后一版起的提交 == 从 0 版重放。"""
    commits = _healthy_sample_commits()
    artifact = build_checkpoint(commits[:3], 2)
    ckpt_file = InputFile(filename=artifact.filename, data=artifact.data)
    existing = manifest([f"data/part-{i:05d}-healthy.parquet"
                         for i in range(6)])

    full = analyze(commits, [existing], 0, 4)
    via_ckpt = analyze([ckpt_file] + commits[3:], [existing], 2, 4)

    full_refs = {v["version"]: v["referenced_files"] for v in full["versions"]}
    for report in via_ckpt["versions"]:
        assert report["base"] == "checkpoint-2"
        assert report["replay_complete"] is True
        assert report["status"] == STATUS_RECOVERABLE
        assert report["referenced_files"] == full_refs[report["version"]]
