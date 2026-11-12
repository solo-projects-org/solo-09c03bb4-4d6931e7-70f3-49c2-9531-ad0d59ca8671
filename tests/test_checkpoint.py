"""checkpoint 生成逻辑测试：重放规则、能力边界与错误提示。

运行（仓库根目录）：
    /tmp/dvenv/bin/python -I -m pytest tests -q
"""

from __future__ import annotations

import io
import json

import pyarrow.parquet as pq
import pytest

from app.analyzer import (
    AnalyzerError,
    InputFile,
    STATUS_RECOVERABLE,
    analyze,
    parse_checkpoint,
)
from app.checkpoint import build_checkpoint


# ---------- 构造工具 ----------

PROTOCOL = {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}}
METADATA = {"metaData": {
    "id": "orders-table",
    "name": "orders",
    "description": "订单表",
    "format": {"provider": "parquet", "options": {}},
    "schemaString": '{"type":"struct","fields":[]}',
    "partitionColumns": [],
    "configuration": {},
    "createdTime": 1700000000000,
}}


def commit_file(version: int, actions: list[dict]) -> InputFile:
    payload = "\n".join(json.dumps(a) for a in actions)
    return InputFile(filename=f"{version:020d}.json",
                     data=payload.encode("utf-8"))


def add(path: str, **extra) -> dict:
    action = {"add": {"path": path, "partitionValues": {}, "size": 1024,
                      "modificationTime": 1700000000000, "dataChange": True}}
    action["add"].update(extra)
    return action


def remove(path: str, **extra) -> dict:
    action = {"remove": {"path": path, "deletionTimestamp": 1700000100000,
                         "dataChange": True}}
    action["remove"].update(extra)
    return action


def txn(app_id: str, version: int) -> dict:
    return {"txn": {"appId": app_id, "version": version,
                    "lastUpdated": 1700000200000}}


def f(n: int) -> str:
    return f"data/part-{n:05d}.parquet"


def manifest(paths: list[str]) -> InputFile:
    return InputFile(filename="existing_files.txt",
                     data=("\n".join(paths) + "\n").encode("utf-8"))


def basic_commits() -> list[InputFile]:
    return [
        commit_file(0, [PROTOCOL, METADATA, add(f(0)), add(f(1))]),
        commit_file(1, [txn("ingest", 1), add(f(2))]),
        commit_file(2, [add(f(3)), remove(f(1))]),
        commit_file(3, [txn("ingest", 2), add(f(4))]),
        commit_file(4, [add(f(5)), remove(f(2))]),
    ]


def non_null_rows(table, column: str) -> list[dict]:
    return [row for row in table.column(column).combine_chunks().to_pylist()
            if row is not None]


# ---------- 生成内容与重放规则 ----------

def test_build_checkpoint_contents_and_download_name():
    data, name = build_checkpoint(basic_commits()[:3], 2)
    assert name == f"{2:020d}.checkpoint.parquet"

    table = pq.read_table(io.BytesIO(data))
    assert set(table.column_names) == {"txn", "add", "remove",
                                       "metaData", "protocol"}

    adds = non_null_rows(table, "add")
    assert {a["path"] for a in adds} == {f(0), f(2), f(3)}
    # dataChange 一律写为 false，其余标准字段保留
    assert all(a["dataChange"] is False for a in adds)
    assert all(a["size"] == 1024 for a in adds)
    assert all(a["modificationTime"] == 1700000000000 for a in adds)
    # PyArrow map 列读回为键值对列表，空 map 即空列表
    assert all(dict(a["partitionValues"]) == {} for a in adds)

    # f1 的 remove 是墓碑：保留且不按时间淘汰
    removes = non_null_rows(table, "remove")
    assert [r["path"] for r in removes] == [f(1)]
    assert removes[0]["dataChange"] is False
    assert removes[0]["deletionTimestamp"] == 1700000100000

    # 保留目标版生效的 protocol 与 metaData
    assert non_null_rows(table, "protocol") == [
        {"minReaderVersion": 1, "minWriterVersion": 2}]
    metadata = non_null_rows(table, "metaData")
    assert len(metadata) == 1
    assert metadata[0]["id"] == "orders-table"
    assert metadata[0]["name"] == "orders"
    assert metadata[0]["format"]["provider"] == "parquet"
    assert dict(metadata[0]["format"]["options"]) == {}
    assert metadata[0]["partitionColumns"] == []
    assert metadata[0]["schemaString"] == METADATA["metaData"]["schemaString"]

    # 每个 appId 只留最后提交的 txn
    txns = non_null_rows(table, "txn")
    assert txns == [{"appId": "ingest", "version": 1,
                     "lastUpdated": 1700000200000}]


def test_txn_keeps_last_per_app_id():
    commits = [
        commit_file(0, [PROTOCOL, METADATA, txn("a", 1), txn("b", 5)]),
        commit_file(1, [txn("a", 2)]),
    ]
    data, _ = build_checkpoint(commits, 1)
    txns = non_null_rows(pq.read_table(io.BytesIO(data)), "txn")
    by_app = {t["appId"]: t["version"] for t in txns}
    assert by_app == {"a": 2, "b": 5}


def test_metadata_and_protocol_of_target_version_win():
    upgraded_meta = {"metaData": {**METADATA["metaData"],
                                  "configuration": {"delta.appendOnly": "true"}}}
    commits = [
        commit_file(0, [PROTOCOL, METADATA, add(f(0))]),
        commit_file(1, [upgraded_meta]),
    ]
    data, _ = build_checkpoint(commits, 1)
    metadata = non_null_rows(pq.read_table(io.BytesIO(data)), "metaData")
    assert dict(metadata[0]["configuration"]) == {"delta.appendOnly": "true"}


def test_last_action_per_path_wins():
    commits = [
        commit_file(0, [PROTOCOL, METADATA, add(f(0)), add(f(1))]),
        commit_file(1, [remove(f(0))]),          # f0 变墓碑
        commit_file(2, [add(f(1), size=2048)]),  # f1 重加，覆盖旧 add
    ]
    data, _ = build_checkpoint(commits, 2)
    table = pq.read_table(io.BytesIO(data))
    adds = non_null_rows(table, "add")
    assert [(a["path"], a["size"]) for a in adds] == [(f(1), 2048)]
    removes = non_null_rows(table, "remove")
    assert [r["path"] for r in removes] == [f(0)]


def test_remove_then_readd_in_same_commit_keeps_add():
    commits = [
        commit_file(0, [PROTOCOL, METADATA, add(f(0))]),
        commit_file(1, [remove(f(0)), add(f(0))]),
    ]
    data, _ = build_checkpoint(commits, 1)
    table = pq.read_table(io.BytesIO(data))
    assert [a["path"] for a in non_null_rows(table, "add")] == [f(0)]
    assert non_null_rows(table, "remove") == []


def test_tombstones_not_expired_by_time():
    commits = [
        commit_file(0, [PROTOCOL, METADATA, add(f(0))]),
        # deletionTimestamp 极早也不淘汰
        commit_file(1, [remove(f(0), deletionTimestamp=1)]),
    ]
    data, _ = build_checkpoint(commits, 1)
    removes = non_null_rows(pq.read_table(io.BytesIO(data)), "remove")
    assert [r["path"] for r in removes] == [f(0)]
    assert removes[0]["deletionTimestamp"] == 1


def test_extra_commits_beyond_target_ignored():
    data, name = build_checkpoint(basic_commits(), 1)
    assert name == f"{1:020d}.checkpoint.parquet"
    adds = non_null_rows(pq.read_table(io.BytesIO(data)), "add")
    assert {a["path"] for a in adds} == {f(0), f(1), f(2)}


def test_empty_commit_keeps_continuity():
    commits = [
        commit_file(0, [PROTOCOL, METADATA, add(f(0))]),
        commit_file(1, []),  # 空提交仍是合法的一环
    ]
    data, _ = build_checkpoint(commits, 1)
    adds = non_null_rows(pq.read_table(io.BytesIO(data)), "add")
    assert [a["path"] for a in adds] == [f(0)]


def test_non_commit_files_ignored():
    logs = basic_commits()[:2] + [
        InputFile(filename="_last_checkpoint", data=b"{}"),
        InputFile(filename=f"{1:020d}.checkpoint.parquet", data=b"junk"),
        InputFile(filename="note.txt", data=b"hi"),
    ]
    data, name = build_checkpoint(logs, 1)
    assert name == f"{1:020d}.checkpoint.parquet"


def test_generated_checkpoint_parses_with_analyzer():
    data, name = build_checkpoint(basic_commits()[:3], 2)
    parsed = parse_checkpoint(2, data, name)
    assert parsed.adds == sorted([f(0), f(2), f(3)])
    assert parsed.tombstones == [f(1)]
    assert parsed.dv_detected is False


def test_generated_checkpoint_matches_replay_from_zero():
    """生成的检查点 + 后续提交，核查结果应与从 0 版重放一致。"""
    commits = basic_commits()
    data, name = build_checkpoint(commits[:3], 2)
    ckpt = InputFile(filename=name, data=data)
    existing = manifest([f(i) for i in range(6)])

    full = analyze(commits, [existing], 0, 4)
    via_ckpt = analyze([ckpt] + commits[3:], [existing], 2, 4)

    full_refs = {v["version"]: v["referenced_files"] for v in full["versions"]}
    for report in via_ckpt["versions"]:
        assert report["status"] == STATUS_RECOVERABLE
        assert report["base"] == "checkpoint-2"
        assert report["referenced_files"] == full_refs[report["version"]]


# ---------- 断档 / 无效日志 / 超出支持范围 ----------

def test_missing_commit_gap_rejected():
    commits = [c for c in basic_commits() if not c.filename.startswith("00000000000000000003")]
    with pytest.raises(AnalyzerError, match="断档.*v3"):
        build_checkpoint(commits, 4)


def test_missing_commit_zero_rejected():
    with pytest.raises(AnalyzerError, match="断档.*v0"):
        build_checkpoint(basic_commits()[1:], 2)


def test_no_recognized_commits_rejected():
    weird = InputFile(filename="README.md", data=b"hi")
    with pytest.raises(AnalyzerError, match="没有可识别"):
        build_checkpoint([weird], 0)
    with pytest.raises(AnalyzerError, match="没有可识别"):
        build_checkpoint([], 0)


def test_invalid_json_rejected():
    bad = InputFile(filename=f"{0:020d}.json", data=b"{not json")
    with pytest.raises(AnalyzerError, match="日志无效|既不是 ndjson"):
        build_checkpoint([bad], 0)


def test_conflicting_duplicate_commit_rejected():
    c0 = basic_commits()[0]
    other = commit_file(0, [PROTOCOL, METADATA, add(f(9))])
    with pytest.raises(AnalyzerError, match="重复上传且内容不一致"):
        build_checkpoint([c0, other], 0)
    # 内容相同的重复上传可以容忍
    data, _ = build_checkpoint([c0, InputFile(c0.filename, c0.data)], 0)
    assert data


def test_missing_protocol_or_metadata_rejected():
    with pytest.raises(AnalyzerError, match="protocol"):
        build_checkpoint([commit_file(0, [METADATA, add(f(0))])], 0)
    with pytest.raises(AnalyzerError, match="metaData"):
        build_checkpoint([commit_file(0, [PROTOCOL, add(f(0))])], 0)


def test_target_version_validation():
    with pytest.raises(AnalyzerError, match="负数"):
        build_checkpoint(basic_commits(), -1)
    with pytest.raises(AnalyzerError, match="整数"):
        build_checkpoint(basic_commits(), "2")


def test_partitioned_table_rejected():
    partitioned_meta = {"metaData": {**METADATA["metaData"],
                                     "partitionColumns": ["dt"]}}
    commits = [commit_file(0, [PROTOCOL, partitioned_meta, add(f(0))])]
    with pytest.raises(AnalyzerError, match="分区"):
        build_checkpoint(commits, 0)

    commits = [commit_file(0, [PROTOCOL, METADATA,
                               add(f(0), partitionValues={"dt": "2026-01-01"})])]
    with pytest.raises(AnalyzerError, match="分区"):
        build_checkpoint(commits, 0)


def test_deletion_vector_rejected():
    dv = {"storageType": "u", "pathOrInlineDv": "abc"}
    with pytest.raises(AnalyzerError, match="删除向量"):
        build_checkpoint([commit_file(0, [PROTOCOL, METADATA,
                                          add(f(0), deletionVector=dv)])], 0)
    with pytest.raises(AnalyzerError, match="删除向量"):
        build_checkpoint([commit_file(0, [PROTOCOL, METADATA, add(f(0))]),
                          commit_file(1, [remove(f(0), deletionVector=dv)])], 1)


def test_deletion_vector_protocol_feature_rejected():
    proto_dv = {"protocol": {"minReaderVersion": 3, "minWriterVersion": 7,
                             "readerFeatures": ["deletionVectors"],
                             "writerFeatures": ["deletionVectors"]}}
    with pytest.raises(AnalyzerError, match="deletionVectors"):
        build_checkpoint([commit_file(0, [proto_dv, METADATA, add(f(0))])], 0)


def test_unsupported_protocol_version_rejected():
    proto_v2 = {"protocol": {"minReaderVersion": 2, "minWriterVersion": 2}}
    with pytest.raises(AnalyzerError, match="超出支持范围"):
        build_checkpoint([commit_file(0, [proto_v2, METADATA, add(f(0))])], 0)

    proto_w5 = {"protocol": {"minReaderVersion": 1, "minWriterVersion": 5}}
    with pytest.raises(AnalyzerError, match="超出支持范围"):
        build_checkpoint([commit_file(0, [proto_w5, METADATA, add(f(0))])], 0)

    # 目标版生效协议被后续提交升级，同样拒绝
    commits = [
        commit_file(0, [PROTOCOL, METADATA, add(f(0))]),
        commit_file(1, [{"protocol": {"minReaderVersion": 1,
                                      "minWriterVersion": 4}}]),
    ]
    with pytest.raises(AnalyzerError, match="超出支持范围"):
        build_checkpoint(commits, 1)


def test_unsupported_protocol_feature_rejected():
    proto_feature = {"protocol": {"minReaderVersion": 3, "minWriterVersion": 7,
                                  "readerFeatures": ["columnMapping"],
                                  "writerFeatures": ["columnMapping"]}}
    with pytest.raises(AnalyzerError, match="超出支持范围"):
        build_checkpoint([commit_file(0, [proto_feature, METADATA,
                                          add(f(0))])], 0)


def test_malformed_actions_rejected():
    with pytest.raises(AnalyzerError, match="路径不是字符串"):
        build_checkpoint([commit_file(0, [PROTOCOL, METADATA,
                                          {"add": {"size": 1}}])], 0)
    with pytest.raises(AnalyzerError, match="日志无效"):
        build_checkpoint([commit_file(0, [PROTOCOL, METADATA,
                                          {"txn": {"version": 1}}])], 0)
    with pytest.raises(AnalyzerError, match="日志无效"):
        build_checkpoint([commit_file(0, [PROTOCOL, METADATA,
                                          add(f(0), size="big")])], 0)
