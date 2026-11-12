"""从 JSON 提交生成经典单文件 Parquet 检查点（重放起点）。

订单表的旧提交转入冷存储前，值班员在网页上填写目标版号，并上传从 0 版
到该版的 Delta Lake JSON 提交（无需数据文件）。本模块连续重放这些提交，
用 PyArrow 生成按目标版号命名的 ``NNN.checkpoint.parquet``：

- 只处理无分区、无删除向量、协议为 minReaderVersion=1、
  minWriterVersion=2 的表；
- 保留目标版生效的 protocol 与 metaData；
- 每条路径只写生效的 add 或 remove（墓碑不按时间淘汰）；
- 文件动作的 dataChange 一律写为 false，其余标准字段保留；
- 每个 appId 只保留最后提交的 txn。

提交断档、日志无效或超出支持范围时抛出 AnalyzerError，网页据此说明
原因且不提供下载。
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .analyzer import (
    DV_FEATURE_NAME,
    AnalyzerError,
    InputFile,
    _decode_utf8,
    _extract_protocol_features,
    _iter_json_objects,
    normalize_data_path,
)

# Delta 日志提交文件名：20 位补零版本号 + .json
_COMMIT_RE = re.compile(r"^(\d{20})\.json$")

# 本工具唯一支持的表协议
SUPPORTED_PROTOCOL = (1, 2)

# ---- 经典单文件检查点列结构（protocol (1,2)，无删除向量相关列） ----
TXN_TYPE = pa.struct([
    ("appId", pa.string()),
    ("version", pa.int64()),
    ("lastUpdated", pa.int64()),
])
ADD_TYPE = pa.struct([
    ("path", pa.string()),
    ("partitionValues", pa.map_(pa.string(), pa.string())),
    ("size", pa.int64()),
    ("modificationTime", pa.int64()),
    ("dataChange", pa.bool_()),
    ("tags", pa.map_(pa.string(), pa.string())),
    ("stats", pa.string()),
])
REMOVE_TYPE = pa.struct([
    ("path", pa.string()),
    ("deletionTimestamp", pa.int64()),
    ("dataChange", pa.bool_()),
    ("extendedFileMetadata", pa.bool_()),
    ("partitionValues", pa.map_(pa.string(), pa.string())),
    ("size", pa.int64()),
    ("tags", pa.map_(pa.string(), pa.string())),
])
METADATA_TYPE = pa.struct([
    ("id", pa.string()),
    ("name", pa.string()),
    ("description", pa.string()),
    ("format", pa.struct([
        ("provider", pa.string()),
        ("options", pa.map_(pa.string(), pa.string())),
    ])),
    ("schemaString", pa.string()),
    ("partitionColumns", pa.list_(pa.string())),
    ("configuration", pa.map_(pa.string(), pa.string())),
    ("createdTime", pa.int64()),
])
PROTOCOL_TYPE = pa.struct([
    ("minReaderVersion", pa.int32()),
    ("minWriterVersion", pa.int32()),
])


@dataclass
class CommitActions:
    """一个 JSON 提交中与检查点相关的全部动作（保持文件内顺序）。"""

    version: int
    source: str
    file_actions: list[tuple[str, dict]] = field(default_factory=list)
    protocol: dict | None = None
    metadata: dict | None = None
    txns: list[dict] = field(default_factory=list)


@dataclass
class ReplayState:
    """从 0 版连续重放到目标版号后的表状态。"""

    active: dict[str, dict] = field(default_factory=dict)      # 归一化路径 -> 生效 add
    tombstones: dict[str, dict] = field(default_factory=dict)  # 归一化路径 -> 生效 remove
    txns: dict[str, dict] = field(default_factory=dict)        # appId -> 最后的 txn
    protocol: dict | None = None
    metadata: dict | None = None


@dataclass
class CheckpointArtifact:
    """生成完毕、可供下载的检查点。"""

    filename: str
    data: bytes
    target_version: int
    add_count: int
    remove_count: int
    txn_count: int
    table_id: str | None


def checkpoint_filename(target_version: int) -> str:
    return f"{target_version:020d}.checkpoint.parquet"


# ---------- 提交解析 ----------


def _guard_action(action: dict, kind: str, source: str) -> None:
    """单个 add / remove 动作的能力检查：无删除向量、无分区。"""
    if action.get("deletionVector") is not None:
        raise AnalyzerError(
            f"提交 {source} 的 {kind} 动作带有 deletionVector："
            "本工具只处理无删除向量的表"
        )
    if action.get("partitionValues"):
        raise AnalyzerError(
            f"提交 {source} 的 {kind} 动作带有非空 partitionValues："
            "本工具只处理无分区表"
        )


def parse_commit_actions(version: int, data: bytes, source: str) -> CommitActions:
    """解析一个 JSON 提交，保留检查点所需的全部动作。"""
    parsed = CommitActions(version=version, source=source)
    text = _decode_utf8(data, source)
    for obj in _iter_json_objects(text, source):
        if not isinstance(obj, dict):
            continue
        if isinstance(obj.get("add"), dict):
            _guard_action(obj["add"], "add", source)
            parsed.file_actions.append(("add", obj["add"]))
        if isinstance(obj.get("remove"), dict):
            _guard_action(obj["remove"], "remove", source)
            parsed.file_actions.append(("remove", obj["remove"]))
        if isinstance(obj.get("protocol"), dict):
            features = _extract_protocol_features(obj["protocol"])
            if DV_FEATURE_NAME in features:
                raise AnalyzerError(
                    f"提交 {source} 的 protocol 特性包含 {DV_FEATURE_NAME}："
                    "本工具只处理无删除向量的表"
                )
            parsed.protocol = obj["protocol"]
        if isinstance(obj.get("metaData"), dict):
            parsed.metadata = obj["metaData"]
        if isinstance(obj.get("txn"), dict):
            parsed.txns.append(obj["txn"])
        # commitInfo / cdc / domainMetadata 等动作不进检查点，忽略
    return parsed


def _classify_and_parse(commit_files: list[InputFile],
                        target_version: int) -> list[CommitActions]:
    """校验上传清单并返回 0..target 连续排列的提交。"""
    by_version: dict[int, CommitActions] = {}
    for item in commit_files:
        name = item.filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        match = _COMMIT_RE.match(name)
        if not match:
            raise AnalyzerError(
                f"{name} 不是 20 位版本号命名的 JSON 提交"
                "（形如 00000000000000000003.json）；生成检查点只需上传 "
                "JSON 提交，请勿上传检查点、_last_checkpoint 或其他文件"
            )
        version = int(match.group(1))
        if version > target_version:
            raise AnalyzerError(
                f"上传了高于目标版号 v{target_version} 的提交 {name}；"
                "请只上传 0 版到目标版号的提交，或把目标版号调大"
            )
        if version in by_version:
            raise AnalyzerError(
                f"版本 v{version} 的提交重复上传（{name}），请去重后再生成"
            )
        by_version[version] = parse_commit_actions(version, item.data, name)

    missing = [v for v in range(target_version + 1) if v not in by_version]
    if missing:
        shown = "、".join(f"v{v}" for v in missing[:10])
        if len(missing) > 10:
            shown += f" 等共 {len(missing)} 个"
        raise AnalyzerError(
            f"提交断档：从 0 版到目标版号 v{target_version} 缺少提交 {shown}，"
            "无法连续重放，请补齐缺失的 JSON 提交后再生成"
        )
    return [by_version[v] for v in range(target_version + 1)]


# ---------- 重放 ----------


def _replay(commits: list[CommitActions]) -> ReplayState:
    """按版本顺序、提交内按动作顺序重放，得到目标版号的表状态。"""
    state = ReplayState()
    for commit in commits:
        for kind, action in commit.file_actions:
            try:
                key = normalize_data_path(action.get("path"))
            except AnalyzerError as exc:
                raise AnalyzerError(
                    f"提交 {commit.source} 的 {kind} 动作路径无效：{exc}"
                ) from exc
            if kind == "add":
                state.active[key] = action
                state.tombstones.pop(key, None)
            else:
                state.tombstones[key] = action
                state.active.pop(key, None)
        if commit.protocol is not None:
            state.protocol = commit.protocol
        if commit.metadata is not None:
            state.metadata = commit.metadata
        for txn_action in commit.txns:
            app_id = txn_action.get("appId")
            if isinstance(app_id, str):
                state.txns[app_id] = txn_action
    return state


def _effective_protocol(state: ReplayState) -> dict:
    """目标版号生效的 protocol，必须恰为受支持的 (1, 2)。"""
    protocol = state.protocol
    if protocol is None:
        raise AnalyzerError(
            "0 版到目标版号的提交中没有任何 protocol 动作，"
            "无法确认表协议，不予生成"
        )
    min_reader = protocol.get("minReaderVersion")
    min_writer = protocol.get("minWriterVersion")
    for name, value in (("minReaderVersion", min_reader),
                        ("minWriterVersion", min_writer)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise AnalyzerError(
                f"目标版生效 protocol 的 {name} 不是整数：{value!r}，不予生成"
            )
    if (min_reader, min_writer) != SUPPORTED_PROTOCOL:
        raise AnalyzerError(
            f"目标版生效协议为 minReaderVersion={min_reader}、"
            f"minWriterVersion={min_writer}，超出支持范围：本工具只处理 "
            f"minReaderVersion={SUPPORTED_PROTOCOL[0]}、"
            f"minWriterVersion={SUPPORTED_PROTOCOL[1]} 的表"
        )
    return {"minReaderVersion": min_reader, "minWriterVersion": min_writer}


def _effective_metadata(state: ReplayState) -> dict:
    """目标版号生效的 metaData，必须存在且表无分区。"""
    metadata = state.metadata
    if metadata is None:
        raise AnalyzerError(
            "0 版到目标版号的提交中没有任何 metaData 动作，"
            "无法在检查点中保留表元数据，不予生成"
        )
    if metadata.get("partitionColumns"):
        raise AnalyzerError(
            f"表按 {metadata['partitionColumns']} 分区：本工具只处理无分区表"
        )
    return metadata


# ---------- 动作清洗（只保留标准字段，并做类型校验） ----------


def _opt_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise AnalyzerError(f"字段 {field_name} 应为整数，收到 {value!r}")
    return value


def _opt_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AnalyzerError(f"字段 {field_name} 应为字符串，收到 {value!r}")
    return value


def _opt_bool(value: Any, field_name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise AnalyzerError(f"字段 {field_name} 应为布尔值，收到 {value!r}")
    return value


def _opt_str_map(value: Any, field_name: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise AnalyzerError(f"字段 {field_name} 应为字符串键值对，收到 {value!r}")
    return dict(value)


def _opt_str_list(value: Any, field_name: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise AnalyzerError(f"字段 {field_name} 应为字符串数组，收到 {value!r}")
    return list(value)


def _sanitize_add(add: dict) -> dict:
    path = add.get("path")
    stats = add.get("stats")
    if stats is not None and not isinstance(stats, str):
        # stats 按规范是 JSON 字符串；容错地把异常值序列化保留
        stats = json.dumps(stats, ensure_ascii=False)
    return {
        "path": path,
        "partitionValues": _opt_str_map(
            add.get("partitionValues"), f"add({path}).partitionValues") or {},
        "size": _opt_int(add.get("size"), f"add({path}).size"),
        "modificationTime": _opt_int(
            add.get("modificationTime"), f"add({path}).modificationTime"),
        "dataChange": False,  # 检查点中的文件动作一律 dataChange=false
        "tags": _opt_str_map(add.get("tags"), f"add({path}).tags"),
        "stats": stats,
    }


def _sanitize_remove(remove: dict) -> dict:
    path = remove.get("path")
    return {
        "path": path,
        "deletionTimestamp": _opt_int(
            remove.get("deletionTimestamp"), f"remove({path}).deletionTimestamp"),
        "dataChange": False,  # 检查点中的文件动作一律 dataChange=false
        "extendedFileMetadata": _opt_bool(
            remove.get("extendedFileMetadata"),
            f"remove({path}).extendedFileMetadata"),
        "partitionValues": _opt_str_map(
            remove.get("partitionValues"), f"remove({path}).partitionValues"),
        "size": _opt_int(remove.get("size"), f"remove({path}).size"),
        "tags": _opt_str_map(remove.get("tags"), f"remove({path}).tags"),
    }


def _sanitize_metadata(metadata: dict) -> dict:
    fmt = metadata.get("format")
    if fmt is not None and not isinstance(fmt, dict):
        raise AnalyzerError(f"字段 metaData.format 应为对象，收到 {fmt!r}")
    return {
        "id": _opt_str(metadata.get("id"), "metaData.id"),
        "name": _opt_str(metadata.get("name"), "metaData.name"),
        "description": _opt_str(metadata.get("description"),
                                "metaData.description"),
        "format": ({
            "provider": _opt_str(fmt.get("provider"),
                                 "metaData.format.provider"),
            "options": _opt_str_map(fmt.get("options"),
                                    "metaData.format.options") or {},
        } if isinstance(fmt, dict) else None),
        "schemaString": _opt_str(metadata.get("schemaString"),
                                 "metaData.schemaString"),
        "partitionColumns": _opt_str_list(
            metadata.get("partitionColumns"), "metaData.partitionColumns") or [],
        "configuration": _opt_str_map(
            metadata.get("configuration"), "metaData.configuration") or {},
        "createdTime": _opt_int(metadata.get("createdTime"),
                                "metaData.createdTime"),
    }


def _sanitize_txn(txn_action: dict) -> dict:
    app_id = txn_action.get("appId")
    return {
        "appId": app_id,
        "version": _opt_int(txn_action.get("version"), f"txn({app_id}).version"),
        "lastUpdated": _opt_int(txn_action.get("lastUpdated"),
                                f"txn({app_id}).lastUpdated"),
    }


# ---------- 检查点写出 ----------


def _write_parquet(state: ReplayState, protocol: dict,
                   metadata: dict) -> bytes:
    """把重放状态写成经典单文件检查点：每行一个动作，其余列为 null。"""
    rows: list[tuple[str, dict]] = []
    for app_id in sorted(state.txns):
        rows.append(("txn", _sanitize_txn(state.txns[app_id])))
    for key in sorted(state.active):
        rows.append(("add", _sanitize_add(state.active[key])))
    for key in sorted(state.tombstones):
        rows.append(("remove", _sanitize_remove(state.tombstones[key])))
    rows.append(("metaData", _sanitize_metadata(metadata)))
    rows.append(("protocol", protocol))

    def column(kind: str, type_: pa.DataType) -> pa.Array:
        return pa.array(
            [payload if row_kind == kind else None for row_kind, payload in rows],
            type=type_)

    try:
        table = pa.table({
            "txn": column("txn", TXN_TYPE),
            "add": column("add", ADD_TYPE),
            "remove": column("remove", REMOVE_TYPE),
            "metaData": column("metaData", METADATA_TYPE),
            "protocol": column("protocol", PROTOCOL_TYPE),
        })
        buf = io.BytesIO()
        pq.write_table(table, buf)
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise AnalyzerError(f"组装检查点 Parquet 失败：{exc}") from exc
    return buf.getvalue()


def build_checkpoint(commit_files: list[InputFile],
                     target_version: int) -> CheckpointArtifact:
    """生成入口：校验上传、连续重放、写出检查点字节。"""
    if not isinstance(target_version, int) or isinstance(target_version, bool):
        raise AnalyzerError("目标版号必须是整数")
    if target_version < 0:
        raise AnalyzerError("目标版号不能为负数")
    if not commit_files:
        raise AnalyzerError(
            "请至少上传一个 JSON 提交（20 位补零版本号命名，如 "
            "00000000000000000000.json）"
        )

    commits = _classify_and_parse(commit_files, target_version)
    state = _replay(commits)
    protocol = _effective_protocol(state)
    metadata = _effective_metadata(state)
    data = _write_parquet(state, protocol, metadata)

    table_id = metadata.get("id")
    return CheckpointArtifact(
        filename=checkpoint_filename(target_version),
        data=data,
        target_version=target_version,
        add_count=len(state.active),
        remove_count=len(state.tombstones),
        txn_count=len(state.txns),
        table_id=table_id if isinstance(table_id, str) else None,
    )
