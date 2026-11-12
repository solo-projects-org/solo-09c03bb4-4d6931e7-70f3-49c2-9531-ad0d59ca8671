"""从 JSON 提交重放生成经典单文件 Parquet 检查点。

订单表的旧提交转入冷存储前，值班员在此为目标版本生成重放起点：上传
从 0 版到目标版的全部 Delta Lake JSON 提交（无需数据文件），本模块连续
重放 ``add`` / ``remove`` 动作，产出按目标版号命名的
``NNN….checkpoint.parquet``，供下载留存。

仅支持**无分区、无删除向量**且协议为 ``minReaderVersion=1`` /
``minWriterVersion=2`` 的表；提交断档、日志无效或超出支持范围时抛出
:class:`AnalyzerError`，调用方据此拒绝提供下载。

生成规则（与 Delta 经典检查点语义一致）：

- 每条路径只写生效的 ``add`` 或 ``remove``（同路径后出现的动作生效，
  同一提交内 remove 先于 add 应用，与分析器重放口径一致）；
- ``remove`` 墓碑不按时间淘汰，全部保留；
- 文件动作的 ``dataChange`` 一律写为 ``false``，其余标准字段按提交中的
  原值保留（路径保持日志中的 URI 编码形态）；
- ``protocol`` / ``metaData`` 取目标版本生效的最后一份；
- ``txn`` 每个 appId 只留最后提交的一条。
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .analyzer import (
    AnalyzerError,
    DV_FEATURE_NAME,
    InputFile,
    _decode_utf8,
    _iter_json_objects,
    normalize_data_path,
)

_COMMIT_RE = re.compile(r"^(\d{20})\.json$")

SUPPORTED_READER_VERSION = 1
SUPPORTED_WRITER_VERSION = 2

# ---- 经典单文件检查点列结构（无删除向量列） ----
_STRING_MAP = pa.map_(pa.string(), pa.string())
TXN_TYPE = pa.struct([
    ("appId", pa.string()),
    ("version", pa.int64()),
    ("lastUpdated", pa.int64()),
])
ADD_TYPE = pa.struct([
    ("path", pa.string()),
    ("partitionValues", _STRING_MAP),
    ("size", pa.int64()),
    ("modificationTime", pa.int64()),
    ("dataChange", pa.bool_()),
    ("tags", _STRING_MAP),
])
REMOVE_TYPE = pa.struct([
    ("path", pa.string()),
    ("deletionTimestamp", pa.int64()),
    ("dataChange", pa.bool_()),
    ("extendedFileMetadata", pa.bool_()),
    ("partitionValues", _STRING_MAP),
    ("size", pa.int64()),
    ("tags", _STRING_MAP),
])
METADATA_TYPE = pa.struct([
    ("id", pa.string()),
    ("name", pa.string()),
    ("description", pa.string()),
    ("format", pa.struct([
        ("provider", pa.string()),
        ("options", _STRING_MAP),
    ])),
    ("schemaString", pa.string()),
    ("partitionColumns", pa.list_(pa.string())),
    ("configuration", _STRING_MAP),
    ("createdTime", pa.int64()),
])
PROTOCOL_TYPE = pa.struct([
    ("minReaderVersion", pa.int32()),
    ("minWriterVersion", pa.int32()),
])


@dataclass
class _CommitActions:
    """为生成检查点而解析出的单个提交内容（保留原始动作字典）。"""

    version: int
    source: str
    adds: list[dict] = field(default_factory=list)
    removes: list[dict] = field(default_factory=list)
    txns: list[dict] = field(default_factory=list)
    protocol: dict | None = None
    metadata: dict | None = None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_int_fields(action: dict, fields: tuple[str, ...],
                      source: str, kind: str) -> None:
    for name in fields:
        value = action.get(name)
        if value is not None and not _is_int(value):
            raise AnalyzerError(
                f"日志无效：提交 {source} 的 {kind} 动作字段 {name} "
                f"不是整数：{value!r}"
            )


def _check_map_field(action: dict, name: str, source: str, kind: str) -> None:
    value = action.get(name)
    if value is not None and not isinstance(value, dict):
        raise AnalyzerError(
            f"日志无效：提交 {source} 的 {kind} 动作字段 {name} "
            f"不是对象：{value!r}"
        )


def _check_no_deletion_vector(action: dict, source: str, kind: str) -> None:
    if action.get("deletionVector") is not None:
        raise AnalyzerError(
            f"表包含删除向量：提交 {source} 的 {kind} 动作带有 "
            f"deletionVector 字段。本工具只处理无删除向量的表。"
        )


def _check_unpartitioned(action: dict, source: str, kind: str) -> None:
    partition_values = action.get("partitionValues")
    if partition_values:
        raise AnalyzerError(
            f"分区表超出支持范围：提交 {source} 的 {kind} 动作带有非空 "
            f"partitionValues。本工具只处理无分区的表。"
        )


def _validate_add(add: dict, source: str) -> None:
    # path 合法性复用归一化函数（非字符串会抛 AnalyzerError）
    normalize_data_path(add.get("path"))
    _check_no_deletion_vector(add, source, "add")
    _check_map_field(add, "partitionValues", source, "add")
    _check_unpartitioned(add, source, "add")
    _check_map_field(add, "tags", source, "add")
    _check_int_fields(add, ("size", "modificationTime"), source, "add")


def _validate_remove(remove: dict, source: str) -> None:
    normalize_data_path(remove.get("path"))
    _check_no_deletion_vector(remove, source, "remove")
    _check_map_field(remove, "partitionValues", source, "remove")
    _check_unpartitioned(remove, source, "remove")
    _check_map_field(remove, "tags", source, "remove")
    _check_int_fields(remove, ("deletionTimestamp", "size"), source, "remove")


def _validate_protocol(protocol: dict, source: str) -> None:
    features: list[str] = []
    for key in ("readerFeatures", "writerFeatures", "features"):
        value = protocol.get(key)
        if isinstance(value, list):
            features.extend(str(v) for v in value)
    if DV_FEATURE_NAME in features:
        raise AnalyzerError(
            f"表启用了删除向量：提交 {source} 的 protocol 特性包含 "
            f"{DV_FEATURE_NAME}。本工具只处理无删除向量的表。"
        )
    if features:
        raise AnalyzerError(
            f"协议特性超出支持范围：提交 {source} 的 protocol 声明了特性 "
            f"{features}。本工具只处理 minReaderVersion="
            f"{SUPPORTED_READER_VERSION}、minWriterVersion="
            f"{SUPPORTED_WRITER_VERSION} 的表。"
        )
    for key in ("minReaderVersion", "minWriterVersion"):
        if not _is_int(protocol.get(key)):
            raise AnalyzerError(
                f"日志无效：提交 {source} 的 protocol 缺少合法的 {key}"
            )


def _validate_metadata(metadata: dict, source: str) -> None:
    table_id = metadata.get("id")
    if not isinstance(table_id, str) or not table_id:
        raise AnalyzerError(
            f"日志无效：提交 {source} 的 metaData 缺少合法的 id"
        )
    partition_columns = metadata.get("partitionColumns")
    if partition_columns is not None:
        if (not isinstance(partition_columns, list)
                or not all(isinstance(c, str) for c in partition_columns)):
            raise AnalyzerError(
                f"日志无效：提交 {source} 的 metaData.partitionColumns "
                f"不是字符串数组"
            )
        if partition_columns:
            raise AnalyzerError(
                f"分区表超出支持范围：提交 {source} 的 metaData 声明了分区列 "
                f"{partition_columns}。本工具只处理无分区的表。"
            )
    _check_map_field(metadata, "configuration", source, "metaData")
    fmt = metadata.get("format")
    if fmt is not None:
        if not isinstance(fmt, dict):
            raise AnalyzerError(
                f"日志无效：提交 {source} 的 metaData.format 不是对象"
            )
        options = fmt.get("options")
        if options is not None and not isinstance(options, dict):
            raise AnalyzerError(
                f"日志无效：提交 {source} 的 metaData.format.options 不是对象"
            )


def _validate_txn(txn: dict, source: str) -> None:
    app_id = txn.get("appId")
    if not isinstance(app_id, str) or not app_id:
        raise AnalyzerError(
            f"日志无效：提交 {source} 的 txn 动作缺少合法的 appId"
        )
    if not _is_int(txn.get("version")):
        raise AnalyzerError(
            f"日志无效：提交 {source} 的 txn 动作缺少合法的 version"
        )
    _check_int_fields(txn, ("lastUpdated",), source, "txn")


def _parse_commit_for_checkpoint(version: int, data: bytes,
                                 source: str) -> _CommitActions:
    """解析并校验一个 JSON 提交，保留生成检查点所需的原始动作。"""
    text = _decode_utf8(data, source)
    parsed = _CommitActions(version=version, source=source)
    for obj in _iter_json_objects(text, source):
        if not isinstance(obj, dict):
            continue
        if "add" in obj and isinstance(obj["add"], dict):
            _validate_add(obj["add"], source)
            parsed.adds.append(obj["add"])
        if "remove" in obj and isinstance(obj["remove"], dict):
            _validate_remove(obj["remove"], source)
            parsed.removes.append(obj["remove"])
        if "txn" in obj and isinstance(obj["txn"], dict):
            _validate_txn(obj["txn"], source)
            parsed.txns.append(obj["txn"])
        if "protocol" in obj and isinstance(obj["protocol"], dict):
            _validate_protocol(obj["protocol"], source)
            parsed.protocol = obj["protocol"]
        if "metaData" in obj and isinstance(obj["metaData"], dict):
            _validate_metadata(obj["metaData"], source)
            parsed.metadata = obj["metaData"]
    return parsed


def _format_versions(versions: list[int]) -> str:
    shown = ", ".join(f"v{v}" for v in versions[:10])
    if len(versions) > 10:
        shown += f" 等共 {len(versions)} 个"
    return shown


def _add_struct(raw: dict) -> dict:
    return {
        "path": raw.get("path"),
        "partitionValues": raw.get("partitionValues") or {},
        "size": raw.get("size"),
        "modificationTime": raw.get("modificationTime"),
        # 检查点中的文件动作 dataChange 一律为 false
        "dataChange": False,
        "tags": raw.get("tags"),
    }


def _remove_struct(raw: dict) -> dict:
    return {
        "path": raw.get("path"),
        "deletionTimestamp": raw.get("deletionTimestamp"),
        "dataChange": False,
        "extendedFileMetadata": raw.get("extendedFileMetadata"),
        "partitionValues": raw.get("partitionValues"),
        "size": raw.get("size"),
        "tags": raw.get("tags"),
    }


def _metadata_struct(raw: dict) -> dict:
    fmt = raw.get("format")
    return {
        "id": raw.get("id"),
        "name": raw.get("name"),
        "description": raw.get("description"),
        "format": ({"provider": fmt.get("provider"),
                    "options": fmt.get("options") or {}}
                   if isinstance(fmt, dict) else None),
        "schemaString": raw.get("schemaString"),
        "partitionColumns": raw.get("partitionColumns") or [],
        "configuration": raw.get("configuration") or {},
        "createdTime": raw.get("createdTime"),
    }


def build_checkpoint(log_files: list[InputFile],
                     target_version: int) -> tuple[bytes, str]:
    """从 0 版到目标版的连续 JSON 提交生成经典单文件检查点。

    返回 ``(parquet 字节, 下载文件名)``；任何输入 / 能力问题都抛出
    :class:`AnalyzerError`，调用方不得提供下载。
    """
    if not _is_int(target_version):
        raise AnalyzerError("目标版本必须是整数")
    if target_version < 0:
        raise AnalyzerError("目标版本不能为负数")

    commits: dict[int, _CommitActions] = {}
    seen_bytes: dict[int, bytes] = {}
    for item in log_files:
        name = item.filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        match = _COMMIT_RE.match(name)
        if not match:
            continue  # 检查点、指针文件等非 JSON 提交不参与生成
        version = int(match.group(1))
        if version > target_version:
            continue  # 目标版之后的提交与本次生成无关
        if version in seen_bytes:
            if seen_bytes[version] != item.data:
                raise AnalyzerError(
                    f"提交 {name} 重复上传且内容不一致，无法确定采用哪一份"
                )
            continue  # 内容相同的重复上传直接忽略
        seen_bytes[version] = item.data
        commits[version] = _parse_commit_for_checkpoint(version, item.data, name)

    if not commits:
        raise AnalyzerError(
            "上传文件中没有可识别的 JSON 提交（应为 20 位补零版本号命名的 "
            ".json 文件，如 00000000000000000000.json）"
        )

    missing = [v for v in range(target_version + 1) if v not in commits]
    if missing:
        raise AnalyzerError(
            f"提交断档：生成 v{target_version} 检查点需要 0–v{target_version} "
            f"的连续提交，缺少：{_format_versions(missing)}。请补齐后重新上传。"
        )

    # 连续重放 0..target：同一提交内 remove 先于 add 应用（与分析器一致），
    # 同一路径后出现的动作覆盖先前的；墓碑不按时间淘汰。
    file_actions: dict[str, tuple[str, dict]] = {}
    txns: dict[str, dict] = {}
    protocol: dict | None = None
    metadata: dict | None = None
    for version in range(target_version + 1):
        commit = commits[version]
        for remove in commit.removes:
            file_actions[normalize_data_path(remove.get("path"))] = \
                ("remove", remove)
        for add in commit.adds:
            file_actions[normalize_data_path(add.get("path"))] = ("add", add)
        for txn in commit.txns:
            txns[txn["appId"]] = txn  # 每个 appId 只留最后提交的 txn
        if commit.protocol is not None:
            protocol = commit.protocol
        if commit.metadata is not None:
            metadata = commit.metadata

    if protocol is None:
        raise AnalyzerError(
            f"日志无效：0–v{target_version} 的提交中没有任何 protocol 动作，"
            f"无法生成检查点"
        )
    if metadata is None:
        raise AnalyzerError(
            f"日志无效：0–v{target_version} 的提交中没有任何 metaData 动作，"
            f"无法生成检查点"
        )
    reader_version = protocol["minReaderVersion"]
    writer_version = protocol["minWriterVersion"]
    if (reader_version != SUPPORTED_READER_VERSION
            or writer_version != SUPPORTED_WRITER_VERSION):
        raise AnalyzerError(
            f"协议超出支持范围：v{target_version} 生效协议为 minReaderVersion="
            f"{reader_version}、minWriterVersion={writer_version}，本工具只处理 "
            f"minReaderVersion={SUPPORTED_READER_VERSION}、minWriterVersion="
            f"{SUPPORTED_WRITER_VERSION} 的表。"
        )

    adds = sorted((path, raw) for path, (kind, raw) in file_actions.items()
                  if kind == "add")
    removes = sorted((path, raw) for path, (kind, raw) in file_actions.items()
                     if kind == "remove")

    # 每个动作占一行，仅对应列非空；行序确定，便于核对与测试
    columns: dict[str, list] = {name: [] for name in
                                ("txn", "add", "remove", "metaData", "protocol")}

    def emit(**row: Any) -> None:
        for name in columns:
            columns[name].append(row.get(name))

    for app_id in sorted(txns):
        txn = txns[app_id]
        emit(txn={"appId": txn["appId"], "version": txn["version"],
                  "lastUpdated": txn.get("lastUpdated")})
    for _path, raw in adds:
        emit(add=_add_struct(raw))
    for _path, raw in removes:
        emit(remove=_remove_struct(raw))
    emit(metaData=_metadata_struct(metadata))
    emit(protocol={"minReaderVersion": reader_version,
                   "minWriterVersion": writer_version})

    try:
        table = pa.table({
            "txn": pa.array(columns["txn"], type=TXN_TYPE),
            "add": pa.array(columns["add"], type=ADD_TYPE),
            "remove": pa.array(columns["remove"], type=REMOVE_TYPE),
            "metaData": pa.array(columns["metaData"], type=METADATA_TYPE),
            "protocol": pa.array(columns["protocol"], type=PROTOCOL_TYPE),
        })
        buffer = io.BytesIO()
        pq.write_table(table, buffer)
    except (pa.ArrowException, ValueError, TypeError) as exc:
        raise AnalyzerError(
            f"日志无效：提交中的字段值无法写入检查点（{exc}）"
        ) from exc

    download_name = f"{target_version:020d}.checkpoint.parquet"
    return buffer.getvalue(), download_name
