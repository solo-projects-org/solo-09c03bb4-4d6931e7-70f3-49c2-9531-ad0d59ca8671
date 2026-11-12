"""Flask 网页：Delta Lake 版本可恢复性分析 + 重放起点检查点生成。

值班员在页面上填写版本区间，上传 _delta_log 留存的 JSON 提交、
经典单文件 Parquet 检查点，以及现存数据文件的表内相对路径清单，
提交后逐版展示可恢复性结论。页面也可一键加载仓库内置的本地样例。

旧提交转入冷存储前，还可在页面下半部生成重放起点：填目标版号、
上传 0 版到该版的全部 JSON 提交，即可下载按目标版号命名的经典单文件
Parquet 检查点；本地样例同样支持直接下载检查点。
"""

from __future__ import annotations

import io
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from .analyzer import AnalyzerError, InputFile, analyze
from .checkpoint import build_checkpoint

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024  # 单次请求上限 256MB

SAMPLE_ROOT = Path(os.environ.get("SAMPLE_ROOT",
                                  Path(__file__).resolve().parent.parent / "samples"))

# 样例目录名 -> 页面展示名 / 默认版本区间
SAMPLES = {
    "healthy_table": {
        "label": "健康表样例（0-4 版均可重建）",
        "range": (0, 4),
    },
    "incident_table": {
        "label": "误删事故样例（部分版本缺提交或缺数据文件）",
        "range": (0, 5),
    },
}


def _read_uploads(field_name: str) -> list[InputFile]:
    files = []
    for storage in request.files.getlist(field_name):
        if not storage or storage.filename == "":
            continue
        files.append(InputFile(filename=storage.filename,
                               data=storage.read()))
    return files


def _parse_version(raw: str, field_label: str) -> int:
    raw = (raw or "").strip()
    if raw == "":
        raise AnalyzerError(f"请填写{field_label}")
    try:
        return int(raw)
    except ValueError as exc:
        raise AnalyzerError(f"{field_label}必须是整数，收到：{raw!r}") from exc


def _run_analysis(log_files, manifest_files, from_version, to_version):
    return analyze(log_files=log_files,
                   manifest_files=manifest_files,
                   from_version=from_version,
                   to_version=to_version)


@app.get("/")
def index():
    return render_template("index.html", samples=SAMPLES)


@app.post("/analyze")
def analyze_upload():
    try:
        from_version = _parse_version(request.form.get("from_version"), "起始版本")
        to_version = _parse_version(request.form.get("to_version"), "结束版本")
        log_files = _read_uploads("log_files")
        manifest_files = _read_uploads("manifest_files")
        report = _run_analysis(log_files, manifest_files,
                               from_version, to_version)
    except AnalyzerError as exc:
        return render_template("index.html", samples=SAMPLES, error=str(exc),
                               form=request.form), 400
    return render_template("index.html", samples=SAMPLES, report=report)


@app.get("/api/samples")
def list_samples():
    return jsonify({
        key: {"label": meta["label"], "range": list(meta["range"])}
        for key, meta in SAMPLES.items()
    })


def _checkpoint_response(data: bytes, download_name: str):
    return send_file(io.BytesIO(data),
                     mimetype="application/octet-stream",
                     as_attachment=True,
                     download_name=download_name)


@app.post("/checkpoint")
def generate_checkpoint():
    """上传 0 版到目标版的 JSON 提交，生成并下载经典单文件检查点。"""
    try:
        target_version = _parse_version(request.form.get("target_version"),
                                        "目标版本")
        commit_files = _read_uploads("commit_files")
        data, download_name = build_checkpoint(commit_files, target_version)
    except AnalyzerError as exc:
        return render_template("index.html", samples=SAMPLES, error=str(exc),
                               form=request.form), 400
    return _checkpoint_response(data, download_name)


@app.post("/api/checkpoint-sample/<sample_name>")
def checkpoint_sample(sample_name: str):
    """用服务器本地样例的 JSON 提交生成检查点，无需手动上传。"""
    if sample_name not in SAMPLES:
        return jsonify({"error": f"未知样例：{sample_name}"}), 404
    log_dir = SAMPLE_ROOT / sample_name / "_delta_log"
    if not log_dir.is_dir():
        return jsonify({"error": f"样例日志目录不存在：{log_dir}"}), 404

    payload = request.get_json(silent=True) or {}
    try:
        target_version = int(payload.get("target_version"))
    except (TypeError, ValueError):
        return jsonify({"error": "目标版本必须是整数"}), 400

    commit_files = [InputFile(filename=path.name, data=path.read_bytes())
                    for path in sorted(log_dir.iterdir())
                    if path.is_file() and path.suffix == ".json"]
    try:
        data, download_name = build_checkpoint(commit_files, target_version)
    except AnalyzerError as exc:
        return jsonify({"error": str(exc)}), 400
    except OSError as exc:
        return jsonify({"error": f"读取样例失败：{exc}"}), 500
    return _checkpoint_response(data, download_name)


@app.post("/api/analyze-sample/<sample_name>")
def analyze_sample(sample_name: str):
    """直接加载服务器本地样例目录，无需手动上传。"""
    if sample_name not in SAMPLES:
        return jsonify({"error": f"未知样例：{sample_name}"}), 404
    sample_dir = SAMPLE_ROOT / sample_name
    if not sample_dir.is_dir():
        return jsonify({"error": f"样例目录不存在：{sample_dir}"}), 404

    try:
        payload = request.get_json(silent=True) or {}
        meta = SAMPLES[sample_name]
        from_version = int(payload.get("from_version", meta["range"][0]))
        to_version = int(payload.get("to_version", meta["range"][1]))

        log_dir = sample_dir / "_delta_log"
        log_files = []
        for path in sorted(log_dir.iterdir()):
            if path.is_file():
                log_files.append(InputFile(filename=path.name,
                                           data=path.read_bytes()))
        manifest_path = sample_dir / "existing_files.txt"
        manifest_files = [InputFile(
            filename=manifest_path.name,
            data=manifest_path.read_bytes(),
        )]
        report = _run_analysis(log_files, manifest_files,
                               from_version, to_version)
    except AnalyzerError as exc:
        return jsonify({"error": str(exc)}), 400
    except (OSError, ValueError) as exc:
        return jsonify({"error": f"读取样例失败：{exc}"}), 500
    return jsonify(report)


@app.errorhandler(413)
def too_large(_err):
    return render_template("index.html", samples=SAMPLES,
                           error="上传内容超过 256MB 上限"), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
