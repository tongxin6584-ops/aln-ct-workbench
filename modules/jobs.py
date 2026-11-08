"""Streamlit 可断线重连的后台运行记录；不依赖浏览器会话。"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("编号只允许英文字母、数字、下划线和连字符")
    return value


def start_job(root, input_path, patient_id, side, run_id):
    root = Path(root).resolve()
    patient_id, run_id = _safe_name(patient_id), _safe_name(run_id)
    if side not in ("left", "right", "bilateral"):
        raise ValueError("侧别无效")
    job_dir = root / "workspace" / "_jobs"
    job_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = job_dir / f"{run_id}.json"
    if metadata_path.exists() or (root / "results" / patient_id / run_id).exists():
        raise FileExistsError(f"运行编号已存在: {run_id}")
    if any(job_status(root, item).get("status") == "running" for item in job_dir.glob("*.json")):
        raise RuntimeError("已有分析任务运行中；请等待完成后再提交，避免GPU显存不足")
    metadata = {"run_id": run_id, "patient_id": patient_id, "side": side,
                "input_path": str(Path(input_path).resolve()),
                "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
                "console_log": str(root / "workspace" / "_logs" / f"{patient_id}_{run_id}_console.log")}
    _write_json(metadata_path, metadata)
    try:
        Path(metadata["console_log"]).parent.mkdir(parents=True, exist_ok=True)
        with Path(metadata["console_log"]).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "modules.jobs", "worker", "--root", str(root),
                 "--run-id", run_id], cwd=root, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"})
        # Worker may finish before this update; the exit marker remains authoritative.
        metadata["pid"] = process.pid
        _write_json(metadata_path, metadata)
    except Exception:
        metadata_path.unlink(missing_ok=True)
        raise
    return metadata


def _pid_alive(pid):
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def job_status(root, metadata_path):
    root, metadata_path = Path(root), Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    run_id = _safe_name(metadata["run_id"])
    patient_id = _safe_name(metadata["patient_id"])
    exit_path = root / "workspace" / "_jobs" / f"{run_id}.exit.json"
    result_dir = root / "results" / patient_id / run_id
    if exit_path.is_file():
        exit_data = json.loads(exit_path.read_text(encoding="utf-8"))
        metadata["status"] = "completed" if exit_data["returncode"] == 0 else "failed"
        metadata["returncode"] = exit_data["returncode"]
        metadata["finished_at_utc"] = exit_data["finished_at_utc"]
    elif any(result_dir.glob("*_report.json")):
        metadata["status"] = "completed"
    elif _pid_alive(metadata.get("pid")) or "pid" not in metadata:
        metadata["status"] = "running"
    else:
        metadata["status"] = "interrupted"
    metadata["result_dir"] = str(result_dir)
    return metadata


def list_jobs(root):
    return [job_status(root, path) for path in sorted(
        (Path(root) / "workspace" / "_jobs").glob("*.json"), reverse=True)
        if not path.name.endswith(".exit.json")]


def list_results(root):
    return sorted((Path(root) / "results").glob("*/*/*_report.json"), reverse=True)


def worker(root, run_id):
    root = Path(root).resolve()
    metadata = json.loads((root / "workspace" / "_jobs" / f"{_safe_name(run_id)}.json").read_text(encoding="utf-8"))
    command = [sys.executable, str(root / "main.py"), "--input", metadata["input_path"],
               "--patient-id", metadata["patient_id"], "--side", metadata["side"],
               "--run-id", metadata["run_id"]]
    try:
        returncode = subprocess.call(command, cwd=root)
    except Exception as exc:
        print(f"后台任务启动失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        returncode = 1
    _write_json(root / "workspace" / "_jobs" / f"{run_id}.exit.json",
                {"returncode": returncode, "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    return returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["worker"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    sys.exit(worker(args.root, args.run_id))
