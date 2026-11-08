"""在独立的Python 3.8环境中执行原UNETR++五折推理。"""

import logging
import os
import shutil
import subprocess
from pathlib import Path

from .config import SETTINGS
from .nii_io import assert_same_grid, load_nii

LOG = logging.getLogger(__name__)


def segment_roi(roi_path, output_path, project_root):
    project_root = Path(project_root).resolve()
    roi_path, output_path = Path(roi_path).resolve(), Path(output_path).resolve()
    model_dir = project_root / "models" / "seg_unetrpp"
    segment = SETTINGS["segment"]
    folds = [int(i) for i in segment["folds"]]
    if folds != [0, 1, 2, 3, 4]:
        raise ValueError("当前产品配置要求完整五折[0,1,2,3,4]")
    checkpoint = str(segment["checkpoint"])
    required = [model_dir / "plans.pkl"] + [
        model_dir / f"fold_{i}" / f"{checkpoint}.model{suffix}"
        for i in folds for suffix in ("", ".pkl")]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("UNETR++模型文件缺失: " + ", ".join(missing))
    if not roi_path.is_file():
        raise FileNotFoundError(roi_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    worker_python = os.environ.get("ALN_UNETR_PYTHON")
    if worker_python:
        command = [worker_python]
    else:
        conda = shutil.which("conda")
        if conda is None:
            raise RuntimeError("找不到conda；请在云端配置aln-unetrpp环境，或设置ALN_UNETR_PYTHON")
        command = [conda, "run", "--no-capture-output", "-n", "aln-unetrpp", "python"]
    command += ["-m", "modules.segment_worker", "--input", str(roi_path),
                "--output", str(output_path), "--model-dir", str(model_dir),
                "--folds", ",".join(map(str, folds)), "--checkpoint", checkpoint,
                "--step-size", str(float(segment["step_size"])),
                "--tta", str(bool(segment["tta"])).lower(),
                "--mixed-precision", str(bool(segment["mixed_precision"])).lower(),
                "--all-in-gpu", str(bool(segment["all_in_gpu"])).lower()]
    env = os.environ.copy()
    imports = [str(project_root), str(project_root / "third_party" / "unetr_plus_plus")]
    env["PYTHONPATH"] = os.pathsep.join(imports + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    LOG.info("UNETR++五折推理启动，ROI=%s", roi_path)
    with subprocess.Popen(command, cwd=project_root, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
        for line in process.stdout:
            LOG.info("[UNETR++] %s", line.rstrip())
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"UNETR++子进程失败，退出码={code}；请查看上方[UNETR++]日志")
    if not output_path.is_file():
        raise FileNotFoundError(f"分割子进程未生成{output_path}")
    assert_same_grid(load_nii(roi_path), load_nii(output_path))
    return output_path
