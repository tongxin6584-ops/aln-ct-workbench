"""端到端入口：只负责按pipeline顺序调用模块，不重复实现算法。"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from modules.assets import verify_model_assets
from modules.classify import classify_nodes
from modules.config import SETTINGS
from modules.grade import grade_nodes
from modules.postprocess import constrain_by_clavicle
from modules.preprocess import run_preprocessing
from modules.report import write_report
from modules.restore import (combine_side_instances, restore_post_mask,
                             save_native_instances, save_result_maps)
from modules.segment import segment_roi
from modules.split_nodes import split_nodes
from modules.status import stage
from modules.visualize import write_previews

LOG = logging.getLogger("aln.pipeline")


def _configure_logging(project_root, patient_id, run_id):
    log_dir = project_root / "workspace" / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(log_dir / f"{patient_id}_{run_id}.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        root.addHandler(handler)
    return log_dir / f"{patient_id}_{run_id}.log"


def run_pipeline(input_path, patient_id, target_side, run_id="run_001", project_root=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", patient_id) or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("patient_id和run_id只允许英文字母、数字、下划线、连字符")
    project_root = Path(project_root or Path(__file__).resolve().parent).resolve()
    started_at_utc = datetime.now(timezone.utc).isoformat()
    started_monotonic = time.monotonic()
    log_path = _configure_logging(project_root, patient_id, run_id)
    result_dir = project_root / "results" / patient_id / run_id
    if result_dir.exists():
        raise FileExistsError(f"结果目录已存在，避免覆盖: {result_dir}")
    with stage("00 核对12个模型资产及SHA256"):
        count = verify_model_assets(project_root)
        LOG.info("模型资产校验通过：%d个文件", count)
    with stage("01-05 输入校验、空间/侧别标准化、去床板、骨骼提取、ROI裁剪"):
        run_dir = run_preprocessing(input_path, patient_id, target_side, project_root, run_id)
    metadata = json.loads((run_dir / "metadata" / "transform.json").read_text(encoding="utf-8"))
    sides = ["left", "right"] if target_side == "bilateral" else [target_side]
    all_nodes, side_instance_paths, axes = [], {}, {}
    for side in sides:
        side_dir = run_dir / side
        with stage(f"06 {side} UNETR++五折分割"):
            segment_roi(side_dir / "06_roi_ct.nii.gz", side_dir / "07_raw_mask.nii.gz", project_root)
        with stage(f"07 {side} 锁骨约束后处理"):
            constrain_by_clavicle(side_dir / "07_raw_mask.nii.gz", side_dir / "06_roi_clav.nii.gz",
                                  side_dir / "08_post_mask.nii.gz")
        with stage(f"08 {side} 逆Y/Z裁剪"):
            restore_post_mask(side_dir / "08_post_mask.nii.gz", side_dir / "06_roi_ct.nii.gz",
                              side_dir / "03_side_ct.nii.gz", metadata["sides"][side],
                              side_dir / "09_side_mask.nii.gz")
        with stage(f"08 {side} 单颗淋巴结拆分"):
            instance_path = side_dir / "10_instances.nii.gz"
            nodes = split_nodes(side_dir / "09_side_mask.nii.gz", instance_path,
                                side, patient_id, len(all_nodes) + 1)
            side_instance_paths[side] = instance_path
        with stage(f"09A {side} 解剖分级"):
            axes[side] = grade_nodes(nodes, side_dir / "05_clavicle.nii.gz",
                                     side_dir / "05_rib5.nii.gz")
            LOG.warning("%s分级完成：%d颗淋巴结", side, len(nodes))
        all_nodes.extend(nodes)
    canonical_dir, native_dir = run_dir / "11_canonical_results", run_dir / "12_native_results"
    with stage("09B 回填RAS全画幅并恢复原始CT网格"):
        canonical_instances = combine_side_instances(side_instance_paths, metadata["sides"],
            run_dir / "02_ras_full" / "ct_ras.nii.gz", canonical_dir / "instances.nii.gz")
        native_instances = save_native_instances(canonical_instances,
            run_dir / "02_ras_full" / "ct_ras.nii.gz",
            run_dir / "01_native_full" / "input.nii.gz", metadata["input"]["resampled_to_1mm"],
            native_dir / "instances_native.nii.gz")
    with stage("10 原始全画幅CT影像组学与LightGBM分类"):
        classify_nodes(all_nodes, native_instances, run_dir / "01_native_full" / "input.nii.gz",
                       project_root / "models" / "Best_Model_LightGBM.pkl",
                       project_root / "configs" / "radiomics.yaml", native_dir / "nodes")
        LOG.warning("分类完成：%d颗；零节点时输出空报告", len(all_nodes))
    with stage("11 输出全画幅NIfTI、JSON、Markdown、CSV和预览图"):
        save_result_maps(canonical_instances, native_instances, all_nodes,
                         run_dir / "02_ras_full" / "ct_ras.nii.gz",
                         run_dir / "01_native_full" / "input.nii.gz",
                         metadata["input"]["resampled_to_1mm"], run_dir, result_dir)
        write_previews(run_dir / "01_native_full" / "input.nii.gz",
                       result_dir / "aln_instances_native.nii.gz", all_nodes, result_dir / "previews")
        features_path = native_dir / "radiomics_features.json"
        radiomics_version = (json.loads(features_path.read_text(encoding="utf-8")).get("radiomics_version")
                             if features_path.is_file() else None)
        write_report(all_nodes, patient_id, target_side, run_id, metadata["input"], axes, result_dir,
                     project_root=project_root, input_path=input_path, parameters=SETTINGS,
                     started_at_utc=started_at_utc,
                     finished_at_utc=datetime.now(timezone.utc).isoformat(),
                     duration_seconds=round(time.monotonic() - started_monotonic, 1),
                     radiomics_version=radiomics_version)
    LOG.warning("全部阶段完成：患者=%s，节点=%d，结果=%s，日志=%s",
                patient_id, len(all_nodes), result_dir, log_path)
    return result_dir


def main():
    parser = argparse.ArgumentParser(description="静脉期胸部CT到结构化腋窝淋巴结报告")
    parser.add_argument("--input", required=True, help="全画幅3D NIfTI路径")
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--side", choices=("left", "right", "bilateral"), required=True)
    parser.add_argument("--run-id", default="run_001")
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()
    try:
        print(run_pipeline(args.input, args.patient_id, args.side, args.run_id, args.project_root))
    except Exception as exc:
        print(f"[流水线失败] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
