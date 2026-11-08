"""从同一份节点结果生成 JSON、Markdown 和 CSV。"""

import csv
import hashlib
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_sha256(root):
    """无 Git 仓库时使用源码内容哈希标识代码版本。"""
    root = Path(root)
    files = [root / "main.py", root / "app.py"]
    files += sorted((root / "modules").glob("*.py"))
    files += sorted((root / "configs").glob("*.yaml"))
    files += sorted((root / "configs").glob("*.json"))
    files += sorted((root / "third_party" / "unetr_plus_plus" / "unetr_pp").rglob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def provenance(root, radiomics_version=None):
    root = Path(root)
    manifest = json.loads((root / "configs" / "model_manifest.json").read_text(encoding="utf-8"))
    normalization = json.loads((root / "configs" / "feature_normalization.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    try:
        total_version = importlib.metadata.version("TotalSegmentator")
    except importlib.metadata.PackageNotFoundError:
        total_version = None
    return {
        "code_sha256": code_sha256(root),
        "unetrpp": {**manifest["segmentation"],
                     "plans_sha256": files["models/seg_unetrpp/plans.pkl"],
                     "fold_weight_sha256": {str(i): files[f"models/seg_unetrpp/fold_{i}/model_best.model"] for i in range(5)}},
        "totalsegmentator": {"package_version": total_version, "task": "total",
                              "roi_subset": ["clavicula_left", "rib_left_5"]},
        "lightgbm": {"model_sha256": files["models/Best_Model_LightGBM.pkl"],
                      "feature_order_source": "model package:features",
                      "selected_features": normalization["features"],
                      "feature_normalization_sha256": sha256(root / "configs" / "feature_normalization.json"),
                      "label_rule": "model.predict; probability=predict_proba[:,1]"},
        "radiomics": {"config_sha256": sha256(root / "configs" / "radiomics.yaml"),
                       "runtime_version": radiomics_version},
    }


def _show(value):
    if value is None:
        return "未记录"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def render_markdown(report):
    """兼容旧版 run_002；缺失字段明确写未记录。"""
    nodes = report.get("nodes", [])
    source = report.get("input", {})
    timing = report.get("timing", {})
    prov = report.get("provenance", {})
    lines = ["# 腋窝淋巴结结构化分析报告", "",
             f"- 脱敏病例编号：{_show(report.get('patient_id'))}",
             f"- 运行编号：{_show(report.get('run_id'))}",
             f"- 分析侧别：{_show(report.get('target_side'))}",
             f"- 状态：{_show(report.get('status'))}",
             f"- 淋巴结数量：{len(nodes)}", "",
             "## 输入影像", "",
             f"- 输入格式：{_show(source.get('format'))}",
             f"- 输入文件 SHA256：{_show(source.get('sha256'))}",
             "- 影像期相：由操作者选择静脉期；程序未自动验证期相",
             f"- 空间元信息：{_show(report.get('input_space'))}", "",
             "## 模型和参数", "",
             f"- 代码 SHA256：{_show(prov.get('code_sha256'))}",
             f"- UNETR++：{_show(prov.get('unetrpp'))}",
             f"- TotalSegmentator：{_show(prov.get('totalsegmentator'))}",
             f"- LightGBM：{_show(prov.get('lightgbm'))}",
             f"- PyRadiomics：{_show(prov.get('radiomics'))}",
             f"- 运行参数：{_show(report.get('processing_parameters'))}", "",
             "## 执行时间", "",
             f"- 开始：{_show(timing.get('started_at_utc'))}",
             f"- 结束：{_show(timing.get('finished_at_utc'))}",
             f"- 总耗时（秒）：{_show(timing.get('duration_seconds'))}", "",
             "## 单颗淋巴结", "",
             "| ID | 侧别 | 原始空间质心坐标（mm） | 体积（mm³） | 恶性概率 | 分类 | 解剖等级 |",
             "| --- | --- | --- | ---: | ---: | --- | --- |"]
    for node in nodes:
        cells = [_show(node.get(key)) for key in ("node_id", "side", "centroid_native_world",
                                                 "volume_mm3", "malignant_probability", "malignancy", "level")]
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
    if not nodes:
        lines.append("| 无有效节点 | — | — | — | — | — | — |")
    lines += ["", "## 输出文件", ""]
    lines += [f"- {name}" for name in report.get("artifacts", [])] or ["- 旧版报告未记录文件清单"]
    return "\n".join(lines) + "\n"


def write_report(nodes, patient_id, target_side, run_id, input_metadata, bone_axes, output_dir,
                 *, project_root=None, input_path=None, parameters=None, started_at_utc=None,
                 finished_at_utc=None, duration_seconds=None, radiomics_version=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(node) for node in nodes]
    report = {
        "schema_version": 2, "patient_id": patient_id, "target_side": target_side,
        "run_id": run_id, "status": "completed" if rows else "completed_no_nodes",
        "node_count": len(rows), "input_space": input_metadata, "reference_axes": bone_axes,
        "nodes": rows,
        "input": {"format": ".nii.gz" if str(input_path).endswith(".nii.gz") else ".nii",
                  "sha256": sha256(input_path) if input_path else None,
                  "phase_expected": "venous", "phase_automatically_verified": False},
        "processing_parameters": parameters,
        "provenance": provenance(project_root, radiomics_version) if project_root else {},
        "timing": {"started_at_utc": started_at_utc, "finished_at_utc": finished_at_utc,
                   "duration_seconds": duration_seconds},
        "artifacts": [f"{patient_id}_report.json", f"{patient_id}_report.md",
                      f"{patient_id}_nodes.csv", "aln_instances_native.nii.gz",
                      "levels_native.nii.gz", "malignancy_native.nii.gz",
                      "previews/overview.png", "previews/node_montage.png"],
    }
    json_path = output_dir / f"{patient_id}_report.json"
    md_path = output_dir / f"{patient_id}_report.md"
    csv_path = output_dir / f"{patient_id}_nodes.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    columns = list(rows[0]) if rows else ["node_id", "instance_id", "side", "local_id",
        "voxel_count_1mm", "volume_mm3", "centroid_side_world", "centroid_native_world", "t", "d_mm",
        "level", "malignant_probability", "malignancy", "extracted_feature_count"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, md_path, csv_path
