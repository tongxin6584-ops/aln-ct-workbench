"""原始CT组学特征由旧环境提取；主环境完成两次标准化和LightGBM分类。"""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import SETTINGS
from .nii_io import assert_same_grid, get_centroid_world, load_nii, save_nii

LOG = logging.getLogger(__name__)


def _extract_features_in_legacy_env(manifest_path, output_path, project_root):
    worker_python = os.environ.get("ALN_UNETR_PYTHON")
    if worker_python:
        command = [worker_python]
    else:
        conda = shutil.which("conda")
        if conda is None:
            raise RuntimeError("找不到conda；请先配置aln-unetrpp环境")
        command = [conda, "run", "--no-capture-output", "-n", "aln-unetrpp", "python"]
    command += ["-m", "modules.radiomics_worker", "--manifest", str(manifest_path),
                "--output", str(output_path)]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(project_root), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    with subprocess.Popen(command, cwd=project_root, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
        for line in process.stdout:
            LOG.info("[PyRadiomics] %s", line.rstrip())
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"PyRadiomics子进程失败，退出码={code}；请查看上方日志")
    if not output_path.is_file():
        raise FileNotFoundError(f"PyRadiomics未生成特征文件: {output_path}")
    return json.loads(output_path.read_text(encoding="utf-8"))


def classify_nodes(nodes, native_instances, native_ct_path, model_path,
                   radiomics_config_path, native_node_dir):
    package = joblib.load(model_path)
    if not isinstance(package, dict) or not {"model", "scaler", "features"} <= package.keys():
        raise ValueError("LightGBM模型包必须包含model、scaler、features")
    model, scaler, features = (package[key] for key in ("model", "scaler", "features"))
    features = list(features)
    if len(features) != SETTINGS["classify"]["expected_selected_features"] or len(set(features)) != len(features):
        raise ValueError("模型包中的15个特征名称数量或唯一性不正确")
    if list(model.classes_) != [0, 1]:
        raise ValueError("LightGBM类别必须按[0,1]排列，1表示恶性")
    config_dir = Path(radiomics_config_path).parent
    normalization = json.loads((config_dir / "feature_normalization.json").read_text(encoding="utf-8"))
    if normalization["features"] != features:
        raise ValueError("第一层标准化特征顺序与模型包不一致")
    first_mean = np.asarray(normalization["mean"], dtype=float)
    first_scale = np.asarray(normalization["scale"], dtype=float)
    if (len(first_mean) != len(features) or len(first_scale) != len(features)
            or not np.isfinite(first_mean).all() or not np.isfinite(first_scale).all()
            or np.any(first_scale <= 0)):
        raise ValueError("第一层标准化参数无效")
    ct = load_nii(native_ct_path)
    if native_instances.shape != ct.shape:
        raise ValueError("native实例图与原始全画幅CT尺寸不一致")
    native_node_dir = Path(native_node_dir)
    native_node_dir.mkdir(parents=True, exist_ok=True)
    requests = []
    for node in nodes:
        one = native_instances == node.instance_id
        if not np.any(one):
            raise ValueError(f"{node.node_id}在native网格消失；请检查1mm→native最近邻重采样")
        mask_path = native_node_dir / f"{node.node_id}.nii.gz"
        save_nii(one, ct.affine, mask_path, np.uint8)
        assert_same_grid(ct, load_nii(mask_path))
        node.centroid_native_world = get_centroid_world(one, ct.affine).tolist()
        requests.append({"node_id": node.node_id, "mask": str(mask_path)})
    if not nodes:
        LOG.warning("本病例无有效节点；跳过PyRadiomics并生成空节点报告")
        return nodes
    project_root = config_dir.parent
    manifest_path = native_node_dir.parent / "radiomics_request.json"
    output_path = native_node_dir.parent / "radiomics_features.json"
    manifest_path.write_text(json.dumps({"ct": str(native_ct_path),
        "config": str(radiomics_config_path), "features": features,
        "nodes": requests}, ensure_ascii=False, indent=2), encoding="utf-8")
    result = _extract_features_in_legacy_env(manifest_path, output_path, project_root)
    by_id = {item["node_id"]: item for item in result["nodes"]}
    if set(by_id) != {node.node_id for node in nodes}:
        raise ValueError("PyRadiomics返回的节点ID与请求不一致")
    for node in nodes:
        item = by_id[node.node_id]
        values = item["selected"]
        if list(values) != features:
            raise ValueError(f"{node.node_id}的特征顺序与模型包不一致")
        node.extracted_feature_count = int(item["feature_count"])
        if node.extracted_feature_count != 1502:
            LOG.warning("%s提取到%d个特征，师姐记录为1502；15维特征仍需核查",
                        node.node_id, node.extracted_feature_count)
        raw = pd.DataFrame([[values[name] for name in features]], columns=features).astype(float)
        if not np.isfinite(raw.to_numpy()).all():
            raise ValueError(f"{node.node_id}的15维特征含NaN或Inf")
        # 80.py先标准化全部特征；训练脚本又拟合包内scaler。
        first = pd.DataFrame((raw.to_numpy() - first_mean) / first_scale, columns=features)
        scaled = scaler.transform(first)
        node.malignant_probability = float(model.predict_proba(scaled)[0, 1])
        if not 0 <= node.malignant_probability <= 1:
            raise ValueError(f"{node.node_id}的恶性概率不在[0,1]")
        node.malignancy = "malignant" if int(model.predict(scaled)[0]) == 1 else "benign"
        LOG.info("分类完成 %s：恶性概率=%.4f，判定=%s",
                 node.node_id, node.malignant_probability, node.malignancy)
    return nodes
