"""Python 3.8环境中的79.py特征提取工人；不加载LightGBM模型。"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml
import radiomics
from radiomics import featureextractor

logging.getLogger("radiomics").setLevel(logging.ERROR)


def run(manifest_path, output_path):
    request = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    config = yaml.safe_load(Path(request["config"]).read_text(encoding="utf-8"))
    image_types = config.pop("image_types")
    extractor = featureextractor.RadiomicsFeatureExtractor(**config)
    extractor.enableAllFeatures()
    for name, options in image_types.items():
        extractor.enableImageTypeByName(name, customArgs=options)
    ct = sitk.ReadImage(request["ct"])
    results = []
    for node in request["nodes"]:
        mask = sitk.ReadImage(node["mask"])
        if (ct.GetSize() != mask.GetSize()
                or not np.allclose(ct.GetSpacing(), mask.GetSpacing(), atol=1e-4)
                or not np.allclose(ct.GetOrigin(), mask.GetOrigin(), atol=1e-4)
                or not np.allclose(ct.GetDirection(), mask.GetDirection(), atol=1e-4)):
            raise ValueError("CT与%s的Mask不在同一SimpleITK网格" % node["node_id"])
        try:
            extracted = extractor.execute(ct, mask)
        except Exception as error:
            raise RuntimeError("%s特征提取失败: %s" % (node["node_id"], error)) from error
        values = {key: value.item() if hasattr(value, "item") else value
                  for key, value in extracted.items() if not key.startswith("diagnostics_")}
        missing = [name for name in request["features"] if name not in values]
        if missing:
            raise ValueError("%s缺少特征: %s" % (node["node_id"], missing))
        selected = {name: float(values[name]) for name in request["features"]}
        if not np.isfinite(list(selected.values())).all():
            raise ValueError("%s的15维特征含NaN或Inf" % node["node_id"])
        results.append({"node_id": node["node_id"], "feature_count": len(values),
                        "selected": selected})
        print("[完成] %s PyRadiomics特征=%d" % (node["node_id"], len(values)), flush=True)
    Path(output_path).write_text(json.dumps({"nodes": results, "radiomics_version": radiomics.__version__},
                                            ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.manifest, args.output)
