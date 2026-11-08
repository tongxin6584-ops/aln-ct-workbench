"""从工程根目录的pipeline.yaml读取第1阶段参数。"""

from pathlib import Path

import yaml

_PATH = Path(__file__).resolve().parents[1] / "configs" / "pipeline.yaml"
with _PATH.open(encoding="utf-8") as _file:
    SETTINGS = yaml.safe_load(_file)
    _PREPROCESS = SETTINGS["preprocess"]

HU_THRESHOLD = float(_PREPROCESS["hu_threshold"])
BED_Z_RATIO_THRESHOLD = float(_PREPROCESS["bed_z_ratio_threshold"])
Y_GAUSSIAN_SIGMA = float(_PREPROCESS["y_gaussian_sigma"])
Y_THRESHOLD_RATIO = float(_PREPROCESS["y_threshold_ratio"])
TARGET_SPACING_MM = tuple(float(x) for x in _PREPROCESS["target_spacing_mm"])
MIN_NODE_VOXELS = int(SETTINGS["nodes"]["min_voxels"])
BOUNDARY_12 = tuple(float(x) for x in SETTINGS["grade"]["boundary_12"])
BOUNDARY_23 = tuple(float(x) for x in SETTINGS["grade"]["boundary_23"])
