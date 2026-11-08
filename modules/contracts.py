"""跨模块共享的单颗淋巴结结果。"""

from dataclasses import dataclass


@dataclass
class NodeResult:
    node_id: str
    instance_id: int
    side: str
    local_id: int
    voxel_count_1mm: int
    volume_mm3: float
    centroid_side_world: list[float]
    centroid_native_world: list[float] | None = None
    t: float | None = None
    d_mm: float | None = None
    level: str | None = None
    malignant_probability: float | None = None
    malignancy: str | None = None
    extracted_feature_count: int | None = None
