"""复用105.py的三维参考轴定义，代入固定二次分界线。"""

import numpy as np

from .config import BOUNDARY_12, BOUNDARY_23
from .nii_io import assert_same_grid, load_nii


def _extreme_world(mask_img, axis, choose_max):
    pts = np.argwhere(np.asanyarray(mask_img.dataobj) > 0)
    if not len(pts):
        raise ValueError("骨骼Mask为空，无法建立分级参考轴")
    xyz = pts @ mask_img.affine[:3, :3].T + mask_img.affine[:3, 3]
    return xyz[np.argmax(xyz[:, axis]) if choose_max else np.argmin(xyz[:, axis])]


def grade_nodes(nodes, clav_path, rib_path):
    clav, rib = load_nii(clav_path), load_nii(rib_path)
    assert_same_grid(clav, rib)
    # 师姐105.py：左锁骨内侧X最大、左第5肋外侧X最小。
    p_end = _extreme_world(clav, 0, True)
    p_start = _extreme_world(rib, 0, False)
    axis = p_end - p_start
    length2 = float(np.dot(axis, axis))
    if length2 < 1e-8:
        raise ValueError("骨骼极值点重合，无法建立分级参考轴")
    for node in nodes:
        vector = np.asarray(node.centroid_side_world) - p_start
        t = float(np.dot(vector, axis) / length2)
        d = float(np.linalg.norm(np.cross(axis, vector)) / np.sqrt(length2))
        a, b, c = BOUNDARY_12
        if d > a * t * t + b * t + c:
            level = "Level I"
        else:
            a, b, c = BOUNDARY_23
            level = "Level II" if d > a * t * t + b * t + c else "Level III"
        node.t, node.d_mm, node.level = t, d, level
    return {"p_start_mm": p_start.tolist(), "p_end_mm": p_end.tolist()}
