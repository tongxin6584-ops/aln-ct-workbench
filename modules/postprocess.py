"""用同ROI锁骨掩膜剔除质心高于锁骨最高点的连通域。"""

import logging

import numpy as np
from scipy import ndimage

from .nii_io import assert_same_grid, load_nii, save_nii, voxel_to_world

LOG = logging.getLogger(__name__)


def constrain_by_clavicle(raw_path, clav_path, output_path):
    raw, clav = load_nii(raw_path), load_nii(clav_path)
    assert_same_grid(raw, clav)
    mask = np.asanyarray(raw.dataobj) > 0
    clav_points = np.argwhere(np.asanyarray(clav.dataobj) > 0)
    if not len(clav_points):
        raise ValueError("ROI锁骨Mask为空，无法执行假阳性约束")
    # RAS物理Z坐标可处理非轴对齐affine。
    top_z = float(np.max(clav_points @ clav.affine[2, :3] + clav.affine[2, 3]))
    labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    kept = mask.copy()
    removed = 0
    for label_id in range(1, count + 1):
        center_voxel = ndimage.center_of_mass(mask, labels, label_id)
        center_z = float(voxel_to_world(raw.affine, center_voxel)[2])
        if center_z > top_z:
            kept[labels == label_id] = False
            removed += 1
    save_nii(kept, raw.affine, output_path, np.uint8)
    LOG.warning("锁骨约束完成：候选连通域=%d，剔除=%d，保留=%d，锁骨最高Z=%.2f mm",
                count, removed, count - removed, top_z)
    if not np.any(kept):
        LOG.warning("锁骨约束后Mask为空；后续将生成零节点报告")
    return output_path
