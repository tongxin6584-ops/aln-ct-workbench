"""统一 NIfTI 读写及网格检查。"""

from pathlib import Path

import nibabel as nib
import numpy as np


def load_nii(path):
    return nib.load(str(path))


def save_nii(data, affine, path, dtype=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(data, dtype=dtype)
    nib.save(nib.Nifti1Image(array, affine), str(path))
    return path


def assert_same_grid(a, b, atol=1e-4):
    if a.shape != b.shape or not np.allclose(a.affine, b.affine, atol=atol):
        raise ValueError("CT 与 Mask 不在同一体素网格")


def voxel_to_world(affine, voxel):
    """把连续体素坐标转换为RAS物理坐标，单位mm。"""
    return (np.asarray(affine) @ np.r_[voxel, 1.0])[:3]


def get_centroid_world(mask, affine):
    points = np.argwhere(mask)
    if len(points) == 0:
        raise ValueError("空Mask没有质心")
    return voxel_to_world(affine, points.mean(axis=0))
