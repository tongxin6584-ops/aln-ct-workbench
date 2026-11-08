"""输入校验与仅在必要时执行的1 mm重采样。"""

import logging

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from .config import TARGET_SPACING_MM
from .nii_io import load_nii

LOG = logging.getLogger(__name__)


def prepare_input(input_path, resampled_path):
    img = load_nii(input_path)
    if len(img.shape) != 3:
        raise ValueError("输入必须为3D NIfTI")
    affine = img.affine
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-8:
        raise ValueError("affine 缺失、非有限或不可逆")
    data = np.asarray(img.dataobj)
    if not np.isfinite(data).all():
        raise ValueError("CT 含 NaN 或 Inf")
    spacing = nib.affines.voxel_sizes(affine)
    metadata = {
        "native_shape": list(img.shape),
        "native_affine": affine.tolist(),
        "native_spacing": spacing.tolist(),
        "native_orientation": "".join(nib.aff2axcodes(affine)),
        "working_shape": list(img.shape),
        "working_affine": affine.tolist(),
        "native_to_ras_ornt": nib.orientations.ornt_transform(
            nib.orientations.io_orientation(affine),
            nib.orientations.axcodes2ornt(("R", "A", "S"))).tolist(),
        "resampled_to_1mm": False,
    }
    if np.allclose(spacing, TARGET_SPACING_MM, atol=1e-3):
        return img, metadata
    LOG.warning("⚠️ 自动重采样：%s mm → 1×1×1 mm³", spacing.tolist())
    image = sitk.ReadImage(str(input_path))
    old_size = image.GetSize()
    old_spacing = image.GetSpacing()
    # 保持首末体素中心覆盖的物理范围，不额外扩张一层边界。
    new_size = [max(1, int(round((n - 1) * s / t)) + 1)
                for n, s, t in zip(old_size, old_spacing, TARGET_SPACING_MM)]
    resampled = sitk.Resample(image, new_size, sitk.Transform(), sitk.sitkBSpline,
                              image.GetOrigin(), TARGET_SPACING_MM, image.GetDirection(), -1000.0,
                              sitk.sitkFloat32)
    sitk.WriteImage(resampled, str(resampled_path))
    result = load_nii(resampled_path)
    metadata["resampled_to_1mm"] = True
    metadata["working_shape"] = list(result.shape)
    metadata["working_affine"] = result.affine.tolist()
    return result, metadata
