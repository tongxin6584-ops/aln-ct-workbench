"""与原始全画幅可逆的切片及镜像信息。"""

import numpy as np


def crop_array(data, affine, starts, stops):
    starts = np.asarray(starts, dtype=int)
    stops = np.asarray(stops, dtype=int)
    if np.any(starts < 0) or np.any(stops > data.shape) or np.any(stops <= starts):
        raise ValueError("无效裁剪边界")
    cropped = data[tuple(slice(int(a), int(b)) for a, b in zip(starts, stops))]
    new_affine = np.array(affine, dtype=float, copy=True)
    new_affine[:3, 3] = affine[:3, :3] @ starts + affine[:3, 3]
    return cropped, new_affine


def split_side(data, affine, side):
    mid = data.shape[0] // 2
    if side == "left":
        if mid == 0:
            raise ValueError("X轴长度不足以切半")
        part, part_affine = crop_array(data, affine, (0, 0, 0), (mid, data.shape[1], data.shape[2]))
        return part, part_affine, {"x_start": 0, "x_stop": mid, "mirrored": False}
    if side != "right":
        raise ValueError("side 必须为 left 或 right")
    part, _ = crop_array(data, affine, (mid, 0, 0), data.shape)
    # 统一左侧形态是算法坐标系；回填时依靠 x_start/x_stop 和 mirrored，不使用此 affine 当作患者真实右侧坐标。
    mirrored = np.flip(part, axis=0)
    pseudo_affine = np.array(affine, copy=True)
    return mirrored, pseudo_affine, {"x_start": mid, "x_stop": data.shape[0], "mirrored": True}


def roi_to_side(roi, record):
    """逆Y/Z裁剪；输入和边界均来自同一侧的transform.json。"""
    side = np.zeros(record["side_shape"], dtype=roi.dtype)
    ys, ye, zs, ze = (record[k] for k in ("y_start", "y_stop", "z_start", "z_stop"))
    expected = (side.shape[0], ye - ys, ze - zs)
    if roi.shape != expected:
        raise ValueError(f"ROI尺寸{roi.shape}与记录的裁剪范围{expected}不符")
    side[:, ys:ye, zs:ze] = roi
    return side


def side_to_canonical(side, record, canonical_shape):
    """右侧逆X镜像，然后按原半侧X范围填回RAS全画幅。"""
    full = np.zeros(canonical_shape, dtype=side.dtype)
    xs, xe = record["x_start"], record["x_stop"]
    if side.shape != (xe - xs, *canonical_shape[1:]):
        raise ValueError("单侧Mask与RAS全画幅尺寸不匹配")
    full[xs:xe] = np.flip(side, axis=0) if record["mirrored"] else side
    return full


def canonical_to_native(data, canonical_affine, native_img, was_resampled):
    """方向变换逐体素还原；做过1mm重采样时再用最近邻映射回native网格。"""
    import nibabel as nib

    if was_resampled:
        from nibabel.processing import resample_from_to
        source = nib.Nifti1Image(data, canonical_affine)
        result = resample_from_to(source, (native_img.shape, native_img.affine), order=0)
        native_data = np.rint(np.asanyarray(result.dataobj)).astype(data.dtype)
    else:
        transform = nib.orientations.ornt_transform(
            nib.orientations.io_orientation(canonical_affine),
            nib.orientations.io_orientation(native_img.affine))
        native_data = nib.orientations.apply_orientation(data, transform)
    if native_data.shape != native_img.shape:
        raise ValueError("还原后Mask尺寸与原始CT不一致")
    return native_data
