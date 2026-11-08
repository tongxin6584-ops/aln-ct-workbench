"""按transform.json逐级还原ROI、单侧、RAS全画幅与原始CT网格。"""

from pathlib import Path

import numpy as np

from .geometry import roi_to_side, side_to_canonical, canonical_to_native
from .nii_io import assert_same_grid, load_nii, save_nii


def restore_post_mask(post_mask_path, roi_ct_path, side_ct_path, record, output_path):
    post, roi_ct, side_ct = map(load_nii, (post_mask_path, roi_ct_path, side_ct_path))
    assert_same_grid(post, roi_ct)
    side_data = roi_to_side((np.asanyarray(post.dataobj) > 0).astype(np.uint8), record)
    if side_data.shape != side_ct.shape:
        raise ValueError("逆裁剪结果与单侧CT尺寸不一致")
    save_nii(side_data, side_ct.affine, output_path, np.uint8)
    return output_path


def combine_side_instances(side_instance_paths, records, ras_ct_path, output_path):
    ras = load_nii(ras_ct_path)
    combined = np.zeros(ras.shape, dtype=np.int32)
    for side, path in side_instance_paths.items():
        side_img = load_nii(path)
        restored = side_to_canonical(np.asanyarray(side_img.dataobj).astype(np.int32),
                                     records[side], ras.shape)
        if np.any((combined != 0) & (restored != 0)):
            raise ValueError("左右侧回填发生节点标签重叠")
        combined += restored
    save_nii(combined, ras.affine, output_path, np.int32)
    return combined


def save_native_instances(canonical_instances, ras_ct_path, native_ct_path,
                          was_resampled, output_path):
    ras, native = load_nii(ras_ct_path), load_nii(native_ct_path)
    data = canonical_to_native(canonical_instances, ras.affine, native, was_resampled)
    save_nii(data, native.affine, output_path, np.int32)
    assert_same_grid(native, load_nii(output_path))
    return data


def save_result_maps(canonical_instances, native_instances, nodes, ras_ct_path,
                     native_ct_path, was_resampled, run_dir, result_dir):
    run_dir, result_dir = Path(run_dir), Path(result_dir)
    canonical_dir, native_dir = run_dir / "11_canonical_results", run_dir / "12_native_results"
    ras, native = load_nii(ras_ct_path), load_nii(native_ct_path)
    maps = {"malignancy": np.zeros_like(canonical_instances, dtype=np.uint8),
            "levels": np.zeros_like(canonical_instances, dtype=np.uint8)}
    level_codes = {"Level I": 1, "Level II": 2, "Level III": 3}
    for node in nodes:
        region = canonical_instances == node.instance_id
        maps["malignancy"][region] = 2 if node.malignancy == "malignant" else 1
        maps["levels"][region] = level_codes[node.level]
    save_nii(canonical_instances > 0, ras.affine, canonical_dir / "aln_binary.nii.gz", np.uint8)
    save_nii(canonical_instances, ras.affine, canonical_dir / "instances.nii.gz", np.int32)
    save_nii(native_instances > 0, native.affine, native_dir / "aln_binary_native.nii.gz", np.uint8)
    save_nii(native_instances, native.affine, native_dir / "instances_native.nii.gz", np.int32)
    save_nii(native_instances, native.affine, result_dir / "aln_instances_native.nii.gz", np.int32)
    for name, canonical_data in maps.items():
        save_nii(canonical_data, ras.affine, canonical_dir / f"{name}.nii.gz", np.uint8)
        native_data = canonical_to_native(canonical_data, ras.affine, native, was_resampled)
        save_nii(native_data, native.affine, native_dir / f"{name}_native.nii.gz", np.uint8)
        save_nii(native_data, native.affine, result_dir / f"{name}_native.nii.gz", np.uint8)
        assert_same_grid(native, load_nii(result_dir / f"{name}_native.nii.gz"))
