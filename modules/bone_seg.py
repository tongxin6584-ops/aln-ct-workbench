"""对去床板后的单侧CT调用TotalSegmentator Python API。"""

from pathlib import Path

import numpy as np

from .nii_io import assert_same_grid, load_nii


def extract_bones(ct_path, clavicle_path, rib5_path):
    from totalsegmentator.python_api import totalsegmentator
    output_dir = Path(clavicle_path).parent / "_totalseg"
    output_dir.mkdir(parents=True, exist_ok=True)
    # 对应原process.sh里的 -rsr <骨骼名>，合并为同一次推理。
    totalsegmentator(input=str(ct_path), output=str(output_dir),
                     task="total", roi_subset=["clavicula_left", "rib_left_5"],
                     ml=False, fast=False, device="gpu")
    expected = {
        "clavicula_left": Path(clavicle_path),
        "rib_left_5": Path(rib5_path),
    }
    from .nii_io import save_nii
    ct = load_nii(ct_path)
    for name, destination in expected.items():
        source = output_dir / (name + ".nii.gz")
        if not source.is_file():
            raise FileNotFoundError(f"TotalSegmentator未输出{name}: {source}")
        mask = load_nii(source)
        assert_same_grid(ct, mask)
        if not np.any(np.asanyarray(mask.dataobj) > 0):
            raise ValueError(f"骨骼Mask为空: {name}")
        save_nii(np.asanyarray(mask.dataobj) > 0, ct.affine, destination, np.uint8)
    return expected
