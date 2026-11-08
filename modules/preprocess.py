"""封装师姐的60/45/62/66.py，运行到UNETR++输入ROI为止。"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d

from .bone_seg import extract_bones
from .config import BED_Z_RATIO_THRESHOLD, HU_THRESHOLD, Y_GAUSSIAN_SIGMA, Y_THRESHOLD_RATIO
from .geometry import crop_array, split_side
from .nii_io import assert_same_grid, load_nii, save_nii
from .status import stage
from .validate import prepare_input

LOG = logging.getLogger(__name__)


def remove_bed(data):
    """原45.py的矢状面二值化→开运算→连通域→非主体长Z区域置气。"""
    clean = data.copy()
    nz = data.shape[2]
    for x in range(data.shape[0]):
        opened = ndimage.binary_opening(data[x] > HU_THRESHOLD,
                                         structure=np.ones((3, 3)), iterations=1)
        labels, count = ndimage.label(opened)
        if count == 0:
            continue
        sizes = np.bincount(labels.ravel())
        largest = int(np.argmax(sizes[1:]) + 1)
        for label_idx, bounds in enumerate(ndimage.find_objects(labels), start=1):
            if bounds is None or label_idx == largest:
                continue
            if bounds[1].stop - bounds[1].start > nz * BED_Z_RATIO_THRESHOLD:
                clean[x][labels == label_idx] = -1000
    return clean


def find_y_bounds(data):
    """原66.py：灰度投影、sigma=2、中点双向寻找动态阈值。"""
    curve = gaussian_filter1d(np.sum(data, axis=(0, 2), dtype=np.float64), sigma=Y_GAUSSIAN_SIGMA)
    middle = len(curve) // 2
    threshold = float(np.min(curve) + Y_THRESHOLD_RATIO * (np.max(curve) - np.min(curve)))
    start, end = 0, len(curve) - 1
    for i in range(middle, -1, -1):
        if curve[i] < threshold:
            start = i
            break
    for i in range(middle, len(curve)):
        if curve[i] < threshold:
            end = i
            break
    if end < start:
        raise ValueError(f"Y轴投影边界无效：{start}, {end}")
    return start, end


def _save_crop(data, affine, path, z_start, y_start, y_stop, dtype):
    z_data, z_affine = crop_array(data, affine, (0, 0, z_start),
                                  (data.shape[0], data.shape[1], data.shape[2]))
    roi, roi_affine = crop_array(z_data, z_affine, (0, y_start, 0),
                                 (z_data.shape[0], y_stop, z_data.shape[2]))
    save_nii(roi, roi_affine, path, dtype)


def run_preprocessing(input_path, patient_id, target_side, project_root, run_id="run_001"):
    """阶段入口；完成后停在06_roi_*，不调用分割。"""
    if target_side not in {"left", "right", "bilateral"}:
        raise ValueError("target_side必须为left/right/bilateral")
    if not patient_id or not patient_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("patient_id只能含字母、数字、下划线、连字符")
    if not run_id or not run_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_id只能含字母、数字、下划线、连字符")
    input_path = Path(input_path).resolve(strict=True)
    if not input_path.name.endswith((".nii", ".nii.gz")):
        raise ValueError("输入文件扩展名必须为.nii或.nii.gz")
    run_dir = Path(project_root).resolve() / "workspace" / patient_id / run_id
    if run_dir.exists():
        raise FileExistsError(f"本次运行目录已存在，避免覆盖：{run_dir}")
    native_path = run_dir / "01_native_full" / "input.nii.gz"
    native_path.parent.mkdir(parents=True)
    with stage("01 输入校验及非1mm影像重采样"):
        if input_path.name.endswith(".nii.gz"):
            shutil.copy2(input_path, native_path)
        else:
            nib.save(nib.load(str(input_path)), str(native_path))
        working_img, metadata = prepare_input(native_path, run_dir / "01_native_full" / "resampled_1mm.nii.gz")
        LOG.warning("输入spacing=%s mm；自动重采样=%s", metadata["native_spacing"],
                    metadata["resampled_to_1mm"])
    with stage("02 RAS方向标准化"):
        ras_path = run_dir / "02_ras_full" / "ct_ras.nii.gz"
        ras_path.parent.mkdir(parents=True)
        ras_img = nib.as_closest_canonical(working_img)
        ras_data = np.asarray(ras_img.dataobj, dtype=np.float32)
        if nib.aff2axcodes(ras_img.affine) != ("R", "A", "S"):
            raise ValueError("无法将输入标准化为RAS轴向")
        save_nii(ras_data, ras_img.affine, ras_path, np.float32)
        metadata["ras_shape"] = list(ras_data.shape)
        metadata["ras_affine"] = ras_img.affine.tolist()
        metadata["working_to_ras_ornt"] = nib.orientations.ornt_transform(
            nib.orientations.io_orientation(working_img.affine),
            nib.orientations.io_orientation(ras_img.affine)).tolist()
    sides = ["left", "right"] if target_side == "bilateral" else [target_side]
    records = {}
    for side in sides:
        side_dir = run_dir / side
        with stage(f"02 {side}侧切半与镜像"):
            side_data, side_affine, side_info = split_side(ras_data, ras_img.affine, side)
            save_nii(side_data, side_affine, side_dir / "03_side_ct.nii.gz", np.float32)
        with stage(f"03 {side}侧去床板"):
            clean = remove_bed(side_data)
            clean_path = save_nii(clean, side_affine, side_dir / "04_no_bed.nii.gz", np.float32)
        with stage(f"04 {side}侧TotalSegmentator骨骼提取"):
            clav_path = side_dir / "05_clavicle.nii.gz"
            rib_path = side_dir / "05_rib5.nii.gz"
            extract_bones(clean_path, clav_path, rib_path)
            clav, rib = load_nii(clav_path), load_nii(rib_path)
            assert_same_grid(load_nii(clean_path), clav)
            assert_same_grid(load_nii(clean_path), rib)
            clav_data = np.asanyarray(clav.dataobj)
            rib_data = np.asanyarray(rib.dataobj)
        with stage(f"05 {side}侧第5肋骨Z裁剪与Y投影裁剪"):
            rib_z = np.where(rib_data > 0)[2]
            if rib_z.size == 0:
                raise ValueError(f"{side}的第5肋骨Mask为空")
            z_start = int(rib_z.min())
            z_ct = clean[:, :, z_start:]
            y_start, y_end = find_y_bounds(z_ct)
            y_stop = y_end + 1  # 原66.py使用闭区间，Python切片需+1。
            if not np.any(clav_data[:, y_start:y_stop, z_start:] > 0):
                raise ValueError(f"{side}的锁骨在Y/Z裁剪后为空；请检查输入和裁剪边界")
            if not np.any(rib_data[:, y_start:y_stop, z_start:] > 0):
                raise ValueError(f"{side}的第5肋骨在Y/Z裁剪后为空；请检查输入和裁剪边界")
            for data, path, dtype in (
                (clean, side_dir / "06_roi_ct.nii.gz", np.float32),
                (clav_data, side_dir / "06_roi_clav.nii.gz", np.uint8),
                (rib_data, side_dir / "06_roi_rib5.nii.gz", np.uint8),
            ):
                _save_crop(data, side_affine, path, z_start, y_start, y_stop, dtype)
        records[side] = {**side_info, "side_shape": list(side_data.shape),
                         "side_affine": side_affine.tolist(),
                         "z_start": z_start, "z_stop": int(clean.shape[2]),
                         "y_start": y_start, "y_stop": y_stop,
                         "roi_shape": list(load_nii(side_dir / "06_roi_ct.nii.gz").shape)}
        LOG.info("%s: X[%s:%s] Z[%s:%s] Y[%s:%s] ROI=%s", side,
                 side_info["x_start"], side_info["x_stop"], z_start, clean.shape[2],
                 y_start, y_stop, records[side]["roi_shape"])
    meta_dir = run_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "input.json").write_text(json.dumps({"patient_id": patient_id,
        "target_side": target_side, "source_path": str(input_path)}, ensure_ascii=False, indent=2), encoding="utf-8")
    (meta_dir / "transform.json").write_text(json.dumps({"input": metadata,
        "sides": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="仅运行输入校验、侧别标准化、去床板、骨骼提取和ROI裁剪")
    parser.add_argument("--input", required=True, help="静脉期胸部CT .nii或.nii.gz")
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--side", choices=("left", "right", "bilateral"), required=True)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--run-id", default="run_001")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(run_preprocessing(args.input, args.patient_id, args.side, args.project_root, args.run_id))


if __name__ == "__main__":
    main()
