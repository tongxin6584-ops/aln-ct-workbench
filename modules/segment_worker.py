"""只在aln-unetrpp环境运行；直接调用师姐保留的推理函数。"""

import argparse
import shutil
from pathlib import Path


def run_worker(input_path, output_path, model_dir, folds, checkpoint, tta,
               mixed_precision, all_in_gpu, step_size):
    import nibabel as nib
    import numpy as np
    import torch
    from unetr_pp.inference.predict import predict_from_folder

    if not torch.cuda.is_available():
        raise RuntimeError("UNETR++原代码不支持CPU推理；当前环境未检测到CUDA GPU")
    input_path, output_path, model_dir = map(Path, (input_path, output_path, model_dir))
    job = output_path.parent / "_unetrpp_job"
    input_dir, prediction_dir = job / "input", job / "prediction"
    input_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, input_dir / "aln_0000.nii.gz")
    # 保留原代码的预处理、五折集成、TTA和NIfTI导出逻辑。
    predict_from_folder(str(model_dir), str(input_dir), str(prediction_dir),
                        folds=folds, save_npz=False,
                        num_threads_preprocessing=1, num_threads_nifti_save=1,
                        lowres_segmentations=None, part_id=0, num_parts=1,
                        tta=tta, mixed_precision=mixed_precision, overwrite_existing=True,
                        mode="normal", overwrite_all_in_gpu=all_in_gpu,
                        step_size=step_size, checkpoint_name=checkpoint)
    prediction = prediction_dir / "aln.nii.gz"
    if not prediction.is_file():
        raise FileNotFoundError(f"UNETR++未输出预测文件: {prediction}")
    reference, raw = nib.load(str(input_path)), nib.load(str(prediction))
    if reference.shape != raw.shape or not np.allclose(reference.affine, raw.affine, atol=1e-4):
        raise ValueError("UNETR++输出网格与输入ROI不一致")
    mask = np.asanyarray(raw.dataobj)
    if not np.isin(np.unique(mask), [0, 1]).all():
        raise ValueError("UNETR++输出含0/1之外的标签；请核对模型与plans.pkl")
    nib.save(nib.Nifti1Image((mask > 0).astype(np.uint8), reference.affine), str(output_path))
    print(f"[完成] UNETR++预测体素={int(np.count_nonzero(mask))}，输出={output_path}", flush=True)


if __name__ == "__main__":
    def boolean(value):
        if value not in ("true", "false"):
            raise argparse.ArgumentTypeError("布尔参数只能是true或false")
        return value == "true"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--checkpoint", default="model_best")
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument("--tta", type=boolean, default=True)
    parser.add_argument("--mixed-precision", type=boolean, default=True)
    parser.add_argument("--all-in-gpu", type=boolean, default=False)
    args = parser.parse_args()
    run_worker(args.input, args.output, args.model_dir,
               [int(i) for i in args.folds.split(",")], args.checkpoint,
               args.tta, args.mixed_precision, args.all_in_gpu, args.step_size)
