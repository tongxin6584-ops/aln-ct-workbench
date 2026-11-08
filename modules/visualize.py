"""在用户原始全画幅CT上叠加native实例标签，生成预览截图。"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .nii_io import load_nii


def _plane(data, labels, axis, index):
    return np.rot90(np.take(data, index, axis=axis)), np.rot90(np.take(labels, index, axis=axis))


def write_previews(native_ct_path, instance_path, nodes, output_dir):
    ct_img, mask_img = load_nii(native_ct_path), load_nii(instance_path)
    if ct_img.shape != mask_img.shape or not np.allclose(ct_img.affine, mask_img.affine, atol=1e-4):
        raise ValueError("可视化时原始CT与实例标签图不对齐")
    ct = np.asarray(ct_img.dataobj, dtype=np.float32)
    labels = np.asanyarray(mask_img.dataobj).astype(np.int32)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (ax, title) in enumerate(zip(axes, ("Sagittal", "Coronal", "Axial"))):
        counts = np.count_nonzero(labels, axis=tuple(i for i in range(3) if i != axis))
        index = int(np.argmax(counts)) if np.any(counts) else ct.shape[axis] // 2
        background, overlay = _plane(ct, labels, axis, index)
        ax.imshow(background, cmap="gray", vmin=-200, vmax=300)
        ax.imshow(np.ma.masked_equal(overlay, 0), cmap="tab20", alpha=0.55,
                  vmin=0, vmax=max(20, len(nodes)))
        ax.set_title(f"{title} {index}")
        ax.axis("off")
    fig.tight_layout()
    overview = output_dir / "overview.png"
    fig.savefig(overview, dpi=120)
    plt.close(fig)
    count = max(1, len(nodes))
    cols = min(4, count)
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, node in zip(axes.flat, nodes):
        positions = np.argwhere(labels == node.instance_id)
        if not len(positions):
            continue
        z = int(round(float(positions[:, 2].mean())))
        ax.imshow(np.rot90(ct[:, :, z]), cmap="gray", vmin=-200, vmax=300)
        ax.imshow(np.ma.masked_equal(np.rot90((labels[:, :, z] == node.instance_id).astype(np.uint8)), 0),
                  cmap="autumn", alpha=0.6, vmin=0, vmax=1)
        ax.set_title(f"{node.node_id} | {node.level} | p={node.malignant_probability:.2f}")
        ax.axis("off")
    if not nodes:
        axes[0, 0].text(0.5, 0.5, "No nodes found", ha="center", va="center")
    fig.tight_layout()
    montage = output_dir / "node_montage.png"
    fig.savefig(montage, dpi=120)
    plt.close(fig)
    return overview, montage
