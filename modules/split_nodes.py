"""沿用78.py的三维6连通和至少10体素判据。"""

import logging
from pathlib import Path

import numpy as np
from scipy import ndimage

from .config import MIN_NODE_VOXELS
from .contracts import NodeResult
from .nii_io import load_nii, save_nii, get_centroid_world

LOG = logging.getLogger(__name__)


def split_nodes(side_mask_path, output_path, side, patient_id, first_instance_id):
    image = load_nii(side_mask_path)
    mask = np.asanyarray(image.dataobj) > 0
    labeled, found = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    counts = np.bincount(labeled.ravel())
    instances = np.zeros(image.shape, dtype=np.int32)
    nodes = []
    node_dir = Path(output_path).parent / "nodes"
    node_dir.mkdir(parents=True, exist_ok=True)
    voxel_mm3 = abs(float(np.linalg.det(image.affine[:3, :3])))
    prefix = "L" if side == "left" else "R"
    for source_id in range(1, found + 1):
        count = int(counts[source_id])
        if count < MIN_NODE_VOXELS:
            continue
        local_id = len(nodes) + 1
        instance_id = first_instance_id + len(nodes)
        one = labeled == source_id
        instances[one] = instance_id
        node_id = f"{prefix}_node_{local_id:03d}"
        save_nii(one, image.affine, node_dir / f"{node_id}.nii.gz", np.uint8)
        nodes.append(NodeResult(node_id, instance_id, side, local_id, count,
                                count * voxel_mm3,
                                get_centroid_world(one, image.affine).tolist()))
    save_nii(instances, image.affine, output_path, np.int32)
    LOG.warning("单体拆分完成：原连通域=%d，保留≥%d体素=%d，过滤=%d",
                found, MIN_NODE_VOXELS, len(nodes), found - len(nodes))
    return nodes
