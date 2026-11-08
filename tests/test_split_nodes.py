"""3D六连通应过滤9体素并保留10体素组件。"""

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from modules.split_nodes import split_nodes


class SplitNodesTest(unittest.TestCase):
    def test_minimum_voxel_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.zeros((10, 10, 10), dtype=np.uint8)
            mask[1, :, 1] = 1
            mask[8, :9, 8] = 1
            nib.save(nib.Nifti1Image(mask, np.eye(4)), root / "side.nii.gz")
            nodes = split_nodes(root / "side.nii.gz", root / "instances.nii.gz",
                                "left", "case001", 1)
            labels = np.asanyarray(nib.load(root / "instances.nii.gz").dataobj)
            self.assertEqual(len(nodes), 1)
            self.assertEqual(nodes[0].voxel_count_1mm, 10)
            self.assertEqual(int(np.count_nonzero(labels)), 10)


if __name__ == "__main__":
    unittest.main()
