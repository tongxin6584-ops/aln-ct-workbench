"""锁骨上方判据按物理Z比较连通域质心。"""

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from modules.postprocess import constrain_by_clavicle


class PostprocessTest(unittest.TestCase):
    def test_remove_only_component_above_clavicle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            affine = np.eye(4)
            mask = np.zeros((5, 5, 10), dtype=np.uint8)
            mask[1, 1, 1:3] = 1
            mask[3, 3, 7:9] = 1
            clav = np.zeros_like(mask)
            clav[2, 2, 5] = 1
            nib.save(nib.Nifti1Image(mask, affine), root / "raw.nii.gz")
            nib.save(nib.Nifti1Image(clav, affine), root / "clav.nii.gz")
            constrain_by_clavicle(root / "raw.nii.gz", root / "clav.nii.gz", root / "out.nii.gz")
            result = np.asanyarray(nib.load(root / "out.nii.gz").dataobj)
            self.assertEqual(int(result.sum()), 2)
            self.assertEqual(int(result[3, 3, 7]), 0)


if __name__ == "__main__":
    unittest.main()
