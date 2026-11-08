"""固定二次分界线的Level I/II/III判定。"""

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from modules.contracts import NodeResult
from modules.grade import grade_nodes


class GradingTest(unittest.TestCase):
    def test_three_levels_with_known_reference_axis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rib = np.zeros((11, 2, 2), dtype=np.uint8)
            clav = np.zeros_like(rib)
            rib[0, 0, 0] = 1
            clav[10, 0, 0] = 1
            nib.save(nib.Nifti1Image(rib, np.eye(4)), root / "rib.nii.gz")
            nib.save(nib.Nifti1Image(clav, np.eye(4)), root / "clav.nii.gz")
            nodes = [NodeResult(str(i), i, "left", i, 10, 10.0, [5.0, y, 0.0])
                     for i, y in enumerate((100.0, 30.0, 0.0), 1)]
            grade_nodes(nodes, root / "clav.nii.gz", root / "rib.nii.gz")
            self.assertEqual([node.level for node in nodes], ["Level I", "Level II", "Level III"])
            self.assertTrue(all(abs(node.t - 0.5) < 1e-12 for node in nodes))


if __name__ == "__main__":
    unittest.main()
