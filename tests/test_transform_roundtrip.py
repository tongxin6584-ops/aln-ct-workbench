"""验证左右侧镜像与Y/Z裁剪可以逐体素回填。"""

import unittest

import numpy as np

from modules.geometry import crop_array, roi_to_side, side_to_canonical, split_side


class TransformRoundtripTest(unittest.TestCase):
    def test_left_and_right_on_odd_width(self):
        full = np.arange(5 * 8 * 9, dtype=np.int32).reshape(5, 8, 9)
        affine = np.eye(4)
        for side in ("left", "right"):
            half, half_affine, info = split_side(full, affine, side)
            crop, crop_affine = crop_array(half, half_affine, (0, 1, 2),
                                           (half.shape[0], 7, 8))
            np.testing.assert_array_equal(crop_affine[:3, 3], [0, 1, 2])
            record = {**info, "side_shape": list(half.shape),
                      "y_start": 1, "y_stop": 7, "z_start": 2, "z_stop": 8}
            recovered = side_to_canonical(roi_to_side(crop, record), record, full.shape)
            expected = np.zeros_like(full)
            expected[info["x_start"]:info["x_stop"], 1:7, 2:8] = full[
                info["x_start"]:info["x_stop"], 1:7, 2:8]
            np.testing.assert_array_equal(recovered, expected)

    def test_resampled_labels_return_to_native_grid(self):
        try:
            import nibabel as nib
        except ImportError:
            self.skipTest("需要aln-app环境中的nibabel")
        from modules.geometry import canonical_to_native
        native_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        native = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.uint8), native_affine)
        canonical = np.zeros((7, 7, 7), dtype=np.uint8)
        canonical[2:5, 2:5, 2:5] = 3
        recovered = canonical_to_native(canonical, np.eye(4), native, True)
        self.assertEqual(recovered.shape, native.shape)
        self.assertEqual(int(recovered[1, 1, 1]), 3)
        self.assertEqual(int(recovered[3, 3, 3]), 0)


if __name__ == "__main__":
    unittest.main()
