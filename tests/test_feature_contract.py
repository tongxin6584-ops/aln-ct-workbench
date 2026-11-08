"""验证重建的首层标准化与已有15特征模型资产一致。"""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FeatureContractTest(unittest.TestCase):
    def test_first_standardization_and_model_feature_order(self):
        asset = json.loads((ROOT / "configs" / "feature_normalization.json").read_text(encoding="utf-8"))
        names, mean, scale = asset["features"], asset["mean"], asset["scale"]
        self.assertEqual((len(names), len(mean), len(scale)), (15, 15, 15))
        # 原训练CSV第一行的同一特征，检查原值到80.py标准化值的映射。
        self.assertAlmostEqual((-209.76176963403032 - mean[0]) / scale[0],
                               0.18634079825067243, places=12)
        package = (ROOT / "models" / "Best_Model_LightGBM.pkl").read_bytes()
        positions = [package.find(name.encode()) for name in names]
        self.assertTrue(all(position >= 0 for position in positions))
        self.assertEqual(positions, sorted(positions))


if __name__ == "__main__":
    unittest.main()
