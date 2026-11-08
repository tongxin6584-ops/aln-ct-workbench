"""云端真实病例冒烟测试；设置ALN_TEST_CASE后才执行。"""

import os
import unittest
import uuid
from pathlib import Path


@unittest.skipUnless(os.environ.get("ALN_TEST_CASE"), "需在云端设置ALN_TEST_CASE")
class SmokeTest(unittest.TestCase):
    def test_one_case(self):
        from main import run_pipeline
        path = Path(os.environ["ALN_TEST_CASE"])
        result = run_pipeline(path, "smoke_case", os.environ.get("ALN_TEST_SIDE", "left"),
                              "smoke_" + uuid.uuid4().hex[:8])
        self.assertTrue((result / "smoke_case_report.json").is_file())
        self.assertTrue((result / "smoke_case_report.md").is_file())
        self.assertTrue((result / "aln_instances_native.nii.gz").is_file())


if __name__ == "__main__":
    unittest.main()
