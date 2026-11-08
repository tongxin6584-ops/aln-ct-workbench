"""不启动GPU即可核对报告一致性与断线任务恢复。"""

import json
import tempfile
import unittest
from pathlib import Path

from modules.contracts import NodeResult
from modules.jobs import job_status, list_results
from modules.report import render_markdown, write_report


class ReportAndJobTest(unittest.TestCase):
    def test_report_formats_have_all_nodes_and_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.nii.gz"
            source.write_bytes(b"nifti-placeholder")
            nodes = [NodeResult(node_id=f"L_node_{i:03}", instance_id=i, side="left", local_id=i,
                                voxel_count_1mm=20, volume_mm3=20.0,
                                centroid_side_world=[1.0, 2.0, 3.0],
                                centroid_native_world=[1.0, 2.0, 3.0],
                                malignant_probability=0.25, malignancy="benign", level="I")
                     for i in range(1, 16)]
            output = root / "results" / "case" / "run"
            json_path, md_path, csv_path = write_report(
                nodes, "case", "left", "run", {"native_spacing": [1, 1, 1]}, {}, output,
                input_path=source, parameters={"test": True}, started_at_utc="2026-11-06T01:00:00+00:00",
                finished_at_utc="2026-11-06T01:02:00+00:00", duration_seconds=120)
            report = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = md_path.read_text(encoding="utf-8")
            self.assertEqual(report["node_count"], 15)
            self.assertEqual(report["input"]["sha256"], __import__("hashlib").sha256(source.read_bytes()).hexdigest())
            self.assertEqual(report["timing"]["duration_seconds"], 120)
            self.assertEqual(len(csv_path.read_text(encoding="utf-8-sig").splitlines()), 16)
            self.assertEqual(sum(1 for line in markdown.splitlines() if line.startswith("| L_node_")), 15)
            self.assertEqual(list_results(root), [json_path])

    def test_legacy_report_does_not_invent_provenance(self):
        markdown = render_markdown({"patient_id": "case", "nodes": [{"node_id": "L_node_001"}]})
        self.assertIn("未记录", markdown)
        self.assertIn("L_node_001", markdown)

    def test_job_status_recovers_from_disk_after_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = root / "workspace" / "_jobs"
            jobs.mkdir(parents=True)
            metadata = jobs / "web_001.json"
            metadata.write_text(json.dumps({"run_id": "web_001", "patient_id": "case",
                                            "pid": 99999999}), encoding="utf-8")
            (jobs / "web_001.exit.json").write_text(json.dumps({
                "returncode": 1, "finished_at_utc": "2026-11-06T01:00:00+00:00"}), encoding="utf-8")
            self.assertEqual(job_status(root, metadata)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
