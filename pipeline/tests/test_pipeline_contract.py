"""不依赖 checkpoint 的冻结语义与输出契约测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipeline.predict_three_layer import initialize_output, load_manifest


class PipelineContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = {
            "pipeline_version": "1.0.0",
            "project_root": "..",
            "label_mapping": {"0": "Alpha", "1": "Beta"},
            "layer1": {
                "checkpoints": [{"seed": seed} for seed in range(3)],
                "ensemble": {"method": "mean_score", "threshold": 0.5},
            },
            "layer2": {},
            "layer3": {
                "checkpoints": [
                    {"seed": seed, "threshold": 0.5 + seed * 0.1} for seed in range(3)
                ]
            },
        }

    def test_output_names_expose_beta_semantics(self) -> None:
        output = initialize_output(pd.DataFrame({"ID": [1]}), self.manifest)
        self.assertIn("Layer3_Seed0_Beta_Score", output)
        self.assertIn("Stereoselectivity_Mean_Beta_Score", output)
        self.assertNotIn("Alpha_Probability", output)
        self.assertEqual(output.loc[0, "Soft_Domain_Status"], "not_evaluated")

    def test_output_exposes_target_site_resolution_audit(self) -> None:
        output = initialize_output(pd.DataFrame({"ID": [1]}), self.manifest)
        self.assertIn("Target_O4_Source", output)
        self.assertIn("Target_O4_Candidate_Count", output)
        self.assertIn("Target_O4_Free_Candidate_Count", output)
        self.assertIn("Target_O4_Blocked_Candidate_Count", output)

    def test_wrong_label_mapping_is_rejected(self) -> None:
        manifest = dict(self.manifest)
        manifest["label_mapping"] = {"0": "Beta", "1": "Alpha"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "0=Alpha, 1=Beta"):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()
