"""验证第一层位点角色和正负样本标注的一致性。"""

import json
import unittest
from pathlib import Path

import pandas as pd


# 测试文件位于 layer1/tests，项目根目录需要向上两级定位。
ROOT = Path(__file__).resolve().parents[2]


class HardFeasibilitySiteAnnotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = ROOT / "data/processed/hard_feasibility_site_model_ready_2632.csv"
        if not path.exists():
            raise FileNotFoundError(f"冻结的第一层模型输入不存在: {path}")
        cls.model_ready = pd.read_csv(path, encoding="utf-8-sig")

    def test_counts_and_balanced_labels(self) -> None:
        self.assertEqual(len(self.model_ready), 2632)
        self.assertEqual(
            self.model_ready.groupby("hard_feasibility").size().to_dict(),
            {0: 1316, 1: 1316},
        )

    def test_role_json_contains_all_roles(self) -> None:
        expected = {
            "O4",
            "C4",
            "C3",
            "C5",
            "C2",
            "O5",
            "C1",
            "N2_SUB_ENTRY",
            "N2_SUBGRAPH",
            "PG_ENTRY",
            "O4_EXTERNAL",
        }
        for value in self.model_ready["Acceptor_Target_Role_Indices"]:
            self.assertEqual(set(json.loads(value)), expected)

    def test_donor_rfu_and_acceptor_local_site_are_present(self) -> None:
        required = {
            "Donor_RFU_Atom_Indices",
            "Donor_RFU_Role_Indices",
            "Donor_C1_Index",
            "Acceptor_Target_Local_Atom_Indices",
            "Acceptor_Target_Role_Indices",
        }
        self.assertTrue(required.issubset(self.model_ready.columns))

    def test_external_atom_convention(self) -> None:
        positive = self.model_ready.loc[self.model_ready["hard_feasibility"] == 1]
        negative = self.model_ready.loc[self.model_ready["hard_feasibility"] == 0]
        self.assertTrue((positive["Acceptor_O4_External_Atom_Index"] == -1).all())
        self.assertTrue((negative["Acceptor_O4_External_Atom_Index"] >= 0).all())

    def test_model_ready_excludes_direct_status_fields(self) -> None:
        forbidden = {
            "Variant",
            "Original_Acceptor_Canonical_SMILES",
            "Acceptor_Target_O4_State",
            "Acceptor_O4_External_Atom_Role",
            "Acceptor_Site_Annotation_Status",
            "Acceptor_Site_Annotation_Source",
            "Solvent",
            "Catalyst",
            "Temp_C",
            "Time_min",
        }
        self.assertFalse(forbidden.intersection(self.model_ready.columns))


if __name__ == "__main__":
    unittest.main()
