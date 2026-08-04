"""验证第一层结构去重、正负配对和模型输入唯一性。"""

import unittest
from pathlib import Path

import pandas as pd


# 测试文件位于 layer1/tests，项目根目录需要向上两级定位。
ROOT = Path(__file__).resolve().parents[2]


class HardFeasibilityUniqueDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = [
            ROOT / "data/processed/hard_feasibility_unique_pairs_2632.csv",
            ROOT / "data/processed/hard_feasibility_structural_groups_1316.csv",
            ROOT / "data/processed/hard_feasibility_site_model_ready_2632.csv",
        ]
        if missing := [path.name for path in required if not path.exists()]:
            raise FileNotFoundError("冻结的第一层数据产物不存在: " + ", ".join(missing))
        cls.unique = pd.read_csv(
            ROOT / "data/processed/hard_feasibility_unique_pairs_2632.csv",
            encoding="utf-8-sig",
        )
        cls.groups = pd.read_csv(
            ROOT / "data/processed/hard_feasibility_structural_groups_1316.csv",
            encoding="utf-8-sig",
        )
        cls.model_ready = pd.read_csv(
            ROOT / "data/processed/hard_feasibility_site_model_ready_2632.csv",
            encoding="utf-8-sig",
        )

    def test_expected_counts_and_unique_inputs(self) -> None:
        self.assertEqual(len(self.groups), 1316)
        self.assertEqual(len(self.unique), 2632)
        self.assertEqual(len(self.model_ready), 2632)
        self.assertEqual(self.unique["Input_Structure_SHA256"].nunique(), 2632)

    def test_each_structural_group_has_one_positive_and_one_negative(self) -> None:
        grouped = self.unique.groupby("Structural_Group_ID")
        self.assertTrue((grouped.size() == 2).all())
        self.assertTrue((grouped["hard_feasibility"].nunique() == 2).all())
        self.assertTrue((grouped["Variant"].nunique() == 2).all())

    def test_no_condition_columns_are_present(self) -> None:
        forbidden = {
            "Solvent",
            "Catalyst",
            "Temp_C",
            "Time_min",
            "has_solvent",
            "has_catalyst",
            "has_temp",
            "has_time",
        }
        self.assertFalse(forbidden.intersection(self.unique.columns))

    def test_model_ready_table_excludes_direct_construction_metadata(self) -> None:
        expected = {
            "Sample_ID",
            "Pair_ID",
            "Structural_Group_ID",
            "hard_feasibility",
            "Donor_Canonical_SMILES",
            "Acceptor_Canonical_SMILES",
            "Donor_Type",
            "Acceptor_C4_Index",
            "Acceptor_O4_Index",
        }
        self.assertTrue(expected.issubset(self.model_ready.columns))
        forbidden = {
            "Variant",
            "Original_Acceptor_Canonical_SMILES",
            "Structural_Group_SHA256",
            "Input_Structure_SHA256",
            "All_Parent_IDs",
            "All_Parent_Reaction_IDs",
        }
        self.assertFalse(forbidden.intersection(self.model_ready.columns))

    def test_all_parent_reactions_are_preserved_in_group_metadata(self) -> None:
        self.assertEqual(int(self.groups["Duplicate_Count"].sum()), 1561)
        observed_counts = self.groups["All_Parent_IDs"].astype(str).str.split(";").str.len()
        self.assertTrue(observed_counts.eq(self.groups["Duplicate_Count"]).all())


if __name__ == "__main__":
    unittest.main()
