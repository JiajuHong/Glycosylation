"""条件模板库与湿实验筛选逻辑测试。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pipeline.build_condition_library import build_library
from pipeline.screen_conditions import (
    add_ranking_fields,
    prepare_tasks,
    select_diverse_recommendations,
)


DONOR = (
    "O=C(ON[C@@H]1[C@@H](N=[N+]=[N-])[C@H](OCC2=CC=CC=C2)[C@H]"
    "(OC1OC(C(F)(F)F)=NC3=CC=CC=C3)C(OCC4=CC=CC=C4)=O)C(Cl)(Cl)Cl"
)
ACCEPTOR = (
    "C[C@H]1[C@H]([C@H]([C@@H]([C@@H](O1)O[C@H]2[C@@H]([C@H](O[C@@H]"
    "([C@@H]2NC(C)=O)OCCCN(CC3=CC=CC=C3)C(OCC4=CC=CC=C4)=O)C)OCC5=CC=CC=C5)"
    "NC(OCC(Cl)(Cl)Cl)=O)OC(C)=O)O"
)


class ConditionLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = pd.read_csv("data/processed/soft_compatibility_positive_1561.csv")
        cls.library = build_library(cls.source)

    def test_library_contains_only_unique_complete_real_templates(self) -> None:
        self.assertEqual(len(self.library), 539)
        self.assertEqual(int(self.library["Template_Support_Count"].sum()), 1058)
        self.assertFalse(self.library["Template_ID"].duplicated().any())
        self.assertTrue(
            self.library[["has_solvent", "has_catalyst", "has_temp", "has_time"]]
            .eq(1)
            .all(axis=None)
        )
        self.assertEqual(
            self.library.groupby("Donor_Type").size().to_dict(),
            {
                "thioglycoside": 291,
                "trichloroacetimidate": 200,
                "trifluoroacetimidate": 48,
            },
        )
        self.assertFalse(
            {
                "Catalyst_Mechanism_Role",
                "Mechanism_Compatible",
                "Mechanism_Filter_Version",
            }.intersection(self.library.columns)
        )

    def test_task_uses_same_donor_type_and_excludes_exact_original_condition(self) -> None:
        original = self.library.loc[
            self.library["Donor_Type"].eq("trifluoroacetimidate")
        ].iloc[0]
        tasks = pd.DataFrame(
            [
                {
                    "Task_ID": "third_task",
                    "Donor_Canonical_SMILES": DONOR,
                    "Acceptor_Canonical_SMILES": ACCEPTOR,
                    "Target_Config": "Beta",
                    "Original_Solvent": original["Solvent"],
                    "Original_Catalyst": original["Catalyst"],
                    "Original_Temp_C": original["Temp_C"],
                    "Original_Time_min": original["Time_min"],
                }
            ]
        )

        candidates, audit = prepare_tasks(tasks, self.library)

        donor_type_count = int(
            self.library.loc[
                self.library["Donor_Type"].eq("trifluoroacetimidate")
            ].shape[0]
        )
        expected = donor_type_count - 1
        self.assertEqual(len(candidates), expected)
        self.assertTrue(candidates["Donor_Type"].eq("trifluoroacetimidate").all())
        self.assertFalse(candidates["Template_ID"].eq(original["Template_ID"]).any())
        self.assertEqual(
            audit["excluded_original_conditions"], 1
        )
        self.assertEqual(audit["excluded_by_condition_key"], 1)
        self.assertEqual(audit["excluded_by_source_reaction"], 0)

    def test_task_excludes_template_containing_source_reaction_id(self) -> None:
        original = self.library.loc[
            self.library["Donor_Type"].eq("trifluoroacetimidate")
        ].iloc[0]
        tasks = pd.DataFrame(
            [
                {
                    "Task_ID": "source_reaction_task",
                    "Source_Reaction_ID": original["Representative_Reaction_ID"],
                    "Donor_Canonical_SMILES": DONOR,
                    "Acceptor_Canonical_SMILES": ACCEPTOR,
                    "Target_Config": "Beta",
                }
            ]
        )

        candidates, audit = prepare_tasks(tasks, self.library)

        donor_type_count = int(
            self.library.loc[
                self.library["Donor_Type"].eq("trifluoroacetimidate")
            ].shape[0]
        )
        self.assertEqual(len(candidates), donor_type_count - 1)
        self.assertFalse(candidates["Template_ID"].eq(original["Template_ID"]).any())
        self.assertEqual(audit["excluded_original_conditions"], 1)
        self.assertEqual(audit["excluded_by_condition_key"], 0)
        self.assertEqual(audit["excluded_by_source_reaction"], 1)

    def test_third_class_is_normalized_to_exploratory_rescue(self) -> None:
        tasks = pd.DataFrame(
            [
                {
                    "Task_ID": "third_task",
                    "Task_Class": 3,
                    "Donor_Canonical_SMILES": DONOR,
                    "Acceptor_Canonical_SMILES": ACCEPTOR,
                }
            ]
        )

        candidates, audit = prepare_tasks(tasks, self.library)

        self.assertTrue(candidates["Task_Mode"].eq("exploratory_rescue").all())
        self.assertTrue(candidates["Target_Config"].eq("Any").all())
        self.assertEqual(audit["tasks"][0]["Task_Mode"], "exploratory_rescue")


class ConditionRankingTests(unittest.TestCase):
    def make_predictions(self) -> pd.DataFrame:
        rows = []
        configurations = [
            ("c1", "3", "3", "exact_pair_exact_condition", "known_pair", ("Beta", "Beta", "Beta"), 1.00, 0.90),
            ("c2", "4", "3", "same_donor_condition", "known_pair", ("Beta", "Beta", "Beta"), 0.88, 0.89),
            ("c3", "5", "4", "same_acceptor_condition", "novel_pair", ("Beta", "Alpha", "Beta"), 0.84, 0.78),
            ("c4", "6", "5", "analog_condition", "novel_pair", ("Beta", "Alpha", "Beta"), 0.80, 0.76),
            ("c5", "7", "6", "analog_condition", "known_pair", ("Beta", "Beta", "Beta"), 0.70, 0.82),
            ("c6", "8", "7", "same_donor_condition", "novel_pair", ("Alpha", "Alpha", "Beta"), 0.75, 0.20),
        ]
        for candidate, catalyst, solvent, evidence, pair_history, votes, joint, beta_score in configurations:
            rows.append(
                {
                    "Task_ID": "T1",
                    "Candidate_ID": candidate,
                    "Template_ID": candidate,
                    "Target_Config": "Beta",
                    "Final_Status": "predicted",
                    "Pair_History": pair_history,
                    "Condition_Transfer_Evidence": evidence,
                    "Exact_Pair_Condition_Count": int(evidence == "exact_pair_exact_condition"),
                    "Same_Donor_Condition_Count": int(evidence == "same_donor_condition"),
                    "Same_Acceptor_Condition_Count": int(evidence == "same_acceptor_condition"),
                    "Nearest_Joint_Similarity": joint,
                    "Stereoselectivity_Mean_Beta_Score": beta_score,
                    "Layer3_Seed0_Vote": votes[0],
                    "Layer3_Seed1_Vote": votes[1],
                    "Layer3_Seed2_Vote": votes[2],
                    "Catalyst_Component_IDs": catalyst,
                    "Solvent_Component_IDs": solvent,
                    "Temp_C": 0.0,
                    "Template_Support_Count": 5,
                }
            )
        return pd.DataFrame(rows)

    def test_tiers_depend_on_votes_not_evidence_labels(self) -> None:
        ranked = add_ranking_fields(self.make_predictions())
        tiers = dict(zip(ranked["Candidate_ID"], ranked["Recommendation_Tier"]))
        self.assertEqual(
            tiers,
            {
                "c1": "A",
                "c2": "A",
                "c3": "C_EXPLORATORY",
                "c4": "C_EXPLORATORY",
                "c5": "A",
                "c6": "D_NOT_TARGET",
            },
        )
        levels = dict(zip(ranked["Candidate_ID"], ranked["Recommendation_Level"]))
        self.assertEqual(
            levels,
            {
                "c1": "优先",
                "c2": "优先",
                "c3": "探索",
                "c4": "探索",
                "c5": "优先",
                "c6": "不推荐",
            },
        )

    def test_diversity_selection_prefers_distinct_catalysts_within_best_tier(self) -> None:
        ranked = add_ranking_fields(self.make_predictions())
        selected = select_diverse_recommendations(ranked, top_n=2)
        self.assertEqual(selected["Candidate_ID"].tolist(), ["c1", "c2"])
        self.assertEqual(selected["Recommendation_Rank"].tolist(), [1, 2])
        self.assertTrue(np.isfinite(selected["Target_Stereo_Score"]).all())

    def test_pair_history_is_not_a_ranking_key(self) -> None:
        predictions = self.make_predictions().iloc[[3, 4]].copy()
        predictions["Candidate_ID"] = ["known", "novel"]
        predictions["Template_ID"] = ["z_known", "a_novel"]
        predictions["Pair_History"] = ["known_pair", "novel_pair"]
        predictions["Condition_Transfer_Evidence"] = "analog_condition"
        predictions[[
            "Layer3_Seed0_Vote", "Layer3_Seed1_Vote", "Layer3_Seed2_Vote"
        ]] = "Beta"
        predictions["Nearest_Joint_Similarity"] = 0.8
        predictions["Stereoselectivity_Mean_Beta_Score"] = 0.9

        ranked = add_ranking_fields(predictions)

        self.assertEqual(ranked["Candidate_ID"].tolist(), ["novel", "known"])

    def test_same_donor_and_same_acceptor_have_equal_evidence_priority(self) -> None:
        predictions = self.make_predictions().iloc[[1, 2]].copy()
        predictions["Candidate_ID"] = ["same_donor", "same_acceptor"]
        predictions["Template_ID"] = ["same_donor", "same_acceptor"]
        predictions[[
            "Layer3_Seed0_Vote", "Layer3_Seed1_Vote", "Layer3_Seed2_Vote"
        ]] = "Beta"
        predictions["Nearest_Joint_Similarity"] = [0.70, 0.90]

        ranked = add_ranking_fields(predictions)

        self.assertEqual(
            ranked["Candidate_ID"].tolist(), ["same_acceptor", "same_donor"]
        )

    def test_any_target_uses_evidence_only_without_stereo_score(self) -> None:
        predictions = self.make_predictions()
        predictions["Target_Config"] = "Any"
        ranked = add_ranking_fields(predictions)
        self.assertTrue(ranked["Recommendation_Tier"].eq("EVIDENCE_ONLY").all())
        self.assertTrue(ranked["Target_Stereo_Score"].isna().all())
        selected = select_diverse_recommendations(ranked, top_n=2)
        self.assertEqual(len(selected), 2)
        self.assertTrue(
            selected["Recommendation_Reason"]
            .str.contains("stereochemistry not used", regex=False)
            .all()
        )

    def test_rescue_tasks_are_always_exploratory(self) -> None:
        predictions = self.make_predictions()
        predictions["Target_Config"] = "Any"
        predictions["Task_Mode"] = "exploratory_rescue"

        ranked = add_ranking_fields(predictions)

        self.assertTrue(
            ranked["Recommendation_Tier"].eq("C_RESCUE_EXPLORATORY").all()
        )
        self.assertTrue(ranked["Recommendation_Level"].eq("探索性救援").all())
        self.assertTrue(ranked["Target_Stereo_Score"].isna().all())
        selected = select_diverse_recommendations(ranked, top_n=2)
        self.assertEqual(len(selected), 2)
        self.assertTrue(
            selected["Recommendation_Reason"]
            .str.contains("exploratory rescue only", regex=False)
            .all()
        )


if __name__ == "__main__":
    unittest.main()
