from __future__ import annotations

import unittest

import pandas as pd
from rdkit import DataStructs

from layer2.retrieve_literature_evidence import (
    EVIDENCE_COLUMNS,
    Fingerprints,
    prepare_row,
    score_prepared_queries,
)


class LiteratureEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.positives = pd.read_csv(
            "data/processed/soft_compatibility_positive_1561.csv",
            encoding="utf-8-sig",
        )

    def test_exact_success_reaction_has_pair_history_and_condition_evidence(self) -> None:
        prepared, audit = prepare_row(self.positives.iloc[0])
        self.assertEqual(audit["Evidence_Retrieval_Status"], "ok")
        scored, neighbors, metadata = score_prepared_queries(
            pd.DataFrame([prepared]), self.positives, top_k_neighbors=3
        )
        self.assertEqual(scored.loc[0, "Pair_History"], "known_pair")
        self.assertEqual(
            scored.loc[0, "Condition_Transfer_Evidence"],
            "exact_pair_exact_condition",
        )
        self.assertTrue(bool(scored.loc[0, "Exact_Pair_Condition_Precedent"]))
        self.assertEqual(float(scored.loc[0, "Nearest_Joint_Similarity"]), 1.0)
        self.assertTrue(neighbors[0][0]["exact_pair"])
        self.assertIn("Tanimoto", metadata["fingerprint"])

    def test_pair_history_does_not_become_condition_transfer_evidence(self) -> None:
        prepared, _ = prepare_row(self.positives.iloc[0])
        prepared["Solvent_Component_IDs"] = "999999"
        prepared["Catalyst_Component_IDs"] = "999998"
        scored, neighbors, _ = score_prepared_queries(
            pd.DataFrame([prepared]), self.positives, top_k_neighbors=2
        )
        self.assertEqual(scored.loc[0, "Pair_History"], "known_pair")
        self.assertEqual(scored.loc[0, "Condition_Transfer_Evidence"], "template_only")
        self.assertEqual(int(scored.loc[0, "Condition_Precedent_Count"]), 0)
        self.assertFalse(neighbors[0])
        self.assertTrue(pd.isna(scored.loc[0, "Nearest_Joint_Similarity"]))
        forbidden = {
            "Soft_Compatibility_Score",
            "Soft_Domain_Status",
            "Soft_Support_Percentile",
        }
        self.assertFalse(forbidden.intersection(EVIDENCE_COLUMNS))

    def test_joint_similarity_is_non_compensating_minimum(self) -> None:
        prepared, _ = prepare_row(self.positives.iloc[0])
        scored, neighbors, _ = score_prepared_queries(
            pd.DataFrame([prepared]), self.positives, top_k_neighbors=5
        )
        nearest = neighbors[0][0]
        self.assertEqual(
            nearest["joint_similarity"],
            min(nearest["donor_tanimoto"], nearest["acceptor_tanimoto"]),
        )
        self.assertEqual(
            scored.loc[0, "Nearest_Joint_Similarity"], nearest["joint_similarity"]
        )

    def test_fingerprint_distinguishes_stereoisomers(self) -> None:
        fingerprints = Fingerprints()
        clockwise = fingerprints.get("F[C@H](Cl)Br")
        counterclockwise = fingerprints.get("F[C@@H](Cl)Br")
        self.assertLess(DataStructs.TanimotoSimilarity(clockwise, counterclockwise), 1.0)


if __name__ == "__main__":
    unittest.main()
