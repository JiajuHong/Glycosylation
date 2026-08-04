"""多糖环受体在自由推理时的目标 O4 定位回归测试。"""

from __future__ import annotations

import unittest

import pandas as pd
from rdkit import Chem

from layer1.predict_hard_feasibility import (
    annotate_row,
    canonicalize,
    enumerate_target_site_candidates,
    extract_acceptor_target_site,
)


ACCEPTOR = (
    "C[C@H]1[C@H]([C@H]([C@@H]([C@@H](O1)O[C@H]2[C@@H]([C@H](O[C@@H]"
    "([C@@H]2NC(C)=O)OCCCN(CC3=CC=CC=C3)C(OCC4=CC=CC=C4)=O)C)OCC5=CC=CC=C5)"
    "NC(OCC(Cl)(Cl)Cl)=O)OC(C)=O)O"
)

DONOR = (
    "O=C(ON[C@@H]1[C@@H](N=[N+]=[N-])[C@H](OCC2=CC=CC=C2)[C@H]"
    "(OC1OC(C(F)(F)F)=NC3=CC=CC=C3)C(OCC4=CC=CC=C4)=O)C(Cl)(Cl)Cl"
)


def candidate_states(smiles: str) -> list[tuple[dict[str, object], str, int | None]]:
    """返回结构 O4 候选及其游离/封闭状态，仅用于构造测试分子。"""
    molecule = Chem.MolFromSmiles(smiles)
    results = []
    for candidate in enumerate_target_site_candidates(molecule):
        o4 = int(candidate["O4"])
        c4 = int(candidate["C4"])
        atom = molecule.GetAtomWithIdx(o4)
        external = [neighbor.GetIdx() for neighbor in atom.GetNeighbors() if neighbor.GetIdx() != c4]
        state = "free" if not external and atom.GetTotalNumHs() >= 1 else "blocked"
        results.append((candidate, state, None if not external else int(external[0])))
    return results


class InferenceTargetResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        canonical, _, mapping = canonicalize(ACCEPTOR)
        if canonical is None or mapping is None:
            raise AssertionError("测试受体无法规范化")
        cls.canonical = canonical
        cls.original_to_canonical = mapping

    def test_unique_free_site_is_selected_among_multiple_structural_sites(self) -> None:
        result = extract_acceptor_target_site(self.canonical)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["selection_source"], "auto_unique_free")
        self.assertEqual(result["state"], "free")
        self.assertEqual(result["o4_index"], self.original_to_canonical[62])
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["free_candidate_count"], 1)
        self.assertEqual(result["blocked_candidate_count"], 1)

    def test_explicit_blocked_site_remains_available_to_layer1(self) -> None:
        blocked = [entry for entry in candidate_states(self.canonical) if entry[1] == "blocked"]
        self.assertEqual(len(blocked), 1)
        blocked_o4 = int(blocked[0][0]["O4"])

        result = extract_acceptor_target_site(self.canonical, blocked_o4)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["selection_source"], "supplied_target")
        self.assertEqual(result["state"], "blocked")
        self.assertEqual(result["o4_index"], blocked_o4)

    def test_multiple_free_sites_require_an_explicit_target(self) -> None:
        molecule = Chem.MolFromSmiles(self.canonical)
        blocked = [entry for entry in candidate_states(self.canonical) if entry[1] == "blocked"]
        self.assertEqual(len(blocked), 1)
        blocked_o4 = int(blocked[0][0]["O4"])
        external = blocked[0][2]
        self.assertIsNotNone(external)
        editable = Chem.RWMol(molecule)
        editable.RemoveBond(blocked_o4, int(external))
        modified = editable.GetMol()
        Chem.SanitizeMol(modified)

        result = extract_acceptor_target_site(Chem.MolToSmiles(modified, isomericSmiles=True))

        self.assertEqual(result["status"], "multiple_free_target_sites")
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["free_candidate_count"], 2)
        self.assertEqual(result["blocked_candidate_count"], 0)

    def test_multiple_blocked_sites_are_not_reported_as_multiple_free_sites(self) -> None:
        molecule = Chem.MolFromSmiles(self.canonical)
        free = [entry for entry in candidate_states(self.canonical) if entry[1] == "free"]
        self.assertEqual(len(free), 1)
        free_o4 = int(free[0][0]["O4"])
        editable = Chem.RWMol(molecule)
        methyl = editable.AddAtom(Chem.Atom(6))
        editable.AddBond(free_o4, methyl, Chem.BondType.SINGLE)
        modified = editable.GetMol()
        Chem.SanitizeMol(modified)

        result = extract_acceptor_target_site(Chem.MolToSmiles(modified, isomericSmiles=True))

        self.assertEqual(result["status"], "no_free_target_site")
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["free_candidate_count"], 0)
        self.assertEqual(result["blocked_candidate_count"], 2)

        _, audit = annotate_row(
            pd.Series(
                {
                    "Donor_Canonical_SMILES": DONOR,
                    "Acceptor_Canonical_SMILES": Chem.MolToSmiles(
                        modified, isomericSmiles=True
                    ),
                }
            ),
            0,
        )
        self.assertEqual(audit["Inference_Status"], "structurally_infeasible")
        self.assertEqual(audit["Target_O4_State"], "all_blocked")


if __name__ == "__main__":
    unittest.main()
