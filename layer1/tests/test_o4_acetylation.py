"""验证目标 O4 乙酰化的连接、价态与立体化学保持规则。"""

import unittest

from rdkit import Chem

from layer1.o4_acetylation import O4AcetylationError, generate_o4_acetate


class O4AcetylationTests(unittest.TestCase):
    def test_adds_only_one_acetyl_group(self) -> None:
        smiles = "OC1CCCCC1"
        mol = Chem.MolFromSmiles(smiles)
        oxygen_index = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
        carbon_index = mol.GetAtomWithIdx(oxygen_index).GetNeighbors()[0].GetIdx()
        result = generate_o4_acetate(smiles, oxygen_index, carbon_index)
        self.assertEqual(result.generated_atom_count - result.original_atom_count, 3)
        self.assertEqual(result.generated_bond_count - result.original_bond_count, 3)
        self.assertTrue(result.check_acetyl_connectivity)
        self.assertTrue(result.check_only_o4_acetylation)

    def test_rejects_already_substituted_oxygen(self) -> None:
        smiles = "COC1CCCCC1"
        mol = Chem.MolFromSmiles(smiles)
        oxygen_index = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
        ring_carbon = next(atom.GetIdx() for atom in mol.GetAtomWithIdx(oxygen_index).GetNeighbors() if atom.IsInRing())
        with self.assertRaisesRegex(O4AcetylationError, "free hydroxyl"):
            generate_o4_acetate(smiles, oxygen_index, ring_carbon)

    def test_preserves_defined_stereochemistry(self) -> None:
        smiles = "O[C@H]1CCCCO1"
        mol = Chem.MolFromSmiles(smiles)
        oxygen_index = next(
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == 8 and not atom.IsInRing()
        )
        carbon_index = mol.GetAtomWithIdx(oxygen_index).GetNeighbors()[0].GetIdx()
        result = generate_o4_acetate(smiles, oxygen_index, carbon_index)
        self.assertTrue(result.check_stereochemistry_preserved)
        self.assertEqual(result.original_chiral_atom_count, result.generated_original_chiral_atom_count)


if __name__ == "__main__":
    unittest.main()
