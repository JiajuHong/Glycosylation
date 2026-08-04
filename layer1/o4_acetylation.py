"""第一层结构规则：确定性生成目标 O4 乙酰化结构并验证化学正确性。"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


class O4AcetylationError(ValueError):
    """Raised when the requested O4 acetylation is structurally invalid."""


@dataclass(frozen=True)
class O4AcetylationResult:
    original_canonical_smiles: str
    generated_canonical_smiles: str
    original_c4_index: int
    original_o4_index: int
    generated_c4_index: int
    generated_o4_index: int
    generated_o4_external_index: int
    generated_acetyl_carbonyl_o_index: int
    generated_acetyl_methyl_c_index: int
    original_to_generated_indices: tuple[int, ...]
    original_atom_count: int
    generated_atom_count: int
    original_bond_count: int
    generated_bond_count: int
    original_chiral_atom_count: int
    generated_original_chiral_atom_count: int
    check_original_parse: bool
    check_target_indices_valid: bool
    check_o4_is_free_hydroxyl: bool
    check_acetyl_connectivity: bool
    check_valence_legal: bool
    check_stereochemistry_preserved: bool
    check_only_o4_acetylation: bool


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise O4AcetylationError("SMILES cannot be parsed")
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _atom_signature(atom: Chem.Atom) -> tuple[object, ...]:
    return (
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        atom.GetIsotope(),
        atom.GetIsAromatic(),
        atom.GetChiralTag(),
    )


def _bond_signature(bond: Chem.Bond) -> tuple[object, ...]:
    return (
        min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
        max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
        bond.GetBondType(),
        bond.GetIsAromatic(),
        bond.GetStereo(),
    )


def _chiral_atom_count(mol: Chem.Mol, limit: int | None = None) -> int:
    atoms = mol.GetAtoms() if limit is None else (mol.GetAtomWithIdx(i) for i in range(limit))
    return sum(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in atoms)


def _remove_added_acetyl_atoms(mol: Chem.Mol, added_indices: tuple[int, int, int]) -> Chem.Mol:
    editable = Chem.RWMol(mol)
    for atom_index in sorted(added_indices, reverse=True):
        editable.RemoveAtom(atom_index)
    restored = editable.GetMol()
    Chem.SanitizeMol(restored)
    Chem.AssignStereochemistry(restored, force=True, cleanIt=True)
    return restored


def generate_o4_acetate(
    original_acceptor_smiles: str,
    target_o4_index: int,
    target_c4_index: int,
) -> O4AcetylationResult:
    """Add O-C(=O)CH3 to a specified free O4 while preserving the parent graph."""

    original = Chem.MolFromSmiles(str(original_acceptor_smiles))
    if original is None:
        raise O4AcetylationError("check_1_failed: original acceptor cannot be parsed")
    Chem.AssignStereochemistry(original, force=True, cleanIt=True)
    original_canonical = Chem.MolToSmiles(original, isomericSmiles=True)
    original_atom_count = original.GetNumAtoms()
    original_bond_count = original.GetNumBonds()

    if not (0 <= int(target_o4_index) < original_atom_count):
        raise O4AcetylationError("check_2_failed: target O4 index is out of range")
    if not (0 <= int(target_c4_index) < original_atom_count):
        raise O4AcetylationError("check_2_failed: target C4 index is out of range")
    target_o4_index = int(target_o4_index)
    target_c4_index = int(target_c4_index)
    o4 = original.GetAtomWithIdx(target_o4_index)
    c4 = original.GetAtomWithIdx(target_c4_index)
    if o4.GetAtomicNum() != 8 or c4.GetAtomicNum() != 6:
        raise O4AcetylationError("check_2_failed: target atom elements are not O4/C4")
    if original.GetBondBetweenAtoms(target_o4_index, target_c4_index) is None:
        raise O4AcetylationError("check_2_failed: target O4 is not bonded to target C4")
    if o4.GetTotalNumHs() < 1 or o4.GetDegree() != 1:
        raise O4AcetylationError("check_3_failed: target O4 is not a free hydroxyl")

    original_atom_signatures = [_atom_signature(atom) for atom in original.GetAtoms()]
    original_bond_signatures = sorted(_bond_signature(bond) for bond in original.GetBonds())

    editable = Chem.RWMol(original)
    carbonyl_c_index = editable.AddAtom(Chem.Atom(6))
    carbonyl_o_index = editable.AddAtom(Chem.Atom(8))
    methyl_c_index = editable.AddAtom(Chem.Atom(6))
    editable.AddBond(target_o4_index, carbonyl_c_index, Chem.BondType.SINGLE)
    editable.AddBond(carbonyl_c_index, carbonyl_o_index, Chem.BondType.DOUBLE)
    editable.AddBond(carbonyl_c_index, methyl_c_index, Chem.BondType.SINGLE)
    generated = editable.GetMol()
    try:
        Chem.SanitizeMol(generated)
    except Exception as exc:
        raise O4AcetylationError(f"check_5_failed: generated valence is illegal: {exc}") from exc
    Chem.AssignStereochemistry(generated, force=True, cleanIt=True)

    acetyl_connectivity = (
        generated.GetBondBetweenAtoms(target_o4_index, carbonyl_c_index).GetBondType()
        == Chem.BondType.SINGLE
        and generated.GetBondBetweenAtoms(carbonyl_c_index, carbonyl_o_index).GetBondType()
        == Chem.BondType.DOUBLE
        and generated.GetBondBetweenAtoms(carbonyl_c_index, methyl_c_index).GetBondType()
        == Chem.BondType.SINGLE
        and generated.GetAtomWithIdx(target_o4_index).GetTotalNumHs() == 0
        and generated.GetAtomWithIdx(carbonyl_c_index).GetDegree() == 3
        and generated.GetAtomWithIdx(carbonyl_o_index).GetDegree() == 1
        and generated.GetAtomWithIdx(methyl_c_index).GetDegree() == 1
    )
    if not acetyl_connectivity:
        raise O4AcetylationError("check_4_failed: O4-acetyl connectivity is incorrect")

    generated_original_atom_signatures = [
        _atom_signature(generated.GetAtomWithIdx(i)) for i in range(original_atom_count)
    ]
    generated_original_bond_signatures = sorted(
        _bond_signature(bond)
        for bond in generated.GetBonds()
        if bond.GetBeginAtomIdx() < original_atom_count and bond.GetEndAtomIdx() < original_atom_count
    )
    only_o4_acetylation = (
        generated.GetNumAtoms() == original_atom_count + 3
        and generated.GetNumBonds() == original_bond_count + 3
        and generated_original_atom_signatures == original_atom_signatures
        and generated_original_bond_signatures == original_bond_signatures
    )

    restored = _remove_added_acetyl_atoms(
        generated,
        (carbonyl_c_index, carbonyl_o_index, methyl_c_index),
    )
    restored_canonical = Chem.MolToSmiles(restored, isomericSmiles=True)
    stereochemistry_preserved = (
        restored_canonical == original_canonical
        and _chiral_atom_count(generated, original_atom_count) == _chiral_atom_count(original)
        and generated_original_atom_signatures == original_atom_signatures
    )
    if not stereochemistry_preserved:
        raise O4AcetylationError("check_6_failed: parent stereochemistry changed")
    if not only_o4_acetylation:
        raise O4AcetylationError("check_7_failed: changes extend beyond O4 acetylation")

    generated_canonical = Chem.MolToSmiles(generated, isomericSmiles=True)
    reparsed = Chem.MolFromSmiles(generated_canonical)
    if reparsed is None:
        raise O4AcetylationError("check_5_failed: generated canonical SMILES cannot be reparsed")
    atom_mapping = reparsed.GetSubstructMatch(generated, useChirality=True)
    if len(atom_mapping) != generated.GetNumAtoms():
        raise O4AcetylationError("generated atom-index remapping failed")
    generated_o4_index = int(atom_mapping[target_o4_index])
    generated_c4_index = int(atom_mapping[target_c4_index])
    generated_o4_external_index = int(atom_mapping[carbonyl_c_index])
    generated_acetyl_carbonyl_o_index = int(atom_mapping[carbonyl_o_index])
    generated_acetyl_methyl_c_index = int(atom_mapping[methyl_c_index])

    return O4AcetylationResult(
        original_canonical_smiles=original_canonical,
        generated_canonical_smiles=generated_canonical,
        original_c4_index=target_c4_index,
        original_o4_index=target_o4_index,
        generated_c4_index=generated_c4_index,
        generated_o4_index=generated_o4_index,
        generated_o4_external_index=generated_o4_external_index,
        generated_acetyl_carbonyl_o_index=generated_acetyl_carbonyl_o_index,
        generated_acetyl_methyl_c_index=generated_acetyl_methyl_c_index,
        original_to_generated_indices=tuple(int(atom_mapping[i]) for i in range(original_atom_count)),
        original_atom_count=original_atom_count,
        generated_atom_count=generated.GetNumAtoms(),
        original_bond_count=original_bond_count,
        generated_bond_count=generated.GetNumBonds(),
        original_chiral_atom_count=_chiral_atom_count(original),
        generated_original_chiral_atom_count=_chiral_atom_count(generated, original_atom_count),
        check_original_parse=True,
        check_target_indices_valid=True,
        check_o4_is_free_hydroxyl=True,
        check_acetyl_connectivity=True,
        check_valence_legal=True,
        check_stereochemistry_preserved=True,
        check_only_o4_acetylation=True,
    )
