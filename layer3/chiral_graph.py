"""第三层图构造：为 Chiral-GINE 建立保留四面体手性的分子图。

The graph keeps the original RDKit heavy-atom indices unchanged and appends
explicit hydrogens only where a labelled tetrahedral centre needs one.  The
extra tensors are deliberately model-agnostic so the graph construction can
be tested independently from the neural network.

The tetrahedral parity convention and four-neighbour ordering follow the
四面体邻居排序遵循 Pattanaik 等人的 PERM_CAT 定义：
https://github.com/PattanaikL/chiral_gnn
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from rdkit import Chem
from torch_geometric.data import Data


CHIRAL_GRAPH_FEATURE_VERSION = "rdkit_chiral_v1_permcat"


class ChiralData(Data):
    """PyG data with correct batching offsets for tetrahedral index tensors."""

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in {"tetra_neighbor_index", "tetra_edge_index"}:
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key in {"tetra_center_index", "tetra_neighbor_index"}:
            return self.num_nodes
        if key == "tetra_edge_index":
            return self.num_edges
        return super().__inc__(key, value, *args, **kwargs)


def chiral_parity(atom: Chem.Atom) -> int:
    """Return +1 for RDKit CW, -1 for CCW, and 0 otherwise."""

    tag = atom.GetChiralTag()
    if tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
        return 1
    if tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
        return -1
    return 0


def add_tetrahedral_hydrogens(mol: Chem.Mol) -> tuple[Chem.Mol, int]:
    """Append H atoms only to labelled tetrahedral centres.

    RDKit's ``AddHs(..., onlyOnAtoms=...)`` preserves existing atom indices and
    appends new hydrogen atoms.  Returning the original atom count makes that
    invariant explicit and allows heavy-atom-only pooling downstream.
    """

    mol = Chem.Mol(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    original_atom_count = mol.GetNumAtoms()
    centre_ids = [atom.GetIdx() for atom in mol.GetAtoms() if chiral_parity(atom)]
    if centre_ids:
        mol = Chem.AddHs(mol, onlyOnAtoms=centre_ids)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

    for centre_idx in centre_ids:
        degree = len(mol.GetAtomWithIdx(centre_idx).GetNeighbors())
        if degree != 4:
            raise ValueError(
                f"Tetrahedral atom {centre_idx} has {degree} neighbours after explicit-H expansion; expected 4"
            )
    return mol, original_atom_count


def mol_to_chiral_pyg_graph(
    mol: Chem.Mol,
    smiles: str,
    atom_feature_fn: Callable[[Chem.Atom], list[float]],
    bond_feature_fn: Callable[[Chem.Bond], list[float]],
) -> ChiralData:
    """Build a Chiral-GINE graph with PERM_CAT indexing metadata."""

    chiral_mol, original_atom_count = add_tetrahedral_hydrogens(mol)
    x = torch.tensor(
        [atom_feature_fn(atom) for atom in chiral_mol.GetAtoms()], dtype=torch.float32
    )

    edge_pairs: list[list[int]] = []
    edge_attrs: list[list[float]] = []
    directed_edge_lookup: dict[tuple[int, int], int] = {}
    for bond in chiral_mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        features = bond_feature_fn(bond)
        for source, target in ((begin, end), (end, begin)):
            directed_edge_lookup[(source, target)] = len(edge_pairs)
            edge_pairs.append([source, target])
            edge_attrs.append(features)

    bond_feature_dim = len(bond_feature_fn(next(iter(chiral_mol.GetBonds())))) if edge_pairs else 0
    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, bond_feature_dim), dtype=torch.float32)

    parity_atoms = torch.tensor(
        [chiral_parity(atom) for atom in chiral_mol.GetAtoms()], dtype=torch.long
    )
    tetra_centres: list[int] = []
    tetra_neighbours: list[list[int]] = []
    tetra_edges: list[list[int]] = []
    for atom in chiral_mol.GetAtoms():
        centre = atom.GetIdx()
        if not parity_atoms[centre]:
            continue
        neighbours = [neighbour.GetIdx() for neighbour in atom.GetNeighbors()]
        if len(neighbours) != 4:
            raise ValueError(f"Tetrahedral atom {centre} does not have exactly four neighbours")
        tetra_centres.append(centre)
        tetra_neighbours.append(neighbours)
        tetra_edges.append([directed_edge_lookup[(neighbour, centre)] for neighbour in neighbours])

    data = ChiralData(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = smiles
    data.atom_idx = torch.arange(chiral_mol.GetNumAtoms(), dtype=torch.long)
    data.heavy_atom_mask = torch.arange(chiral_mol.GetNumAtoms()) < original_atom_count
    data.parity_atoms = parity_atoms
    data.tetra_center_index = torch.tensor(tetra_centres, dtype=torch.long)
    data.tetra_neighbor_index = torch.tensor(tetra_neighbours, dtype=torch.long).reshape(-1, 4)
    data.tetra_edge_index = torch.tensor(tetra_edges, dtype=torch.long).reshape(-1, 4)
    data.num_heavy_atoms = original_atom_count
    data.graph_feature_version = CHIRAL_GRAPH_FEATURE_VERSION
    return data
