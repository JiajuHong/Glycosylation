# Chiral-GINE implementation notes

## Scope

This project implements a chirality-aware alternative to the existing GINE
encoder. It does **not** replace or modify `models/gine_encoder.py`; ordinary
GINE remains the default and is selected with `--encoder-type gine`.

The Chiral-GINE option is selected with:

```bash
python -m layer3.train_gine --encoder-type chiral_gine --model-type crossattn
```

Development uses the existing fixed `split_pair_group` train/validation/test
split. Fast checks may use one seed; stability checks use seeds 0/1/2 on the
same fixed split. Five-fold cross-validation is outside the current
development scope.

For the validated stable training configuration, use:

```bash
python -m layer3.train_gine \
  --encoder-type chiral_gine \
  --model-type crossattn \
  --seed 0
```

Selecting `chiral_gine` automatically enables strict deterministic execution,
PERM_CAT dropout 0.1, a 0.5x PERM_CAT learning rate, gradient clipping at 2.0,
a 10-epoch warmup, cosine decay, and patience 40. Individual CLI options can
still override these defaults for explicit ablations. Selecting ordinary
`gine` retains its original training defaults and the fixed data split is not
changed.

## Algorithm provenance

The tetrahedral PERM_CAT aggregation is an independent implementation based on:

- Pattanaik et al., *Message Passing Networks for Molecules with Tetrahedral
  Chirality* (2020), arXiv:2012.00094.
- Reference source: <https://github.com/PattanaikL/chiral_gnn>

Only the tetrahedral aggregation idea and parity convention are adapted. The
reference repository's D-MPNN, training framework, feature dimensions,
pooling, classifier and datasets are not copied into this project.

## Module boundaries

- `layer3/chiral_graph.py`: explicit-H expansion, heavy-atom mask, parity and
  tetrahedral neighbour/edge indices.
- `models/tetra_permutation.py`: the 12 even tetrahedral permutations and
  PERM_CAT aggregation.
- `models/chiral_gine_conv.py`: ordinary GINE sum aggregation for nonchiral
  atoms and PERM_CAT replacement at labelled tetrahedral centres.
- `models/chiral_gine_encoder.py`: the same projection, layer count, residual,
  normalization and dropout structure as the existing GINE encoder.
- `models/glyco_gine_models.py`: selects either encoder without changing the
  global/local/cross-attention model variants.
- `layer3/tests/test_chiral_gine.py`: graph-index, permutation and model integration
  tests, plus optimizer/scheduler/profile stability tests.

## Graph invariants

For each molecule, Chiral-GINE graph data stores:

```text
atom_idx
heavy_atom_mask
parity_atoms
tetra_center_index
tetra_neighbor_index
tetra_edge_index
```

Explicit hydrogens are added only to labelled tetrahedral centres and are
appended after the original atoms. Consequently, RFU and OH indices continue
to address the same heavy atoms. Global pooling applies `heavy_atom_mask`,
while explicit hydrogens still participate in message passing.

The Chiral-GINE cache has a separate feature version and path:

```bash
python -m layer3.build_rdkit_graph_cache --graph-type chiral_gine
# data/processed/rdkit_chiral_graph_cache.pt
```

The existing ordinary-GINE cache remains:

```text
data/processed/rdkit_graph_cache.pt
```

## Execution environment

Model training, forward/backward integration tests, and GPU smoke tests are run
only on `server1` in the conda environment `one`, from `/home/jjhong/gly`.
The local workspace is used for code editing and static inspection only; do not
run model training locally.

Server verification command:

```bash
ssh server1
cd /home/jjhong/gly
conda activate one
python -m unittest discover -s layer3/tests -t . -v
```

## Verification

Run the focused test suite on `server1` in conda `one`:

```bash
cd /home/jjhong/gly
python -m unittest discover -s layer3/tests -t . -v
```

The tests require even-permutation invariance, odd-permutation sensitivity,
correct PyG batch offsets, heavy-atom index preservation, availability of the
ordinary GINE encoder, three-output local interaction invariants, and
forward/backward support for all registered model variants.
