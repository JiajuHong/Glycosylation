# RFU–4-OH three-output interaction

## Scope and compatibility

The finalized local interaction is a new model type:

```text
crossattn_tri
```

The previous `crossattn` implementation remains available for comparison and
its code path is not modified. `crossattn_tri` changes only the local
interaction and local fusion inputs. Chiral-GINE, full-molecule pooling,
condition inputs, dataset split, loss, and prediction head design remain
unchanged for an attributable comparison.

## Three outputs

`RFUOHCrossAttentionTri` returns a named `LocalInteractionOutput` containing:

```text
z_d_local          updated donor RFU state, pooled with C1 as query
z_a_local          updated acceptor 4-OH state, pooled with O4 as query
z_int              explicit relation built from pure bidirectional messages
h_d_updated        node-level donor states after residual attention and FFN
h_a_updated        node-level acceptor states after residual attention and FFN
message_d_from_a   pure acceptor-to-donor attention message
message_a_from_d   pure donor-to-acceptor attention message
```

The two directions use independent multi-head attention parameters. Updated
states use residual attention, LayerNorm, an independent FFN, and a second
LayerNorm. `z_int` does not recompress `z_d_local` and `z_a_local`; it uses:

```text
[m_d, m_a, abs(m_d - m_a), m_d * m_a] -> interaction MLP
```

where `m_d` and `m_a` are masked pools of the pure directional messages.

## Local ablations

The CLI option `--local-output-mode` defines the local inputs:

```text
l1  z_d_local + z_a_local
l2  z_int
l3  z_d_local + z_a_local + z_int  (main model)
```

Checkpoints include the mode in their filename so L1/L2/L3 cannot overwrite
one another.

## Execution environment

Do not run model training locally. All model execution is performed on:

```text
host: server1
project: /home/jjhong/gly
conda environment: one
```

Main development command:

```bash
cd /home/jjhong/gly
conda activate one
python train_gine.py \
  --encoder-type chiral_gine \
  --model-type crossattn_tri \
  --local-output-mode l3 \
  --seed 0
```

Selecting `chiral_gine` automatically activates the validated stable training
defaults. Five-fold cross-validation is not run during the current development
stage.

## Validation sequence

1. Run unit and integration tests in conda `one`.
2. Run a short GPU smoke test for L3.
3. Run the fixed `split_pair_group` seed-0 development experiment.
4. If seed 0 is competitive, run seeds 0/1/2 and report mean, standard
   deviation, and worst-seed performance.
5. Run L1/L2/L3 ablations before adding MolFormer.
