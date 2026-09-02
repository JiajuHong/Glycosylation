# Formal computational experiment V1

The completed results, interpretation boundaries, validation status, and recommended
prospective wet-lab validation are recorded in
`support/experiments/FORMAL_COMPUTATIONAL_V1_WORK_SUMMARY.md`.

## Scope

This experiment is the final reproducible computational comparison for the
stereoselectivity layer. It deliberately excludes development-only searches over
dropout, learning rate, warmup length, patience, and alternative literature
similarity formulas.

The primary split is `split_pair_group`. A temporal `split_year` evaluation is
included for the frozen main architecture because the fixed pair-group test set
was visible during model development.

## Formal matrix

All deep configurations use seeds 0, 1, and 2.

| Config | Split | Encoder | Model | Local output |
|---|---|---|---|---|
| `pair_chiral_global` | pair group | Chiral-GINE | global | — |
| `pair_chiral_local` | pair group | Chiral-GINE | local | — |
| `pair_chiral_crossattn` | pair group | Chiral-GINE | legacy cross-attention | — |
| `pair_chiral_tri_l1` | pair group | Chiral-GINE | three-output cross-attention | local states |
| `pair_chiral_tri_l2` | pair group | Chiral-GINE | three-output cross-attention | explicit interaction |
| `pair_chiral_tri_l3` | pair group | Chiral-GINE | three-output cross-attention | all three outputs |
| `pair_gine_tri_l3_fair` | pair group | ordinary GINE | three-output cross-attention | all three outputs |
| `year_chiral_tri_l3` | year | Chiral-GINE | three-output cross-attention | all three outputs |

The ordinary-GINE comparison explicitly uses the same deterministic execution,
gradient clipping, warmup, cosine schedule, and patience as the Chiral-GINE main
model. PERM-CAT-specific parameters do not exist in ordinary GINE.

### Current baseline scope

Traditional baselines are logistic regression, random forest, and XGBoost using
`structure_condition` and `condition_only`: 3 models x 2 feature sets x 3 splits
= 18 retained runs. The splits are `split_pair_group`, `split_year`, and
`split_random_stratified`. Both feature sets include solvent, catalyst/activator,
temperature and time, with missing-value handling. Molecular fingerprints are
ordinary non-chiral Morgan radius 2, 2048 bits; `Donor_Type` is excluded.

All current stereoselectivity models include conditions. The separate layer-1
structural gate does not use conditions and is not part of this ablation matrix.

### Historical experiments and archives

The original baseline matrix also included `structure_only` (27 runs in total).
Its 9 structure-only runs have been removed from active result tables, not
invalidated. The original tables and source are preserved in
`artifacts/archives/without_conditions_20260902/pre_removal.tar.gz`.

A subsequent three-seed no-condition L3 experiment was completed separately from
the 24-run deep matrix. Its runner, metrics, predictions, logs and checkpoints are
archived under `artifacts/archives/without_conditions_20260902/` using their
original relative paths. Current code does not support that training variant;
reproducing it requires the archived source in a separate directory. Do not
restore archived files over the active project or rerun into archived results.

Retained baseline metrics were filtered from existing runs, not retrained.
Requiring conditions is a task-scope decision, not evidence that conditions
improve classification. See the work summary and `CLOSEOUT_2026_09_02.md` for
the completed ablation results and limitations.

## Interpretation boundaries

- `global` versus `local` tests the contribution of explicitly pooled RFU and O4
  regions.
- `local` versus cross-attention models tests the contribution of donor–acceptor
  communication.
- L1/L2/L3 tests the outputs retained from the three-output interaction block.
  L1 is not a no-interaction model: its local states have already been updated by
  bidirectional attention.
- The temporal split is a robustness evaluation, not a replacement for prospective
  wet-lab validation.
- Validation-set thresholds are used for classification; softmax scores are not
  reported as calibrated reaction-success probabilities.

## Execution

These are reproduction commands, not unfinished work. The baseline command now
produces 18 runs, not the historical 27. It retrains models and writes into the
specified directory; use a new `--output-dir` for a new comparison to preserve
the completed evidence. The deep runner resumes the original 24-run matrix.

```bash
python -m baseline.run_baseline_ml \
  --output-dir results/formal_computational_v1/baselines

python -m support.experiments.run_formal_computational_v1 --gpus 0 1

python -m support.experiments.finalize_formal_computational_v1 --device cuda:0
```

The runner is resumable. A run is skipped only when its one-row metric artifact and
referenced checkpoint both exist and match the expected configuration and seed.

## Output layout

```text
artifacts/checkpoints/formal_computational_v1/  checkpoints and histories
artifacts/metrics/formal_computational_v1/      run metrics, manifest, live state
logs/formal_computational_v1/                   one log per deep run
results/formal_computational_v1/baselines/      baseline metrics and predictions
results/formal_computational_v1/predictions/    per-seed and ensemble predictions
results/formal_computational_v1/                summaries, comparisons and error audit
```
