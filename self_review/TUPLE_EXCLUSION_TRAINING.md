# Training PKD under full-tuple configuration exclusion

## Context

The paper's Section V-D generalization experiment originally excluded MCS
values within an otherwise-fully-observed configuration slice. This has
been generalized to randomly exclude entire **(channel_model_id, MCS,
N_t, N_r, N_ss, BW)** tuples from the whole configuration grid, so PKD's
generalization is tested across the joint configuration space rather than
along the MCS axis alone.

All the code for this is already implemented and pushed to `origin/main`
(commit `52fa6f2`, "Add full-tuple PHY configuration exclusion for
generalization testing"). **Nothing needs to be written — this doc is a
run-book for producing the trained checkpoints the evaluation/plotting
code expects.**

Start by pulling the latest code:

```bash
git pull origin main
pip install -r requirements.txt
```

## What you're producing

One trained PKD checkpoint per exclusion percentage, at:

```
pkd/trained_models/exclude_config_<PCT>/pkd_model.pt
```

e.g. `pkd/trained_models/exclude_config_30/pkd_model.pt` for the 30% run.
This naming convention (`exclude_config_<PCT>`, symmetric with the
pre-existing MCS-only `exclude_mcs_<PCT>` directories) is hardcoded in several
places that read these checkpoints back — don't rename it:
`pkd/example/evaluate_exclusion.py`,
`pkd/baselines/evaluate_baseline_comparison.py`,
`pkd/baselines/evaluate_per_waterfall.py`.

## Per-run steps

For each exclusion percentage `PCT` in **0, 10, 20, 30, 40, 50, 60, 70,
80, 90**:

```bash
cd /path/to/PHY_knowledge_distillation
PYTHONPATH=. python -m pkd.example train --exclusion-config-pct ${PCT}%
```

This will:
1. Load all 4000 `.mat` files under `data/` (takes a few minutes — this
   step is I/O-bound, not GPU-bound).
2. Generate a random `PCT`% full-tuple exclusion set (seed=42, so it's
   reproducible) and save it under
   `pkd/example/exclusions/exclusions_config_random_<PCT>pct_seed42_<timestamp>.{yaml,json}`.
   A `30%` file is already committed from earlier testing — running this
   again will regenerate an equivalent file with a new timestamp, which is
   expected and harmless (evaluation code always picks the most recent
   file for a given percentage via glob).
3. Filter it out of train/val (test set is never filtered).
4. Train for up to 10 epochs (early stopping patience=3), then save the
   best checkpoint to **`pkd_model.pt` at the project root**.

**Critical manual step — do this after every run, before starting the
next percentage:**

```bash
mkdir -p "pkd/trained_models/exclude_config_${PCT}"
mv pkd_model.pt "pkd/trained_models/exclude_config_${PCT}/pkd_model.pt"
```

`train_pkd` always writes to the same `pkd_model.pt` path at the project
root — it does **not** know about exclusion percentages or write into
`pkd/trained_models/` itself. If you skip this move before the next run,
you will silently overwrite the previous percentage's checkpoint and lose
it. This is how the existing MCS-only checkpoints
(`pkd/trained_models/exclude_mcs_<PCT>/`) were produced too — same manual
pattern, just a different directory name for this new tuple-exclusion
experiment.

Recommended: script the full sweep so the move can't be forgotten, e.g.:

```bash
for PCT in 0 10 20 30 40 50 60 70 80 90; do
    PYTHONPATH=. python -m pkd.example train --exclusion-config-pct ${PCT}%
    mkdir -p "pkd/trained_models/exclude_config_${PCT}"
    mv pkd_model.pt "pkd/trained_models/exclude_config_${PCT}/pkd_model.pt"
done
```

## Priority order (if you can't run all 10 right away)

1. **30%** first — needed for Fig. 9/10/11/12 (single-configuration and
   aggregate-across-excluded-tuples figures) and one panel of Fig. 14.
2. **60%** next — needed for Fig. 14's second waterfall panel.
3. Remaining percentages (0, 10, 20, 40, 50, 70, 80, 90) — needed for
   Fig. 13's full exclusion-percentage sweep.

## Sanity checks after each run

- Confirm the checkpoint file exists and is non-trivial in size:
  `ls -la pkd/trained_models/exclude_config_${PCT}/pkd_model.pt`
- Confirm a matching exclusion manifest exists:
  `ls pkd/example/exclusions/exclusions_config_random_${PCT}pct_seed42_*.json`
- Quick smoke test that the checkpoint loads and evaluates without error
  (uses only 3 excluded tuples, so it's fast):

```python
from pkd.example.data_loader import load_real_data
from pkd.example.evaluate_exclusion import (
    load_config_exclusion_manifest, get_excluded_tuples,
    evaluate_model_on_excluded_tuples,
)

train_seq, train_cfg, val_seq, val_cfg, test_seq, test_cfg = load_real_data('data')
manifest = load_config_exclusion_manifest(30)  # match the PCT you just trained
excluded_tuples = get_excluded_tuples(manifest)[:3]
metrics = evaluate_model_on_excluded_tuples(
    'pkd/trained_models/exclude_config_30/pkd_model.pt',
    test_seq, test_cfg, excluded_tuples, device='cuda'
)
print(metrics)
```

Expect `overall_ks_median` and `overall_acf_median` to be small (roughly
0.05-0.15 range based on prior MCS-only-exclusion runs) — if you see
values near 0.5+ (comparable to the untrained nearest-tuple baseline),
something is wrong (e.g. wrong checkpoint path, or the exclusion filter
didn't actually apply during training).

## After training is done

Report back which percentages are done. From there, the driver scripts
that turn checkpoints into the actual paper figures still need to be
wired up/run (`pkd/baselines/evaluate_baseline_comparison.py`'s
`run_tuple_baseline_comparison` + `generate_tuple_comparison_plots` for
Fig. 13, `pkd/baselines/evaluate_per_waterfall.py`'s
`run_per_waterfall_comparison_tuple` + `generate_per_waterfall_plot_tuple`
for Fig. 14, and `pkd/example/evaluate_exclusion.py`'s
`evaluate_model_on_excluded_tuples` for Fig. 9/10/11/12) — that part
doesn't need a GPU and can be done back on the original machine, or here,
once checkpoints exist.
