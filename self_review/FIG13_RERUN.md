# Running Fig. 13 (full-tuple-exclusion sweep) on another machine

## Context

`self_review/gen_fig_13.py` runs the 0-90% full-tuple-exclusion sweep
(PKD vs. nearest-tuple EESM-log-AR baseline) that produces Fig. 13. It
completed successfully once already on this machine (~9 hours on a
4-core CPU) and correctly saved both plots, but crashed on the final
step — dumping raw per-tuple results to JSON — because a nested dict
used tuple keys, which `json.dump` can't serialize. That bug is fixed,
and the script now also checkpoints results to disk after every
exclusion percentage, so a crash only costs whatever percentage was in
progress, not the whole run.

**Already done, don't redo:**
- `figures/sparse_tuple_ks_comparison.png`
- `figures/sparse_tuple_acf_rmse_comparison.png`

Both are correct and saved (from the completed run before the JSON
crash). If the agent's run regenerates them, that's fine (same data,
just overwritten) — but they're not the reason to run this again.

**What's actually missing:**
`self_review/tuple_exclusion_fig13_results.json` — currently on disk
but truncated/invalid (the crash happened mid-write). This is the only
thing this rerun needs to recover.

## What to bring over

None of this is committed yet. Before sending an agent, either commit
these changes and push, or copy the working tree over as-is. The agent
needs all three:

- `self_review/gen_fig_13.py` — the driver; new tuple-key JSON fix
  (`_json_safe`) and per-percentage checkpointing (`on_percentage_done`)
- `pkd/baselines/evaluate_baseline_comparison.py` — added the
  `on_percentage_done` callback parameter to `run_tuple_baseline_comparison`
- `pkd/example/plotting.py` — an earlier, unrelated bugfix to
  `generate_figure3_ccdf_error_by_tuple` (Fig. 10a). Not needed for
  Fig. 13 itself, but bring it anyway since it's an uncommitted fix on
  the same branch and you don't want to lose it.

Also needed (same as the original training run-book,
`self_review/TUPLE_EXCLUSION_TRAINING.md`):
- `pkd/trained_models/exclude_config_{0,10,20,...,90}/pkd_model.pt` —
  all 10 checkpoints
- `pkd/example/exclusions/exclusions_config_random_*pct_seed42_*.json` —
  the exclusion manifests generated during training
- `data/` — the full `.mat` dataset

## Command

```bash
cd /path/to/PHY_knowledge_distillation
PYTHONPATH=. python self_review/gen_fig_13.py
```

That's it — no arguments, no flags. Takes several hours depending on
machine speed (see note below).

## What it produces

- `figures/sparse_tuple_ks_comparison.png` (regenerated, should be
  identical to the existing one)
- `figures/sparse_tuple_acf_rmse_comparison.png` (same)
- `self_review/tuple_exclusion_fig13_results.json` — **this is the
  deliverable**. It now gets overwritten after every exclusion
  percentage (0, 10, 20, ..., 90) completes, not just once at the end.

## GPU note

`run_tuple_baseline_comparison` auto-selects `cuda` if available
(`device=None` default), so no code change is needed for GPU. But only
the **PKD side** (neural model inference, via `evaluate_model_on_excluded_tuples`)
benefits — the **nearest-tuple baseline** (`evaluate_nearest_tuple_baseline`
in `pkd/baselines/evaluate_baselines.py`) is pure NumPy/SciPy classical
AR calibration and fitting, with no GPU path. That baseline side is
likely the actual bottleneck (its per-tuple, per-SNR loop is what scales
with exclusion percentage — tuple counts grow from 40 at 10% up to 360
at 90%), so a faster CPU / more cores matters more than the GPU itself
for total wall-clock time. Any machine noticeably faster than a 4-core
laptop should still finish well under the ~9 hours seen here.

## Sanity checks while/after it runs

- Progress: `tail -f stage2_rerun.log` (or whatever you redirect
  stdout to) should show one `"N% FULL-TUPLE EXCLUSION"` header per
  percentage, each ending with a `"Checkpointed results through N% to
  ..."` line.
- After it finishes: confirm the JSON is valid and has all 9
  percentages populated:

```python
import json
d = json.load(open('self_review/tuple_exclusion_fig13_results.json'))
print(sorted(int(k) for k in d['pkd']))         # should be [0? , 10, 20, ..., 90] -- 0% has no PKD entry (nothing excluded)
print(sorted(int(k) for k in d['nearest_tuple']))
```

- Expected KS-statistic trend (from the completed run on this machine,
  for reference/comparison): PKD stays flat around 0.065-0.072 through
  60% exclusion, then rises to ~0.08 at 70% and ~0.10 at 80% as training
  data gets very sparse — still roughly 5-8x better than the
  nearest-tuple baseline (which itself degrades from ~0.49 to ~0.65
  over the same range) at every percentage.

## After it's done

Send back `self_review/tuple_exclusion_fig13_results.json` (and the two
PNGs if you want to double-check they match). That completes the
full-tuple-exclusion figure set (Figs. 9, 10, 11, 12, 13, 14).
