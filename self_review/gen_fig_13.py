"""Stage 2/3: Fig. 13 -- PKD vs. nearest-tuple baseline, 0-90% full-tuple
exclusion sweep. Independent of stage 1/3 and 3/3.

    PYTHONPATH=. python self_review/gen_fig_13.py
"""
import json

from pkd.baselines.evaluate_baseline_comparison import (
    run_tuple_baseline_comparison,
    generate_tuple_comparison_plots,
)


def _json_safe(obj):
    """Recursively convert non-str dict keys (e.g. the tuple keys used by
    per_tuple_ks/per_tuple_acf in evaluate_nearest_tuple_baseline's output)
    to strings so json.dump doesn't crash after a multi-hour sweep."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


CHECKPOINT_PATH = 'self_review/tuple_exclusion_fig13_results.json'


def _checkpoint(pct, results_so_far):
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(_json_safe(results_so_far), f, indent=2, default=str)
    print(f"  Checkpointed results through {pct}% to {CHECKPOINT_PATH}")


def main():
    results_13 = run_tuple_baseline_comparison(
        exclusion_percentages=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        on_percentage_done=_checkpoint,
    )
    generate_tuple_comparison_plots(results_13)
    _checkpoint('final', results_13)
    print(f"Saved {CHECKPOINT_PATH}")
    print("DONE stage 2/3")


if __name__ == '__main__':
    main()
