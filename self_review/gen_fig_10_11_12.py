"""Stage 1/3: Figs. 9/10/11/12 -- evaluate exclude_config_30 on all its
excluded tuples. Run independently of stage 2/3 so a crash there doesn't
cost this stage's results.

    PYTHONPATH=. python self_review/gen_fig_10_11_12.py
"""
import json

from pkd.example.data_loader import load_real_data
from pkd.example.evaluate_exclusion import (
    load_config_exclusion_manifest,
    get_excluded_tuples,
    evaluate_model_on_excluded_tuples,
)

REPRESENTATIVE_TUPLE = {
    'channel_model_id': 2, 'N_t': 3, 'N_r': 2, 'BW': 40.0, 'N_ss': 2, 'MCS': 7,
}


def main():
    print("Loading full dataset...")
    _, _, _, _, test_sequences, test_configs = load_real_data('data')
    print(f"Loaded {len(test_sequences)} test sequences")

    manifest_30 = load_config_exclusion_manifest(30)
    excluded_tuples_30 = get_excluded_tuples(manifest_30)
    print(f"{len(excluded_tuples_30)} excluded tuples at 30%")
    assert REPRESENTATIVE_TUPLE in excluded_tuples_30, \
        "Representative tuple is not in the 30% excluded set -- check manifest"

    metrics_30 = evaluate_model_on_excluded_tuples(
        'pkd/trained_models/exclude_config_30/pkd_model.pt',
        test_sequences, test_configs, excluded_tuples_30, device='cpu',
    )
    with open('self_review/tuple_exclusion_30pct_metrics.json', 'w') as f:
        json.dump(metrics_30, f, indent=2, default=str)
    print("Saved self_review/tuple_exclusion_30pct_metrics.json")
    print("-> figures/test_quantile_error_tuple_exclusion.png (Fig. 10)")
    print("-> figures/test_ccdf_error_tuple_exclusion.png (Fig. 12)")
    print("DONE stage 1/3")


if __name__ == '__main__':
    main()
