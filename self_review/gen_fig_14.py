"""Stage 3/3: Fig. 14 -- PER-vs-SNR waterfall for the representative tuple
at 30% and 60% full-tuple exclusion. Independent of stage 1/3 and 2/3.

    PYTHONPATH=. python self_review/gen_fig_14.py
"""
import json

from pkd.baselines.evaluate_per_waterfall import (
    run_per_waterfall_comparison_tuple,
    generate_per_waterfall_plot_tuple,
)

REPRESENTATIVE_TUPLE = {
    'channel_model_id': 2, 'N_t': 3, 'N_r': 2, 'BW': 40.0, 'N_ss': 2, 'MCS': 7,
}


def main():
    for pct in (30, 60):
        print(f"\n--- {pct}% ---")
        results_14 = run_per_waterfall_comparison_tuple(
            target_tuple=REPRESENTATIVE_TUPLE,
            exclusion_pct=pct,
            num_snr_points=6,
        )
        generate_per_waterfall_plot_tuple(
            results_14, REPRESENTATIVE_TUPLE, pct,
            save_path=f'figures/per_waterfall_comparison_tuple_{pct}pct.png',
        )
        with open(f'self_review/tuple_exclusion_fig14_{pct}pct_results.json', 'w') as f:
            json.dump(results_14, f, indent=2, default=str)
    print("DONE stage 3/3")


if __name__ == '__main__':
    main()
