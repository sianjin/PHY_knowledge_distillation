"""Fig. 13-style three-method comparison: PKD vs. Nearest-MCS EESM-log-AR
vs. Piecewise-Linear-Interpolation EESM-log-AR, under the same MCS
exclusion sweep, on the same test set, with the same metrics.

Reuses:
  - pkd.example.evaluate_exclusion.{load_exclusion_manifest,
    get_excluded_mcs_for_slice, evaluate_model_on_slice} for PKD, so PKD's
    numbers are computed by the exact same code path as the paper's
    existing Fig. 13.
  - pkd.baselines.evaluate_baselines for both classical baselines.

See self_review/EESM_LOG_AR_SPARSE_MCS_BASELINES.md Sec. 6 for the
original figure proposal.
"""
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import torch

from pkd.example.data_loader import load_real_data
from pkd.example.evaluate_exclusion import (
    load_exclusion_manifest,
    get_excluded_mcs_for_slice,
    evaluate_model_on_slice,
    load_config_exclusion_manifest,
    get_excluded_tuples,
    evaluate_model_on_excluded_tuples,
)

from .evaluate_baselines import (
    _load_exclusion_yaml_for_pct,
    _get_excluded_training_mcs_for_slice,
    _slice_matches,
    calibrate_retained_mcs,
    evaluate_baseline_on_slice,
    AR_ORDER,
    _load_config_exclusion_yaml_for_pct,
    _tuple_dict,
    _axis_ranges,
    calibrate_all_retained_tuples,
    evaluate_nearest_tuple_baseline,
)
from pkd.example.exclusion_filter import apply_exclusion_filter

METHOD_STYLE = {
    'pkd': dict(color='#2E86AB', marker='o', label='PKD'),
    'interpolate': dict(color='#06A77D', marker='s', label='EESM-log-AR + Piecewise-Linear Interp.'),
    'nearest': dict(color='#D62839', marker='^', label='EESM-log-AR + Nearest-MCS'),
}


def run_baseline_comparison(
    data_dir: str = 'data',
    slice_spec: Optional[dict] = None,
    exclusion_percentages: Optional[List[int]] = None,
    ar_order: int = AR_ORDER,
    device: Optional[str] = None,
    max_files: Optional[int] = None,
) -> Dict[str, Dict[int, dict]]:
    """Run PKD + both baselines across the same exclusion sweep and return
    per-method, per-percentage metrics (overall_ks, seen_ks, unseen_ks,
    generalization_gap, overall_acf).
    """
    if exclusion_percentages is None:
        exclusion_percentages = [0, 30, 60, 70, 80, 90]
    if slice_spec is None:
        slice_spec = {
            'channel_model_id': 2,  # Model-B
            'N_t': 3,
            'N_r': 2,
            'BW': 40.0,
            'N_ss': 2,
        }
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading data (train/val/test split, seed=42, matching PKD)...")
    train_sequences, train_configs, _, _, test_sequences, test_configs = load_real_data(
        data_dir, train_ratio=0.7, val_ratio=0.1, max_files=max_files, random_seed=42
    )
    slice_mcs_in_test = sorted({cfg['MCS'] for cfg in test_configs if _slice_matches(cfg, slice_spec)})

    results = {'pkd': {}, 'nearest': {}, 'interpolate': {}}

    for pct in exclusion_percentages:
        print(f"\n{'=' * 70}\n{pct}% MCS EXCLUSION\n{'=' * 70}")

        # ---- PKD: same manifest/checkpoint/metric path as evaluate_exclusion.py ----
        manifest = load_exclusion_manifest(pct)
        excluded_mcs, training_mcs = get_excluded_mcs_for_slice(manifest, slice_spec)
        print(f"  Training MCS: {training_mcs}")
        print(f"  Excluded MCS: {excluded_mcs}")

        checkpoint_path = f'pkd/trained_models/exclude_mcs_{pct}/pkd_model.pt'
        if os.path.exists(checkpoint_path):
            pkd_metrics = evaluate_model_on_slice(
                checkpoint_path, test_sequences, test_configs,
                slice_spec, excluded_mcs, training_mcs, device
            )
            results['pkd'][pct] = pkd_metrics
        else:
            print(f"  Warning: PKD checkpoint not found: {checkpoint_path}")

        # ---- Baselines: same retained-MCS set, same test set, same metrics ----
        exclusion_config = _load_exclusion_yaml_for_pct(pct) if pct > 0 else None
        baseline_excluded_mcs, baseline_training_mcs = _get_excluded_training_mcs_for_slice(
            exclusion_config, slice_spec, slice_mcs_in_test
        )

        if exclusion_config is not None:
            filtered_train_seq, filtered_train_cfg, _ = apply_exclusion_filter(
                train_sequences, train_configs, exclusion_config, split_name='train'
            )
        else:
            filtered_train_seq, filtered_train_cfg = train_sequences, train_configs

        calibrated = calibrate_retained_mcs(
            filtered_train_seq, filtered_train_cfg, slice_spec, baseline_training_mcs, ar_order=ar_order
        )

        for method in ('nearest', 'interpolate'):
            metrics = evaluate_baseline_on_slice(
                calibrated, method, test_sequences, test_configs, slice_spec,
                baseline_excluded_mcs, baseline_training_mcs, ar_order=ar_order,
            )
            print(f"  [{method}] Overall KS: {metrics['overall_ks']:.4f}, "
                  f"Unseen KS: {metrics['unseen_ks']:.4f}, "
                  f"Overall ACF RMSE: {metrics['overall_acf']:.4f}")
            results[method][pct] = metrics

    return results


def generate_comparison_plots(
    results: Dict[str, Dict[int, dict]],
    ks_path: str = 'figures/sparse_mcs_ks_comparison.png',
    acf_rmse_path: str = 'figures/sparse_mcs_acf_rmse_comparison.png',
):
    """Generate the two-panel Fig. 13 comparison (median KS, median ACF
    RMSE vs. MCS exclusion percentage), one series per method, reusing the
    plot style from pkd.example.evaluate_exclusion.generate_plots.
    """
    os.makedirs(os.path.dirname(ks_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(acf_rmse_path) or '.', exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid')

    methods = [m for m in ('pkd', 'interpolate', 'nearest') if results.get(m)]

    # Panel (a): Median KS Statistic vs. Exclusion Percentage (unseen/excluded MCS)
    plt.figure(figsize=(8, 6))
    for method in methods:
        pcts = sorted(results[method].keys())
        unseen_ks = [results[method][p]['unseen_ks'] for p in pcts]
        style = METHOD_STYLE[method]
        plt.plot(pcts, unseen_ks, style['marker'] + '-', linewidth=2, markersize=8,
                  color=style['color'], label=style['label'])
    plt.xlabel('MCS Exclusion Percentage (%)', fontsize=12)
    plt.ylabel('Median KS Statistic (Excluded MCS)', fontsize=12)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ks_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {ks_path}")

    # Panel (b): Median ACF RMSE vs. Exclusion Percentage (unseen/excluded MCS)
    plt.figure(figsize=(8, 6))
    for method in methods:
        pcts = sorted(results[method].keys())
        acf_rmse = [results[method][p]['overall_acf'] for p in pcts]
        style = METHOD_STYLE[method]
        plt.plot(pcts, acf_rmse, style['marker'] + '-', linewidth=2, markersize=8,
                  color=style['color'], label=style['label'])
    plt.xlabel('MCS Exclusion Percentage (%)', fontsize=12)
    plt.ylabel('Median ACF RMSE', fontsize=12)
    plt.ylim(0.05, 0.065)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(acf_rmse_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {acf_rmse_path}")


TUPLE_METHOD_STYLE = {
    'pkd': dict(color='#2E86AB', marker='o', label='PKD'),
    'nearest_tuple': dict(color='#D62839', marker='^', label='EESM-log-AR + Nearest-Tuple'),
}


def run_tuple_baseline_comparison(
    data_dir: str = 'data',
    exclusion_percentages: Optional[List[int]] = None,
    ar_order: int = AR_ORDER,
    device: Optional[str] = None,
    max_files: Optional[int] = None,
) -> Dict[str, Dict[int, dict]]:
    """Full-tuple-exclusion counterpart to run_baseline_comparison
    (Fig. 13, generalized): PKD vs. nearest-tuple-transfer EESM-log-AR,
    swept over exclusion percentage of full (CH, MCS, N_t, N_r, N_ss, BW)
    tuples rather than MCS within one fixed slice. No piecewise-linear
    interpolation baseline (Section V-D design decision -- see
    pkd.baselines.transfer.nearest_tuple_transfer's docstring).

    Returns:
        {'pkd': {pct: metrics}, 'nearest_tuple': {pct: metrics}}
    """
    if exclusion_percentages is None:
        exclusion_percentages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading data (train/val/test split, seed=42, matching PKD)...")
    train_sequences, train_configs, _, _, test_sequences, test_configs = load_real_data(
        data_dir, train_ratio=0.7, val_ratio=0.1, max_files=max_files, random_seed=42
    )

    all_mcs = sorted({cfg['MCS'] for cfg in train_configs + test_configs})
    all_tuple_dicts = [_tuple_dict(cfg) for cfg in train_configs + test_configs]
    axis_ranges = _axis_ranges(all_tuple_dicts)

    results = {'pkd': {}, 'nearest_tuple': {}}

    for pct in exclusion_percentages:
        print(f"\n{'=' * 70}\n{pct}% FULL-TUPLE EXCLUSION\n{'=' * 70}")

        # ---- PKD: same manifest/checkpoint path convention as
        # evaluate_exclusion.py's tuple-exclusion functions ----
        manifest = load_config_exclusion_manifest(pct)
        excluded_tuples = get_excluded_tuples(manifest)
        print(f"  Excluded tuples: {len(excluded_tuples)}")

        checkpoint_path = f'pkd/trained_models/exclude_config_{pct}/pkd_model.pt'
        if os.path.exists(checkpoint_path) and excluded_tuples:
            pkd_metrics = evaluate_model_on_excluded_tuples(
                checkpoint_path, test_sequences, test_configs, excluded_tuples, device
            )
            results['pkd'][pct] = pkd_metrics
        else:
            print(f"  Warning: PKD checkpoint not found or no excluded tuples: {checkpoint_path}")

        # ---- Nearest-tuple baseline: same retained set, same test set,
        # same metrics ----
        exclusion_config = _load_config_exclusion_yaml_for_pct(pct) if pct > 0 else None

        if exclusion_config is not None:
            filtered_train_seq, filtered_train_cfg, _ = apply_exclusion_filter(
                train_sequences, train_configs, exclusion_config, split_name='train'
            )
        else:
            filtered_train_seq, filtered_train_cfg = train_sequences, train_configs

        if not excluded_tuples:
            continue

        calibrated = calibrate_all_retained_tuples(filtered_train_seq, filtered_train_cfg, ar_order=ar_order)

        metrics = evaluate_nearest_tuple_baseline(
            calibrated, test_sequences, test_configs, excluded_tuples,
            axis_ranges, all_mcs, ar_order=ar_order,
        )
        print(f"  [nearest_tuple] Median KS: {metrics['overall_ks_median']:.4f}, "
              f"Median ACF RMSE: {metrics['overall_acf_median']:.4f}")
        results['nearest_tuple'][pct] = metrics

    return results


def generate_tuple_comparison_plots(
    results: Dict[str, Dict[int, dict]],
    ks_path: str = 'figures/sparse_tuple_ks_comparison.png',
    acf_rmse_path: str = 'figures/sparse_tuple_acf_rmse_comparison.png',
):
    """Generate the Fig. 13 (generalized) two-panel comparison (median KS,
    median ACF RMSE vs. full-tuple exclusion percentage), one series per
    method (PKD, nearest-tuple), reusing generate_comparison_plots' style.

    PKD's per-pct metrics use the 'overall_ks_median'/'overall_acf_median'
    keys (evaluate_model_on_excluded_tuples' output); the nearest-tuple
    baseline's use the same key names (evaluate_nearest_tuple_baseline's
    output), so both plot from the same field names.
    """
    os.makedirs(os.path.dirname(ks_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(acf_rmse_path) or '.', exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid')

    methods = [m for m in ('pkd', 'nearest_tuple') if results.get(m)]

    # Panel (a): Median KS Statistic vs. Exclusion Percentage
    plt.figure(figsize=(8, 6))
    for method in methods:
        pcts = sorted(results[method].keys())
        ks_vals = [results[method][p]['overall_ks_median'] for p in pcts]
        style = TUPLE_METHOD_STYLE[method]
        plt.plot(pcts, ks_vals, style['marker'] + '-', linewidth=2, markersize=8,
                  color=style['color'], label=style['label'])
    plt.xlabel('Configuration-Tuple Exclusion Percentage (%)', fontsize=12)
    plt.ylabel('Median KS Statistic (Excluded Tuples)', fontsize=12)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ks_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {ks_path}")

    # Panel (b): Median ACF RMSE vs. Exclusion Percentage
    plt.figure(figsize=(8, 6))
    for method in methods:
        pcts = sorted(results[method].keys())
        acf_vals = [results[method][p]['overall_acf_median'] for p in pcts]
        style = TUPLE_METHOD_STYLE[method]
        plt.plot(pcts, acf_vals, style['marker'] + '-', linewidth=2, markersize=8,
                  color=style['color'], label=style['label'])
    plt.xlabel('Configuration-Tuple Exclusion Percentage (%)', fontsize=12)
    plt.ylabel('Median ACF RMSE', fontsize=12)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(acf_rmse_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {acf_rmse_path}")


if __name__ == '__main__':
    results = run_baseline_comparison()
    generate_comparison_plots(results)
