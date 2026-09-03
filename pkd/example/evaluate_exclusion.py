"""
Evaluate PKD model performance across different MCS exclusion percentages.

Computes 4 key metrics:
1. Overall Median KS statistic
2. Seen vs Unseen MCS generalization gap
3. Overall Median ACF RMSE
4. Overall PIT pass rate

Generates 4 plots in pkd/figures/:
- exclusion_overall_ks.png
- exclusion_generalization_gap.png
- exclusion_acf_rmse.png
- exclusion_pit_passrate.png
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import defaultdict

from pkd.model import PKDModel
from pkd.per_lut import AWGNPERLookup
from pkd.infer import PKDInference
from .data_loader import load_real_data
from .utils import filter_by_slice, infer_most_common_slice
from .plotting import (
    generate_figure1_per_mcs_metrics,
    generate_figure2_quantile_error_by_tuple,
    generate_figure3_ccdf_error_by_tuple,
    generate_figure12_temporal_metrics_by_tuple,
)


def load_exclusion_manifest(exclusion_pct):
    """Load exclusion manifest JSON for given percentage.

    Args:
        exclusion_pct: Exclusion percentage (e.g., 30 for 30%)

    Returns:
        dict: Manifest data, or None if not found
    """
    pattern = f'pkd/example/exclusions/exclusions_random_{exclusion_pct}pct_seed42_*.json'
    files = glob.glob(pattern)

    if not files:
        print(f"  Warning: No manifest found for {exclusion_pct}% exclusion")
        return None

    # Use most recent file if multiple exist
    manifest_path = sorted(files)[-1]

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    print(f"  Loaded manifest: {os.path.basename(manifest_path)}")
    return manifest


def get_excluded_mcs_for_slice(manifest, slice_spec):
    """Extract excluded MCS for a specific configuration slice.

    Args:
        manifest: Exclusion manifest dict
        slice_spec: Dict with slice keys (channel_model_id, N_t, N_r, BW, N_ss)

    Returns:
        tuple: (excluded_mcs list, training_mcs list), or ([], all_mcs) if not found
    """
    if manifest is None:
        # No exclusion - all MCS are training MCS
        return ([], list(range(10)))

    for held_out_slice in manifest['held_out_slices']:
        # Check if this slice matches
        if all(held_out_slice['slice'].get(k) == v for k, v in slice_spec.items()):
            return (held_out_slice['excluded_mcs'], held_out_slice['training_mcs'])

    print(f"  Warning: Slice not found in manifest")
    return ([], list(range(10)))


def evaluate_model_on_slice(checkpoint_path, test_sequences, test_configs,
                            slice_spec, excluded_mcs, training_mcs, device='cpu'):
    """Evaluate a single model and compute metrics.

    Args:
        checkpoint_path: Path to model checkpoint
        test_sequences: Test sequences
        test_configs: Test configurations
        slice_spec: Configuration slice specification
        excluded_mcs: List of MCS that were excluded during training
        training_mcs: List of MCS that were used in training
        device: Device for evaluation

    Returns:
        dict: Metrics (overall_ks, seen_ks, unseen_ks, overall_acf, overall_pit)
    """
    print(f"\nEvaluating: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        # Add backward compatibility
        if 'num_R' not in model_config:
            model_config['num_R'] = 8
    else:
        print("  Warning: No model_config in checkpoint, using defaults")
        model_config = {
            'num_channel_models': 5,
            'num_mcs': 10,
            'num_nss': 4,
            'num_R': 8,
            'ar_order': 10,
            'hidden_dim': 128,
            'kappa_max': 0.95,
            'innovation_type': 'sgn',
            'min_sigma': 0.1,
            'min_beta': 0.5,
            'max_beta': 4.0
        }

    # Create model
    model = PKDModel(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Filter test set by slice
    filtered_sequences, filtered_configs, kept_indices = filter_by_slice(
        test_sequences, test_configs, slice_spec
    )

    print(f"  Filtered to slice: {len(filtered_sequences)} sequences")

    # Create temporary figures directory (generate_figure1_per_mcs_metrics saves plots)
    os.makedirs('figures', exist_ok=True)

    # Generate per-(MCS, SNR) metrics using existing code
    # Note: generate_figure1_per_mcs_metrics expects the raw model, not PKDInference
    results = generate_figure1_per_mcs_metrics(
        model, filtered_sequences, filtered_configs, device=device
    )

    # Compute metrics
    # 1. Overall KS (pool all MCS, SNR)
    all_ks_values = []
    seen_ks_values = []
    unseen_ks_values = []

    for mcs, mcs_data in results.items():
        ks_values = mcs_data['ks_med']  # Array of median KS per SNR

        # Overall: collect all
        all_ks_values.extend(ks_values)

        # Seen vs Unseen split
        if mcs in training_mcs:
            seen_ks_values.extend(ks_values)
        elif mcs in excluded_mcs:
            unseen_ks_values.extend(ks_values)

    overall_ks = np.median(all_ks_values) if all_ks_values else np.nan
    seen_ks = np.median(seen_ks_values) if seen_ks_values else np.nan
    unseen_ks = np.median(unseen_ks_values) if unseen_ks_values else np.nan

    # 2. Overall ACF RMSE
    all_acf_values = []
    for mcs, mcs_data in results.items():
        all_acf_values.extend(mcs_data['acf_rmse_med'])
    overall_acf = np.median(all_acf_values) if all_acf_values else np.nan

    # 3. Overall PIT pass rate
    all_pit_rates = []
    for mcs, mcs_data in results.items():
        all_pit_rates.extend(mcs_data['pit_rate'])
    overall_pit = np.mean(all_pit_rates) * 100 if all_pit_rates else np.nan  # Convert to percentage

    print(f"  Overall KS: {overall_ks:.4f}")
    print(f"  Seen MCS KS: {seen_ks:.4f}, Unseen MCS KS: {unseen_ks:.4f}, Gap: {unseen_ks - seen_ks:.4f}")
    print(f"  Overall ACF RMSE: {overall_acf:.4f}")
    print(f"  Overall PIT pass rate: {overall_pit:.1f}%")

    return {
        'overall_ks': overall_ks,
        'seen_ks': seen_ks,
        'unseen_ks': unseen_ks,
        'generalization_gap': unseen_ks - seen_ks,
        'overall_acf': overall_acf,
        'overall_pit': overall_pit
    }


TUPLE_KEYS = ['channel_model_id', 'N_t', 'N_r', 'BW', 'N_ss', 'MCS']


def load_config_exclusion_manifest(exclusion_pct):
    """Load the full-tuple-exclusion manifest JSON (from
    save_config_exclusion_manifest) for a given percentage.

    This is the full-tuple-exclusion counterpart to
    load_exclusion_manifest: instead of {"held_out_slices": [...]}
    (slice + partial MCS list), the manifest here is a flat
    {"excluded_tuples": [...]} list, since every tuple-exclusion rule
    withholds one full (CH, N_t, N_r, BW, N_ss, MCS) tuple.

    Args:
        exclusion_pct: Exclusion percentage (e.g., 30 for 30%)

    Returns:
        dict: Manifest data, or None if not found
    """
    pattern = f'pkd/example/exclusions/exclusions_config_random_{exclusion_pct}pct_seed42_*.json'
    files = glob.glob(pattern)

    if not files:
        print(f"  Warning: No config-exclusion manifest found for {exclusion_pct}%")
        return None

    manifest_path = sorted(files)[-1]
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    print(f"  Loaded manifest: {os.path.basename(manifest_path)}")
    return manifest


def get_excluded_tuples(manifest):
    """Extract the list of excluded (CH, N_t, N_r, BW, N_ss, MCS) tuple
    dicts from a config-exclusion manifest.

    Returns:
        List[dict], empty list if manifest is None.
    """
    if manifest is None:
        return []
    return manifest['excluded_tuples']


def evaluate_model_on_excluded_tuples(checkpoint_path, test_sequences, test_configs,
                                       excluded_tuples, device='cpu'):
    """Evaluate a single tuple-exclusion-trained PKD model on the excluded
    tuples only, using the *_by_tuple figure functions (aggregated median
    + 10-90 percentile band across all excluded tuples), the full-tuple
    counterpart to evaluate_model_on_slice.

    Args:
        checkpoint_path: Path to model checkpoint (trained with
            exclusion_config_percentage, i.e. full-tuple exclusion)
        test_sequences: Full test sequences (all configs, not pre-filtered)
        test_configs: Full test configs
        excluded_tuples: List of {channel_model_id, N_t, N_r, BW, N_ss,
            MCS} dicts identifying which tuples to evaluate against
        device: Device for evaluation

    Returns:
        dict: {'overall_ks_median'/'p10'/'p90', 'overall_acf_median'/
        'p10'/'p90', 'overall_psd_median'/'p10'/'p90'}
    """
    print(f"\nEvaluating (tuple exclusion): {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        if 'num_R' not in model_config:
            model_config['num_R'] = 8
    else:
        print("  Warning: No model_config in checkpoint, using defaults")
        model_config = {
            'num_channel_models': 5,
            'num_mcs': 10,
            'num_nss': 4,
            'num_R': 8,
            'ar_order': 10,
            'hidden_dim': 128,
            'kappa_max': 0.95,
            'innovation_type': 'sgn',
            'min_sigma': 0.1,
            'min_beta': 0.5,
            'max_beta': 4.0
        }

    model = PKDModel(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    excluded_keys = {tuple(sorted(t.items())) for t in excluded_tuples}
    filtered_sequences, filtered_configs = [], []
    for seq, cfg in zip(test_sequences, test_configs):
        key = tuple(sorted({k: cfg[k] for k in TUPLE_KEYS}.items()))
        if key in excluded_keys:
            filtered_sequences.append(seq)
            filtered_configs.append(cfg)

    print(f"  Filtered to {len(excluded_tuples)} excluded tuples: {len(filtered_sequences)} sequences")

    os.makedirs('figures', exist_ok=True)

    quantile_results = generate_figure2_quantile_error_by_tuple(
        model, filtered_sequences, filtered_configs, device=device
    )
    ccdf_results = generate_figure3_ccdf_error_by_tuple(
        model, filtered_sequences, filtered_configs, device=device
    )
    temporal_results = generate_figure12_temporal_metrics_by_tuple(
        model, filtered_sequences, filtered_configs, device=device
    )

    metrics = {
        'quantile_error_median': quantile_results['median'],
        'quantile_error_p10': quantile_results['p10'],
        'quantile_error_p90': quantile_results['p90'],
        'ccdf_error_median': ccdf_results['median'],
        'ccdf_error_p10': ccdf_results['p10'],
        'ccdf_error_p90': ccdf_results['p90'],
        'overall_ks_median': temporal_results['ks_stat']['median'],
        'overall_ks_p10': temporal_results['ks_stat']['p10'],
        'overall_ks_p90': temporal_results['ks_stat']['p90'],
        'overall_acf_median': temporal_results['acf_rmse']['median'],
        'overall_acf_p10': temporal_results['acf_rmse']['p10'],
        'overall_acf_p90': temporal_results['acf_rmse']['p90'],
        'overall_psd_median': temporal_results['psd_rmse']['median'],
        'overall_psd_p10': temporal_results['psd_rmse']['p10'],
        'overall_psd_p90': temporal_results['psd_rmse']['p90'],
    }

    print(f"  Median KS statistic: {metrics['overall_ks_median']:.4f} "
          f"[{metrics['overall_ks_p10']:.4f}, {metrics['overall_ks_p90']:.4f}]")
    print(f"  Median ACF RMSE: {metrics['overall_acf_median']:.4f} "
          f"[{metrics['overall_acf_p10']:.4f}, {metrics['overall_acf_p90']:.4f}]")
    print(f"  Median normalized PSD RMSE: {metrics['overall_psd_median']:.4f} "
          f"[{metrics['overall_psd_p10']:.4f}, {metrics['overall_psd_p90']:.4f}]")

    return metrics


def generate_plots(results_by_pct, slice_spec):
    """Generate the 4 plots."""

    # Extract data
    percentages = sorted(results_by_pct.keys())
    overall_ks = [results_by_pct[p]['overall_ks'] for p in percentages]
    seen_ks = [results_by_pct[p]['seen_ks'] for p in percentages]
    unseen_ks = [results_by_pct[p]['unseen_ks'] for p in percentages]
    overall_acf = [results_by_pct[p]['overall_acf'] for p in percentages]
    overall_pit = [results_by_pct[p]['overall_pit'] for p in percentages]

    # Create output directory
    os.makedirs('figures', exist_ok=True)

    # Plot style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Plot 1: Overall KS vs Exclusion %
    print("\n  Generating Plot 1: exclusion_overall_ks.png")
    plt.figure(figsize=(8, 6))
    plt.plot(percentages, overall_ks, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    plt.xlabel('Exclusion Percentage (%)', fontsize=12)
    plt.ylabel('Median KS Statistic', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/exclusion_overall_ks.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Generalization Gap (Seen vs Unseen MCS)
    print("  Generating Plot 2: exclusion_generalization_gap.png")
    plt.figure(figsize=(8, 6))
    plt.plot(percentages, seen_ks, 'o-', linewidth=2, markersize=8,
             color='#06A77D', label='Seen MCS')
    plt.plot(percentages, unseen_ks, 's-', linewidth=2, markersize=8,
             color='#D62839', label='Unseen MCS')
    plt.fill_between(percentages, seen_ks, unseen_ks, alpha=0.2, color='gray')
    plt.xlabel('Exclusion Percentage (%)', fontsize=12)
    plt.ylabel('Median KS Statistic', fontsize=12)
    plt.legend(fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/exclusion_generalization_gap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: ACF RMSE vs Exclusion %
    print("  Generating Plot 3: exclusion_acf_rmse.png")
    plt.figure(figsize=(8, 6))
    plt.plot(percentages, overall_acf, 'o-', linewidth=2, markersize=8, color='#A23B72')
    plt.xlabel('Exclusion Percentage (%)', fontsize=12)
    plt.ylabel('Median ACF RMSE', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/exclusion_acf_rmse.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 4: PIT Pass Rate vs Exclusion %
    print("  Generating Plot 4: exclusion_pit_passrate.png")
    plt.figure(figsize=(8, 6))
    plt.plot(percentages, overall_pit, 'o-', linewidth=2, markersize=8, color='#F18F01')
    plt.axhline(y=95, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Ideal (95%)')
    plt.xlabel('Exclusion Percentage (%)', fontsize=12)
    plt.ylabel('PIT Pass Rate (%)', fontsize=12)
    plt.legend(fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/exclusion_pit_passrate.png', dpi=300, bbox_inches='tight')
    plt.close()


def run_exclusion_analysis(data_dir='data', slice_spec=None, exclusion_percentages=None, device=None):
    """Main function to run exclusion percentage analysis.

    Args:
        data_dir: Directory containing data
        slice_spec: Configuration slice (None = auto-infer most common)
        exclusion_percentages: List of percentages to evaluate (default: [0, 30, 60, 70, 80, 90])
        device: Device for evaluation (None = auto-detect)
    """
    if exclusion_percentages is None:
        exclusion_percentages = [0, 30, 60, 70, 80, 90]

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*70)
    print("EXCLUSION PERCENTAGE ANALYSIS")
    print("="*70)

    # Load test data
    print("\nLoading test data...")
    _, _, _, _, test_sequences, test_configs = load_real_data(
        data_dir, max_files=None
    )
    print(f"Loaded {len(test_sequences)} test sequences")

    # Infer or use provided slice
    if slice_spec is None:
        slice_keys = ['channel_model_id', 'N_t', 'N_r', 'BW', 'N_ss']
        slice_spec = infer_most_common_slice(test_configs, slice_keys)
    print(f"\nUsing slice: {slice_spec}")

    # Evaluate each model
    results_by_pct = {}

    for pct in exclusion_percentages:
        print(f"\n{'='*70}")
        print(f"EVALUATING {pct}% EXCLUSION")
        print(f"{'='*70}")

        # Load exclusion manifest
        manifest = load_exclusion_manifest(pct)

        # Get excluded MCS for this slice
        excluded_mcs, training_mcs = get_excluded_mcs_for_slice(manifest, slice_spec)
        print(f"  Training MCS: {training_mcs}")
        print(f"  Excluded MCS: {excluded_mcs}")

        # Load checkpoint
        checkpoint_path = f'pkd/trained_models/exclude {pct}/pkd_model.pt'

        if not os.path.exists(checkpoint_path):
            print(f"  Warning: Checkpoint not found: {checkpoint_path}")
            continue

        # Evaluate
        metrics = evaluate_model_on_slice(
            checkpoint_path, test_sequences, test_configs,
            slice_spec, excluded_mcs, training_mcs, device
        )

        results_by_pct[pct] = metrics

    # Generate plots
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    print(f"{'='*70}")

    generate_plots(results_by_pct, slice_spec)

    print(f"\n{'='*70}")
    print("DONE!")
    print(f"{'='*70}")
    print("\nPlots saved to figures/:")
    print("  - figures/exclusion_overall_ks.png")
    print("  - figures/exclusion_generalization_gap.png")
    print("  - figures/exclusion_acf_rmse.png")
    print("  - figures/exclusion_pit_passrate.png")

    return results_by_pct


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate PKD model performance across MCS exclusion percentages',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-select most common slice
  python -m pkd.example.evaluate_exclusion

  # Specify custom slice
  python -m pkd.example.evaluate_exclusion --channel-model 2 --N-t 3 --N-r 2 --N-ss 2 --BW 20.0

  # Evaluate subset of percentages
  python -m pkd.example.evaluate_exclusion --percentages 0 30 60
        """
    )

    # Slice specification arguments
    parser.add_argument('--channel-model', type=int,
                        help='Channel model ID (0-4)')
    parser.add_argument('--N-t', type=int,
                        help='Number of transmit antennas')
    parser.add_argument('--N-r', type=int,
                        help='Number of receive antennas')
    parser.add_argument('--N-ss', type=int,
                        help='Number of spatial streams')
    parser.add_argument('--BW', type=float,
                        help='Bandwidth in MHz (e.g., 20.0, 40.0)')

    # Other options
    parser.add_argument('--percentages', type=int, nargs='+',
                        default=[0, 30, 60, 70, 80, 90],
                        help='Exclusion percentages to evaluate (default: 0 30 60 70 80 90)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory (default: data)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device for evaluation (cuda/cpu, default: auto-detect)')

    args = parser.parse_args()

    # Build slice_spec if any slice arguments provided
    slice_spec = None
    slice_args = ['channel_model', 'N_t', 'N_r', 'N_ss', 'BW']
    provided_args = {arg: getattr(args, arg) for arg in slice_args
                     if getattr(args, arg) is not None}

    if provided_args:
        # Check if all slice arguments are provided
        if len(provided_args) != 5:
            parser.error('All slice arguments must be provided together: '
                        '--channel-model, --N-t, --N-r, --N-ss, --BW')

        slice_spec = {
            'channel_model_id': args.channel_model,
            'N_t': args.N_t,
            'N_r': args.N_r,
            'BW': args.BW,
            'N_ss': args.N_ss
        }

    run_exclusion_analysis(
        data_dir=args.data_dir,
        slice_spec=slice_spec,
        exclusion_percentages=args.percentages,
        device=args.device
    )
