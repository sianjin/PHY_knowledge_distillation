"""PER-vs-SNR waterfall comparison: ground-truth PHY simulator PER vs.
PKD-generated PER vs. EESM-log-AR baseline (Nearest-MCS, Piecewise-Linear
Interpolation)-generated PER, for one excluded MCS at one exclusion level.

Ground truth PER at each (MCS, SNR) point is read directly from the raw
.mat file's packet_error array (packet_error.mean()), independent of the
train/test split -- PER-vs-SNR is a fixed property of the PHY simulation,
not something that depends on how sequences were split.

PKD/baseline PER is estimated by generating a free-running gamma_eff
sequence (PKD via pkd.infer.PKDInference, baselines via
pkd.baselines.generate.generate_ar_sequence) at each of the target MCS's
own waterfall SNR points, then averaging AWGNPERLookup.lookup(gamma_eff_db,
mcs, packet_length=1000) over the sequence. packet_length=1000 matches the
APEPLength=1000 payload used to generate data/*.mat
(phy/teacher-data-generation/*.m),
so this is the same PER definition as the ground truth, not the LUT's
native L0=1458 default.

See self_review/EESM_LOG_AR_SPARSE_MCS_BASELINES.md for the baseline
design this compares against.
"""
import glob
import os
import re
from typing import Dict, List, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from pkd.infer import PKDInference
from pkd.model import PKDModel
from pkd.per_lut import AWGNPERLookup

from pkd.example.data_loader import load_real_data
from pkd.example.exclusion_filter import apply_exclusion_filter

from .evaluate_baselines import (
    AR_ORDER,
    _load_exclusion_yaml_for_pct,
    _get_excluded_training_mcs_for_slice,
    _slice_matches,
    baseline_params_for_target,
    calibrate_retained_mcs,
    _load_config_exclusion_yaml_for_pct,
    _tuple_dict,
    _axis_ranges,
    calibrate_all_retained_tuples,
    baseline_params_for_target_tuple,
)
from .generate import generate_ar_sequence

PACKET_LENGTH = 1000  # matches APEPLength=1000 used to generate data/*.mat

METHOD_STYLE = {
    'ground_truth': dict(color='black', marker='D', label='Ground Truth (PHY Simulator)'),
    'pkd': dict(color='#2E86AB', marker='o', label='PKD'),
    'interpolate': dict(color='#06A77D', marker='s', label='EESM-log-AR + Piecewise-Linear Interp.'),
    'nearest': dict(color='#D62839', marker='^', label='EESM-log-AR + Nearest-MCS'),
}


def _mat_filename(slice_spec: dict, mcs: int, snr: int) -> str:
    bw = slice_spec['BW']
    bw_str = str(int(bw)) if float(bw).is_integer() else str(bw)
    model_names = {0: 'Model-A', 1: 'A', 2: 'Model-B', 3: 'Model-C', 4: 'Model-D'}
    model_name = model_names.get(slice_spec['channel_model_id'], f"Model-{slice_spec['channel_model_id']}")
    return (
        f"CBW{bw_str}_{model_name}_"
        f"{slice_spec['N_t']}-by-{slice_spec['N_r']}-by-{slice_spec['N_ss']}_"
        f"MCS{mcs}_SNR{snr}.mat"
    )


def ground_truth_per_waterfall(data_dir: str, slice_spec: dict, mcs: int) -> Dict[int, float]:
    """Read ground-truth PER(SNR) for one MCS directly from data/*.mat,
    using every packet in each SNR's file (packet_error.mean()).

    Returns {snr: per}.
    """
    pattern = os.path.join(data_dir, f"*MCS{mcs}_SNR*.mat")
    candidates = glob.glob(pattern)

    # Filter to files that actually match this exact slice (bandwidth,
    # channel model, antenna config), since glob on MCS alone can match
    # multiple slices (e.g. different BW/antenna configs share MCS names).
    snr_re = re.compile(r"SNR(-?\d+)\.mat$")
    per_by_snr = {}
    expected_prefix = _mat_filename(slice_spec, mcs, 0).rsplit('_SNR', 1)[0]
    for path in candidates:
        fname = os.path.basename(path)
        if not fname.startswith(expected_prefix + '_SNR'):
            continue
        m = snr_re.search(fname)
        if not m:
            continue
        snr = int(m.group(1))
        with h5py.File(path, 'r') as f:
            per_by_snr[snr] = float(np.asarray(f['packet_error']).mean())

    return dict(sorted(per_by_snr.items()))


def model_per_waterfall(
    checkpoint_path: str,
    slice_spec: dict,
    mcs: int,
    snr_grid: List[int],
    seq_length: int = 1000,
    n_sequences: int = 50,
    ar_order: int = AR_ORDER,
    device: Optional[str] = None,
    seed: int = 42,
) -> Dict[int, float]:
    """Generate PKD sequences at each SNR in snr_grid for the given
    (slice, mcs) and estimate PER via the AWGN PER LUT, matching
    packet_length=1000 used to generate the ground-truth data.

    Returns {snr: per}.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['model_config']
    if 'num_R' not in model_config:
        model_config['num_R'] = 8
    model = PKDModel(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    per_lut = AWGNPERLookup.load_ldpc_lut()
    inference = PKDInference(model, per_lut, ar_order=ar_order, device=device)

    np.random.seed(seed)  # PKDInference draws from the global np.random state
    per_by_snr = {}
    for snr in snr_grid:
        config = dict(slice_spec)
        config.update({'SNR_bar': float(snr), 'MCS': mcs, 'R_t': 0, 'packet_length': PACKET_LENGTH})

        pers = []
        for _ in range(n_sequences):
            config_traj = [config] * seq_length
            results = inference.run_sequence(config_traj)
            gamma_eff_db = results['gamma_eff']
            per = per_lut.lookup(gamma_eff_db, mcs, packet_length=PACKET_LENGTH)
            pers.append(np.mean(per))
        per_by_snr[snr] = float(np.mean(pers))

    return per_by_snr


def baseline_per_waterfall(
    calibrated_by_mcs_snr: dict,
    method: str,
    slice_spec: dict,
    mcs: int,
    snr_grid: List[int],
    seq_length: int = 1000,
    n_sequences: int = 50,
    ar_order: int = AR_ORDER,
    burn_in: int = 50,
    seed: int = 42,
) -> Dict[int, float]:
    """Generate baseline (nearest/interpolate) sequences at each SNR in
    snr_grid for the target mcs, and estimate PER the same way as
    model_per_waterfall (same PER LUT, same packet_length=1000).

    Returns {snr: per}.
    """
    per_lut = AWGNPERLookup.load_ldpc_lut()
    rng = np.random.default_rng(seed)

    per_by_snr = {}
    for snr in snr_grid:
        params = baseline_params_for_target(calibrated_by_mcs_snr, mcs, snr, snr_grid, method)

        pers = []
        for _ in range(n_sequences):
            X = generate_ar_sequence(params, length=seq_length, ar_order=ar_order, burn_in=burn_in, rng=rng)
            gamma_eff_db = X * 10 / np.log(10)
            per = per_lut.lookup(gamma_eff_db, mcs, packet_length=PACKET_LENGTH)
            pers.append(np.mean(per))
        per_by_snr[snr] = float(np.mean(pers))

    return per_by_snr


def run_per_waterfall_comparison(
    data_dir: str = 'data',
    slice_spec: Optional[dict] = None,
    target_mcs: int = 7,
    exclusion_pct: int = 80,
    ar_order: int = AR_ORDER,
    device: Optional[str] = None,
    max_files: Optional[int] = None,
    num_snr_points: Optional[int] = None,
) -> Dict[str, Dict[int, float]]:
    """Compare ground-truth vs. PKD vs. baseline PER-vs-SNR waterfall
    curves for one excluded MCS at one exclusion percentage.

    Args:
        num_snr_points: if given, only the lowest `num_snr_points` SNR
            values (out of the MCS's full 10-point waterfall grid) are
            evaluated/plotted, e.g. to stay within the PHY-relevant
            operating range without truncating PER on the y-axis.

    Returns {'ground_truth': {snr: per}, 'pkd': {...}, 'nearest': {...},
    'interpolate': {...}}.
    """
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

    print(f"Ground truth PER waterfall for MCS {target_mcs} from {data_dir}...")
    gt_full = ground_truth_per_waterfall(data_dir, slice_spec, target_mcs)
    snr_grid = sorted(gt_full.keys())
    if num_snr_points is not None:
        snr_grid = snr_grid[:num_snr_points]
    gt = {snr: gt_full[snr] for snr in snr_grid}
    print(f"  SNR grid: {snr_grid}")
    for snr, per in gt.items():
        print(f"    SNR={snr:>4d}  PER={per:.4e}")

    print("\nLoading data (train/val/test split, seed=42, matching PKD)...")
    train_sequences, train_configs, _, _, test_sequences, test_configs = load_real_data(
        data_dir, train_ratio=0.7, val_ratio=0.1, max_files=max_files, random_seed=42
    )
    slice_mcs_in_test = sorted({cfg['MCS'] for cfg in test_configs if _slice_matches(cfg, slice_spec)})

    results = {'ground_truth': gt}

    # ---- PKD ----
    checkpoint_path = f'pkd/trained_models/exclude_mcs_{exclusion_pct}/pkd_model.pt'
    if os.path.exists(checkpoint_path):
        print(f"\nGenerating PKD PER waterfall (checkpoint: {checkpoint_path})...")
        pkd_per = model_per_waterfall(checkpoint_path, slice_spec, target_mcs, snr_grid, device=device)
        results['pkd'] = pkd_per
        for snr, per in pkd_per.items():
            print(f"    SNR={snr:>4d}  PER={per:.4e}")
    else:
        print(f"  Warning: PKD checkpoint not found: {checkpoint_path}")

    # ---- Baselines ----
    exclusion_config = _load_exclusion_yaml_for_pct(exclusion_pct) if exclusion_pct > 0 else None
    excluded_mcs, training_mcs = _get_excluded_training_mcs_for_slice(
        exclusion_config, slice_spec, slice_mcs_in_test
    )
    print(f"\n{exclusion_pct}% exclusion -- training MCS: {training_mcs}, excluded MCS: {excluded_mcs}")

    if exclusion_config is not None:
        filtered_train_seq, filtered_train_cfg, _ = apply_exclusion_filter(
            train_sequences, train_configs, exclusion_config, split_name='train'
        )
    else:
        filtered_train_seq, filtered_train_cfg = train_sequences, train_configs

    calibrated = calibrate_retained_mcs(
        filtered_train_seq, filtered_train_cfg, slice_spec, training_mcs, ar_order=ar_order
    )

    for method in ('nearest', 'interpolate'):
        print(f"\nGenerating {method} PER waterfall...")
        baseline_per = baseline_per_waterfall(calibrated, method, slice_spec, target_mcs, snr_grid, ar_order=ar_order)
        results[method] = baseline_per
        for snr, per in baseline_per.items():
            print(f"    SNR={snr:>4d}  PER={per:.4e}")

    return results


def generate_per_waterfall_plot(
    results: Dict[str, Dict[int, float]],
    target_mcs: int,
    exclusion_pct: int,
    save_path: str = 'figures/per_waterfall_comparison.png',
):
    """Plot PER (log scale) vs. SNR for ground truth + all methods.

    Every method shows the same SNR points (whatever is present in
    `results`), with no y-axis cropping.
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid')

    plt.figure(figsize=(8, 6))
    for key in ('ground_truth', 'pkd', 'interpolate', 'nearest'):
        if key not in results:
            continue
        snrs = sorted(results[key].keys())
        pers = [results[key][s] for s in snrs]
        style = METHOD_STYLE[key]
        linestyle = '--' if key == 'ground_truth' else '-'
        plt.semilogy(snrs, pers, linestyle + style['marker'], linewidth=2, markersize=8,
                     color=style['color'], label=style['label'])

    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('PER', fontsize=12)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


# ============================================================
# Full-tuple exclusion (Fig. 14 generalization, Section V-D): PER-SNR
# waterfall for one representative fully-excluded (CH, MCS, N_t, N_r,
# N_ss, BW) tuple. PKD vs. nearest-tuple transfer vs. ground truth only
# (no piecewise-linear interpolation baseline, per the Section V-D design
# decision -- interpolation does not generalize cleanly to a 6-axis
# discrete grid with no natural total ordering). Additive: does not
# modify run_per_waterfall_comparison/generate_per_waterfall_plot above.
# ============================================================

TUPLE_METHOD_STYLE = {
    'ground_truth': dict(color='black', marker='D', label='Ground Truth (PHY Simulator)'),
    'pkd': dict(color='#2E86AB', marker='o', label='PKD'),
    'nearest_tuple': dict(color='#D62839', marker='^', label='EESM-log-AR + Nearest-Tuple'),
}


def baseline_per_waterfall_tuple(
    calibrated_by_tuple_snr: dict,
    target_tuple: Dict[str, float],
    snr_grid: List[int],
    axis_ranges: Dict[str, float],
    all_mcs: List[int],
    seq_length: int = 1000,
    n_sequences: int = 50,
    ar_order: int = AR_ORDER,
    burn_in: int = 50,
    seed: int = 42,
) -> Dict[int, float]:
    """Generate nearest-tuple-transfer baseline sequences at each SNR in
    snr_grid for target_tuple, and estimate PER the same way as
    model_per_waterfall (same PER LUT, same packet_length=1000).

    Returns {snr: per}.
    """
    per_lut = AWGNPERLookup.load_ldpc_lut()
    rng = np.random.default_rng(seed)

    per_by_snr = {}
    for snr in snr_grid:
        params = baseline_params_for_target_tuple(
            calibrated_by_tuple_snr, target_tuple, snr, snr_grid, axis_ranges, all_mcs
        )

        pers = []
        for _ in range(n_sequences):
            X = generate_ar_sequence(params, length=seq_length, ar_order=ar_order, burn_in=burn_in, rng=rng)
            gamma_eff_db = X * 10 / np.log(10)
            per = per_lut.lookup(gamma_eff_db, target_tuple['MCS'], packet_length=PACKET_LENGTH)
            pers.append(np.mean(per))
        per_by_snr[snr] = float(np.mean(pers))

    return per_by_snr


def run_per_waterfall_comparison_tuple(
    data_dir: str = 'data',
    target_tuple: Optional[Dict[str, float]] = None,
    exclusion_pct: int = 30,
    ar_order: int = AR_ORDER,
    device: Optional[str] = None,
    max_files: Optional[int] = None,
    num_snr_points: Optional[int] = None,
) -> Dict[str, Dict[int, float]]:
    """Compare ground-truth vs. PKD vs. nearest-tuple-transfer PER-vs-SNR
    waterfall curves for one fully-excluded (CH, MCS, N_t, N_r, N_ss, BW)
    tuple at one exclusion percentage.

    This is the full-tuple-exclusion counterpart to
    run_per_waterfall_comparison: target_tuple's (channel_model_id, N_t,
    N_r, BW, N_ss) fields are used as the slice_spec for ground-truth
    lookup and PKD generation (both are keyed by slice + MCS already), and
    the baseline is nearest_tuple_transfer over the WHOLE retained-tuple
    pool rather than nearest_mcs_transfer within one fixed slice.

    Args:
        target_tuple: dict with channel_model_id, N_t, N_r, BW, N_ss, MCS
            -- the tuple to feature (should be one of the tuples actually
            excluded by the exclusion_pct% run being evaluated).
        num_snr_points: if given, only the lowest `num_snr_points` SNR
            values are evaluated/plotted.

    Returns {'ground_truth': {snr: per}, 'pkd': {...}, 'nearest_tuple': {...}}.
    """
    if target_tuple is None:
        raise ValueError("target_tuple must be provided (no default -- see Fig. 9/11 tuple selection).")
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    slice_spec = {k: target_tuple[k] for k in ('channel_model_id', 'N_t', 'N_r', 'BW', 'N_ss')}
    target_mcs = target_tuple['MCS']

    print(f"Ground truth PER waterfall for tuple {target_tuple} from {data_dir}...")
    gt_full = ground_truth_per_waterfall(data_dir, slice_spec, target_mcs)
    snr_grid = sorted(gt_full.keys())
    if num_snr_points is not None:
        snr_grid = snr_grid[:num_snr_points]
    gt = {snr: gt_full[snr] for snr in snr_grid}
    print(f"  SNR grid: {snr_grid}")
    for snr, per in gt.items():
        print(f"    SNR={snr:>4d}  PER={per:.4e}")

    print("\nLoading data (train/val/test split, seed=42, matching PKD)...")
    train_sequences, train_configs, _, _, test_sequences, test_configs = load_real_data(
        data_dir, train_ratio=0.7, val_ratio=0.1, max_files=max_files, random_seed=42
    )

    results = {'ground_truth': gt}

    # ---- PKD ----
    checkpoint_path = f'pkd/trained_models/exclude_config_{exclusion_pct}/pkd_model.pt'
    if os.path.exists(checkpoint_path):
        print(f"\nGenerating PKD PER waterfall (checkpoint: {checkpoint_path})...")
        pkd_per = model_per_waterfall(checkpoint_path, slice_spec, target_mcs, snr_grid, device=device)
        results['pkd'] = pkd_per
        for snr, per in pkd_per.items():
            print(f"    SNR={snr:>4d}  PER={per:.4e}")
    else:
        print(f"  Warning: PKD checkpoint not found: {checkpoint_path}")

    # ---- Nearest-tuple baseline ----
    exclusion_config = _load_config_exclusion_yaml_for_pct(exclusion_pct) if exclusion_pct > 0 else None
    if exclusion_config is not None:
        filtered_train_seq, filtered_train_cfg, _ = apply_exclusion_filter(
            train_sequences, train_configs, exclusion_config, split_name='train'
        )
    else:
        filtered_train_seq, filtered_train_cfg = train_sequences, train_configs

    all_mcs = sorted({cfg['MCS'] for cfg in train_configs + test_configs})
    all_tuple_dicts = [_tuple_dict(cfg) for cfg in train_configs + test_configs]
    axis_ranges = _axis_ranges(all_tuple_dicts)

    calibrated = calibrate_all_retained_tuples(filtered_train_seq, filtered_train_cfg, ar_order=ar_order)

    print("\nGenerating nearest-tuple baseline PER waterfall...")
    baseline_per = baseline_per_waterfall_tuple(
        calibrated, target_tuple, snr_grid, axis_ranges, all_mcs, ar_order=ar_order
    )
    results['nearest_tuple'] = baseline_per
    for snr, per in baseline_per.items():
        print(f"    SNR={snr:>4d}  PER={per:.4e}")

    return results


def generate_per_waterfall_plot_tuple(
    results: Dict[str, Dict[int, float]],
    target_tuple: Dict[str, float],
    exclusion_pct: int,
    save_path: str = 'figures/per_waterfall_comparison_tuple.png',
):
    """Plot PER (log scale) vs. SNR for ground truth + PKD + nearest-tuple,
    for one fully-excluded tuple. Mirrors generate_per_waterfall_plot's
    style but with only 3 series (no piecewise-linear interpolation).
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid')

    plt.figure(figsize=(8, 6))
    for key in ('ground_truth', 'pkd', 'nearest_tuple'):
        if key not in results:
            continue
        snrs = sorted(results[key].keys())
        pers = [results[key][s] for s in snrs]
        style = TUPLE_METHOD_STYLE[key]
        linestyle = '--' if key == 'ground_truth' else '-'
        plt.semilogy(snrs, pers, linestyle + style['marker'], linewidth=2, markersize=8,
                     color=style['color'], label=style['label'])

    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('PER', fontsize=12)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


if __name__ == '__main__':
    results = run_per_waterfall_comparison(target_mcs=7, exclusion_pct=80, num_snr_points=7)
    generate_per_waterfall_plot(results, target_mcs=7, exclusion_pct=80)
