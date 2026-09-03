"""Evaluate the Nearest-MCS and Piecewise-Linear-Interpolation EESM-log-AR
baselines under the same MCS-exclusion sweep used for PKD's Fig. 13
(pkd/example/evaluate_exclusion.py), so all three methods can be plotted
on the same axes.

Fair-comparison rules enforced here (see
self_review/EESM_LOG_AR_SPARSE_MCS_BASELINES.md, Sec. 5):
  Rule 1: same retained MCS set as PKD, from the same exclusion manifest.
  Rule 2: calibration only ever reads train-split sequences at retained
          MCS (never excluded MCS, never val/test).
  Rule 3: no interpolation across SNR -- calibration fits each (MCS, SNR)
          cell independently, and transfer never blends a single MCS's
          own params across two of its SNRs. Each MCS is calibrated on
          its own waterfall SNR grid, offset in dB from every other MCS's
          grid (Table II), so raw SNR (dB) is not a comparable axis
          across MCS. "Using retained MCS m's params for target MCS's
          i-th waterfall SNR" is resolved by taking m's own params at ITS
          i-th waterfall SNR (transfer.params_at_matching_rank /
          snr_rank) -- a lookup along the MCS-transfer axis at a matched
          relative position on each MCS's own PER curve, not SNR
          interpolation/blending.
  Rule 4: evaluation uses the same held-out test_sequences as PKD (same
          load_real_data args/seed as pkd/example/data_loader.py).
  Rule 5: same metrics (KS statistic, ACF RMSE) via the same
          pkd.example.utils.compute_acf and scipy.stats.ks_2samp used by
          pkd/example/plotting.py.
"""
import glob
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from pkd.example.data_loader import load_real_data
from pkd.example.exclusion_filter import apply_exclusion_filter, load_exclusion_config
from pkd.example.utils import compute_acf, filter_by_slice

from .calibration import calibrate_ar_params, ARParams
from .transfer import (
    build_baseline_params_for_all_mcs,
    params_at_matching_rank,
    snr_rank,
    nearest_tuple_transfer,
    CONFIG_AXES,
)
from .generate import generate_ar_sequence

TUPLE_KEYS = ['channel_model_id', 'N_t', 'N_r', 'BW', 'N_ss', 'MCS']

AR_ORDER = 10
ALL_MCS = list(range(10))


def _slice_matches(config: dict, slice_spec: dict) -> bool:
    return all(config.get(k) == v for k, v in slice_spec.items())


def _group_train_sequences_by_mcs_snr(
    train_sequences: List[np.ndarray],
    train_configs: List[dict],
    slice_spec: dict,
) -> Dict[int, Dict[int, List[np.ndarray]]]:
    """Group filtered training sequences by MCS, then by integer SNR, for a
    fixed configuration slice. Mirrors the (MCS, SNR) grouping in
    pkd/example/plotting.py::generate_figure1_per_mcs_metrics.
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for seq, cfg in zip(train_sequences, train_configs):
        if not _slice_matches(cfg, slice_spec):
            continue
        grouped[cfg['MCS']][int(cfg['SNR_bar'])].append(seq)
    return grouped


def calibrate_retained_mcs(
    train_sequences: List[np.ndarray],
    train_configs: List[dict],
    slice_spec: dict,
    retained_mcs: List[int],
    ar_order: int = AR_ORDER,
) -> Dict[int, Dict[int, ARParams]]:
    """Calibrate Phi(m, s) for every retained MCS m and every SNR s present
    in the (post-exclusion-filter) training data for this slice.

    Args:
        train_sequences, train_configs: training split AFTER the exclusion
            filter has already been applied (Rule 2 -- excluded MCS must
            not appear here at all, so there is nothing to accidentally
            calibrate from).
        slice_spec: fixed (channel_model_id, N_t, N_r, BW, N_ss) slice.
        retained_mcs: MCS values expected to be retained for this slice,
            per the exclusion manifest (used only to validate coverage).

    Returns:
        {mcs: {snr: ARParams}} for retained MCS actually found in the data.
    """
    grouped = _group_train_sequences_by_mcs_snr(train_sequences, train_configs, slice_spec)

    calibrated: Dict[int, Dict[int, ARParams]] = {}
    for mcs in retained_mcs:
        if mcs not in grouped:
            continue
        calibrated[mcs] = {}
        for snr, seqs in grouped[mcs].items():
            calibrated[mcs][snr] = calibrate_ar_params(seqs, ar_order=ar_order)

    return calibrated


def baseline_params_for_target(
    calibrated_by_mcs_snr: Dict[int, Dict[int, ARParams]],
    target_mcs: int,
    target_snr: int,
    target_mcs_snr_grid,
    method: str,
) -> ARParams:
    """Produce a baseline parameter estimate for one (target_mcs,
    target_snr) test cell.

    Each MCS is calibrated on its own waterfall SNR grid (Table II) that
    is offset in raw dB from every other MCS's grid (higher MCS needs
    higher SNR for the same PER), so matching by nearest SNR in dB would
    mix operating points from different parts of the PER curve. Instead,
    every retained MCS is reduced to its own params at the SAME
    waterfall-rank index as target_snr occupies within target_mcs's own
    grid (transfer.params_at_matching_rank), and the Sec. 3/4 MCS-transfer
    rule (nearest-MCS, or piecewise-linear interpolation between
    bracketing MCS) is applied on top of that per-MCS rank-matched
    snapshot. This keeps "no cross-SNR blending of a single MCS's own
    fit" (Rule 3) while answering every excluded-MCS test point at a
    comparable point on its waterfall curve.

    Args:
        calibrated_by_mcs_snr: {mcs: {snr: ARParams}} from
            calibrate_retained_mcs.
        target_mcs: MCS of the test cell being approximated.
        target_snr: SNR (dB) of the test cell being approximated.
        target_mcs_snr_grid: full sorted SNR grid for target_mcs (from the
            test set), used to compute target_snr's waterfall rank.
        method: 'nearest' or 'interpolate'.

    Returns:
        ARParams for (target_mcs, target_snr).
    """
    rank = snr_rank(target_snr, target_mcs_snr_grid)
    retained_at_rank = params_at_matching_rank(calibrated_by_mcs_snr, rank)
    if not retained_at_rank:
        raise ValueError(
            f"No retained MCS has a rank-{rank} SNR to build baseline params for MCS={target_mcs}."
        )

    return build_baseline_params_for_all_mcs(retained_at_rank, [target_mcs], method)[target_mcs]


def evaluate_baseline_on_slice(
    calibrated_by_mcs_snr: Dict[int, Dict[int, ARParams]],
    method: str,
    test_sequences: List[np.ndarray],
    test_configs: List[dict],
    slice_spec: dict,
    excluded_mcs: List[int],
    training_mcs: List[int],
    ar_order: int = AR_ORDER,
    seq_length: int = 1000,
    burn_in: int = 50,
    seed: int = 42,
) -> dict:
    """Score a baseline's generated sequences against the SAME test
    sequences used for PKD (Rule 4), with the SAME metrics (Rule 5).

    Baseline params are built per (mcs, snr) test cell on demand (see
    baseline_params_for_target), so every test point in the slice --
    including every excluded MCS at every one of its own waterfall SNRs
    -- is scoreable, matching PKD's coverage.

    Returns a dict with overall/seen/unseen median KS and overall median
    ACF RMSE, matching evaluate_exclusion.py::evaluate_model_on_slice's
    output shape so both can be plotted together.
    """
    rng = np.random.default_rng(seed)

    filtered_test_seq, filtered_test_cfg, _ = filter_by_slice(test_sequences, test_configs, slice_spec)

    grouped_test = defaultdict(list)
    snr_grid_by_mcs = defaultdict(set)
    for seq, cfg in zip(filtered_test_seq, filtered_test_cfg):
        mcs, snr = cfg['MCS'], int(cfg['SNR_bar'])
        grouped_test[(mcs, snr)].append(seq)
        snr_grid_by_mcs[mcs].add(snr)

    all_ks, seen_ks, unseen_ks, all_acf = [], [], [], []

    for (mcs, snr), teacher_seqs in sorted(grouped_test.items()):
        params = baseline_params_for_target(
            calibrated_by_mcs_snr, mcs, snr, snr_grid_by_mcs[mcs], method
        )

        ks_vals, acf_vals = [], []
        for teacher_seq in teacher_seqs:
            teacher_seq = np.asarray(teacher_seq)
            student_seq = generate_ar_sequence(
                params, length=len(teacher_seq), ar_order=ar_order, burn_in=burn_in, rng=rng
            )

            ks_stat, _ = stats.ks_2samp(teacher_seq, student_seq)
            ks_vals.append(ks_stat)

            teacher_acf = compute_acf(teacher_seq, max_lag=50)
            student_acf = compute_acf(student_seq, max_lag=50)
            acf_rmse = np.sqrt(np.mean((teacher_acf[1:] - student_acf[1:]) ** 2))
            acf_vals.append(acf_rmse)

        ks_med = np.median(ks_vals)
        acf_med = np.median(acf_vals)

        all_ks.append(ks_med)
        all_acf.append(acf_med)
        if mcs in training_mcs:
            seen_ks.append(ks_med)
        elif mcs in excluded_mcs:
            unseen_ks.append(ks_med)

    overall_ks = np.median(all_ks) if all_ks else np.nan
    seen_ks_med = np.median(seen_ks) if seen_ks else np.nan
    unseen_ks_med = np.median(unseen_ks) if unseen_ks else np.nan
    overall_acf = np.median(all_acf) if all_acf else np.nan

    return {
        'overall_ks': overall_ks,
        'seen_ks': seen_ks_med,
        'unseen_ks': unseen_ks_med,
        'generalization_gap': unseen_ks_med - seen_ks_med,
        'overall_acf': overall_acf,
    }


def _load_exclusion_yaml_for_pct(pct: int) -> Optional[dict]:
    """Load the YAML exclusion config (not the JSON manifest) so we can
    actually filter training sequences, mirroring pkd/example/main.py's
    example_training workflow.
    """
    pattern = f'pkd/example/exclusions/exclusions_random_{pct}pct_seed42_*.yaml'
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return load_exclusion_config(files[-1])


def _get_excluded_training_mcs_for_slice(exclusion_config: dict, slice_spec: dict, all_mcs_in_slice: List[int]):
    if exclusion_config is None:
        return [], all_mcs_in_slice

    for rule in exclusion_config.get('exclusions', []):
        if all(rule['config'].get(k) == v for k, v in slice_spec.items()):
            excluded = rule['mcs_list']
            training = sorted(set(all_mcs_in_slice) - set(excluded))
            return excluded, training

    return [], all_mcs_in_slice


def run_baseline_exclusion_analysis(
    data_dir: str = 'data',
    slice_spec: Optional[dict] = None,
    exclusion_percentages: Optional[List[int]] = None,
    ar_order: int = AR_ORDER,
    max_files: Optional[int] = None,
) -> Dict[str, Dict[int, dict]]:
    """Run both baselines across the same exclusion percentages used for
    PKD's Fig. 13, using the same data split/seed and the same test set.

    Returns:
        {'nearest': {pct: metrics}, 'interpolate': {pct: metrics}}
    """
    if exclusion_percentages is None:
        exclusion_percentages = [0, 30, 60, 70, 80, 90]
    if slice_spec is None:
        slice_spec = {
            'channel_model_id': 2,   # Model-B
            'N_t': 3,
            'N_r': 2,
            'BW': 40.0,
            'N_ss': 2,
        }

    print("Loading data (train/val/test split, seed=42, matching PKD)...")
    train_sequences, train_configs, _, _, test_sequences, test_configs = load_real_data(
        data_dir, train_ratio=0.7, val_ratio=0.1, max_files=max_files, random_seed=42
    )

    slice_mcs_in_test = sorted({cfg['MCS'] for cfg in test_configs if _slice_matches(cfg, slice_spec)})

    results = {'nearest': {}, 'interpolate': {}}

    for pct in exclusion_percentages:
        print(f"\n{'=' * 70}\nBASELINES: {pct}% MCS EXCLUSION\n{'=' * 70}")

        exclusion_config = _load_exclusion_yaml_for_pct(pct) if pct > 0 else None
        excluded_mcs, training_mcs = _get_excluded_training_mcs_for_slice(
            exclusion_config, slice_spec, slice_mcs_in_test
        )
        print(f"  Training MCS: {training_mcs}")
        print(f"  Excluded MCS: {excluded_mcs}")

        # Rule 2: filter train split to retained MCS only, exactly as PKD's
        # own training pipeline does (pkd/example/exclusion_filter.py).
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
            metrics = evaluate_baseline_on_slice(
                calibrated, method, test_sequences, test_configs, slice_spec,
                excluded_mcs, training_mcs, ar_order=ar_order,
            )
            print(f"  [{method}] Overall KS: {metrics['overall_ks']:.4f}, "
                  f"Unseen KS: {metrics['unseen_ks']:.4f}, "
                  f"Overall ACF RMSE: {metrics['overall_acf']:.4f}")
            results[method][pct] = metrics

    return results


# ============================================================
# Full-tuple exclusion (Section V-D generalization from MCS-only
# exclusion to joint (CH, MCS, N_t, N_r, N_ss, BW) exclusion). These
# functions are additive: they do not modify calibrate_retained_mcs,
# evaluate_baseline_on_slice, or run_baseline_exclusion_analysis above,
# which remain the MCS-only-exclusion code path.
# ============================================================

def _tuple_dict(config: dict) -> Dict[str, float]:
    """Extract the (channel_model_id, BW, N_t, N_r, N_ss, MCS) tuple dict
    from a sequence config, keyed as expected by config_tuple_distance /
    nearest_tuple_transfer."""
    return {k: config[k] for k in CONFIG_AXES}


def calibrate_all_retained_tuples(
    train_sequences: List[np.ndarray],
    train_configs: List[dict],
    ar_order: int = AR_ORDER,
) -> Dict[Tuple, Dict[int, ARParams]]:
    """Calibrate Phi(tuple, snr) for every retained (channel_model_id,
    N_t, N_r, BW, N_ss, MCS) tuple and every SNR present in the
    (post-exclusion-filter) training data, across the WHOLE dataset
    rather than one fixed slice.

    This is the full-tuple-exclusion counterpart to calibrate_retained_mcs
    (Rule 2 still applies: train_sequences/train_configs must already have
    excluded tuples filtered out, so there is nothing to accidentally
    calibrate from).

    Args:
        train_sequences, train_configs: training split AFTER the
            full-tuple exclusion filter has already been applied.
        ar_order: AR model order for calibration.

    Returns:
        {tuple_key: {snr: ARParams}} where tuple_key is a sorted tuple of
        (axis, value) pairs over CONFIG_AXES (hashable, dict-independent).
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for seq, cfg in zip(train_sequences, train_configs):
        key = tuple(sorted(_tuple_dict(cfg).items()))
        grouped[key][int(cfg['SNR_bar'])].append(seq)

    calibrated: Dict[Tuple, Dict[int, ARParams]] = {}
    for tuple_key, snr_groups in grouped.items():
        calibrated[tuple_key] = {}
        for snr, seqs in snr_groups.items():
            calibrated[tuple_key][snr] = calibrate_ar_params(seqs, ar_order=ar_order)

    return calibrated


def _axis_ranges(all_tuple_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """Observed max-min range for each numeric CONFIG_AXES axis (BW, N_t,
    N_r, N_ss), for use as config_tuple_distance's normalization."""
    ranges = {}
    for axis in ('BW', 'N_t', 'N_r', 'N_ss'):
        vals = [t[axis] for t in all_tuple_dicts]
        ranges[axis] = max(vals) - min(vals)
    return ranges


def baseline_params_for_target_tuple(
    calibrated_by_tuple_snr: Dict[Tuple, Dict[int, ARParams]],
    target_tuple: Dict[str, float],
    target_snr: int,
    target_snr_grid,
    axis_ranges: Dict[str, float],
    all_mcs: List[int],
) -> ARParams:
    """Produce a baseline parameter estimate for one excluded (tuple, snr)
    test cell using nearest_tuple_transfer, the full-tuple generalization
    of baseline_params_for_target.

    Mirrors baseline_params_for_target's SNR handling exactly (Rule 3):
    target_snr is converted to its waterfall rank within target_tuple's
    own SNR grid, the nearest retained tuple is found once via
    nearest_tuple_transfer, and that tuple's own params at the SAME
    waterfall rank are returned -- never a different tuple's params
    blended across SNR, and never cross-SNR interpolation within a tuple.

    Args:
        calibrated_by_tuple_snr: {tuple_key: {snr: ARParams}} from
            calibrate_all_retained_tuples.
        target_tuple: excluded tuple dict (CONFIG_AXES keys).
        target_snr: SNR (dB) of the test cell being approximated.
        target_snr_grid: full sorted SNR grid for target_tuple (from the
            test set), used to compute target_snr's waterfall rank.
        axis_ranges: passed through to nearest_tuple_transfer.
        all_mcs: full sorted MCS set (typically 0-9).

    Returns:
        ARParams for (target_tuple, target_snr).
    """
    retained_tuples = [dict(k) for k in calibrated_by_tuple_snr.keys()]
    nearest = nearest_tuple_transfer(target_tuple, retained_tuples, axis_ranges, all_mcs)
    nearest_key = tuple(sorted(nearest.items()))

    rank = snr_rank(target_snr, target_snr_grid)
    nearest_snrs = sorted(calibrated_by_tuple_snr[nearest_key].keys())
    if rank >= len(nearest_snrs):
        raise ValueError(
            f"Nearest tuple {nearest} has no rank-{rank} SNR to approximate "
            f"target tuple {target_tuple} at SNR={target_snr}."
        )
    return calibrated_by_tuple_snr[nearest_key][nearest_snrs[rank]]


def evaluate_nearest_tuple_baseline(
    calibrated_by_tuple_snr: Dict[Tuple, Dict[int, ARParams]],
    test_sequences: List[np.ndarray],
    test_configs: List[dict],
    excluded_tuples: List[Dict[str, float]],
    axis_ranges: Dict[str, float],
    all_mcs: List[int],
    ar_order: int = AR_ORDER,
    burn_in: int = 50,
    seed: int = 42,
) -> Dict[str, float]:
    """Score the nearest-tuple-transfer baseline against the same excluded
    tuples' test sequences used for PKD, aggregating two-level (per-tuple
    median across sequences, then median + 10-90 percentile across
    tuples) exactly as pkd.example.plotting's *_by_tuple figure functions.

    Returns:
        dict with 'overall_ks_median', 'overall_ks_p10', 'overall_ks_p90',
        'overall_acf_median', 'overall_acf_p10', 'overall_acf_p90', plus
        'per_tuple_ks' and 'per_tuple_acf' for inspection.
    """
    rng = np.random.default_rng(seed)

    excluded_keys = {tuple(sorted(t.items())) for t in excluded_tuples}

    grouped_test = defaultdict(lambda: defaultdict(list))
    snr_grid_by_tuple = defaultdict(set)
    for seq, cfg in zip(test_sequences, test_configs):
        key = tuple(sorted(_tuple_dict(cfg).items()))
        if key not in excluded_keys:
            continue
        snr = int(cfg['SNR_bar'])
        grouped_test[key][snr].append(seq)
        snr_grid_by_tuple[key].add(snr)

    per_tuple_ks, per_tuple_acf = {}, {}

    for tuple_key, snr_groups in sorted(grouped_test.items()):
        target_tuple = dict(tuple_key)
        ks_vals, acf_vals = [], []

        for snr, teacher_seqs in sorted(snr_groups.items()):
            params = baseline_params_for_target_tuple(
                calibrated_by_tuple_snr, target_tuple, snr,
                snr_grid_by_tuple[tuple_key], axis_ranges, all_mcs,
            )

            for teacher_seq in teacher_seqs:
                teacher_seq = np.asarray(teacher_seq)
                student_seq = generate_ar_sequence(
                    params, length=len(teacher_seq), ar_order=ar_order,
                    burn_in=burn_in, rng=rng,
                )

                ks_stat, _ = stats.ks_2samp(teacher_seq, student_seq)
                ks_vals.append(ks_stat)

                teacher_acf = compute_acf(teacher_seq, max_lag=50)
                student_acf = compute_acf(student_seq, max_lag=50)
                acf_vals.append(np.sqrt(np.mean((teacher_acf[1:] - student_acf[1:]) ** 2)))

        if ks_vals:
            per_tuple_ks[tuple_key] = np.median(ks_vals)
            per_tuple_acf[tuple_key] = np.median(acf_vals)

    ks_arr = np.array(list(per_tuple_ks.values()))
    acf_arr = np.array(list(per_tuple_acf.values()))

    return {
        'overall_ks_median': np.median(ks_arr) if len(ks_arr) else np.nan,
        'overall_ks_p10': np.percentile(ks_arr, 10) if len(ks_arr) else np.nan,
        'overall_ks_p90': np.percentile(ks_arr, 90) if len(ks_arr) else np.nan,
        'overall_acf_median': np.median(acf_arr) if len(acf_arr) else np.nan,
        'overall_acf_p10': np.percentile(acf_arr, 10) if len(acf_arr) else np.nan,
        'overall_acf_p90': np.percentile(acf_arr, 90) if len(acf_arr) else np.nan,
        'per_tuple_ks': per_tuple_ks,
        'per_tuple_acf': per_tuple_acf,
    }


def _load_config_exclusion_yaml_for_pct(pct: int) -> Optional[dict]:
    """Load the full-tuple-exclusion YAML config for a given percentage,
    generated by generate_random_config_exclusions/save_exclusion_config.
    Mirrors _load_exclusion_yaml_for_pct's glob convention but for the
    'exclusions_config_random_*' naming used for tuple-exclusion runs.
    """
    pattern = f'pkd/example/exclusions/exclusions_config_random_{pct}pct_seed42_*.yaml'
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return load_exclusion_config(files[-1])


def run_baseline_tuple_exclusion_analysis(
    data_dir: str = 'data',
    exclusion_percentages: Optional[List[int]] = None,
    ar_order: int = AR_ORDER,
    max_files: Optional[int] = None,
) -> Dict[int, dict]:
    """Run the nearest-tuple-transfer baseline across the full-tuple
    exclusion sweep (Fig. 13's generalized x-axis), using the same data
    split/seed and the same test set as PKD.

    This is the full-tuple-exclusion counterpart to
    run_baseline_exclusion_analysis: instead of one hardcoded
    (CH, N_t, N_r, BW, N_ss) slice with MCS swept 0-90%, this draws the
    excluded set from generate_random_config_exclusions (or its saved
    YAML) and calibrates/searches over the WHOLE dataset. Only the
    nearest-tuple baseline is evaluated here (piecewise-linear
    interpolation does not generalize cleanly to a 6-axis discrete grid
    with no natural total ordering, per the Section V-D design decision).

    Returns:
        {pct: metrics} where metrics matches evaluate_nearest_tuple_baseline's
        return shape.
    """
    if exclusion_percentages is None:
        exclusion_percentages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

    print("Loading data (train/val/test split, seed=42, matching PKD)...")
    train_sequences, train_configs, _, _, test_sequences, test_configs = load_real_data(
        data_dir, train_ratio=0.7, val_ratio=0.1, max_files=max_files, random_seed=42
    )

    all_mcs = sorted({cfg['MCS'] for cfg in train_configs + test_configs})
    all_tuple_dicts = [_tuple_dict(cfg) for cfg in train_configs + test_configs]
    axis_ranges = _axis_ranges(all_tuple_dicts)

    results = {}

    for pct in exclusion_percentages:
        print(f"\n{'=' * 70}\nBASELINE (nearest-tuple): {pct}% CONFIG-TUPLE EXCLUSION\n{'=' * 70}")

        exclusion_config = _load_config_exclusion_yaml_for_pct(pct) if pct > 0 else None

        if exclusion_config is not None:
            filtered_train_seq, filtered_train_cfg, _ = apply_exclusion_filter(
                train_sequences, train_configs, exclusion_config, split_name='train'
            )
            excluded_tuples = []
            for rule in exclusion_config['exclusions']:
                for mcs in rule['mcs_list']:
                    excluded_tuples.append({**rule['config'], 'MCS': mcs})
        else:
            filtered_train_seq, filtered_train_cfg = train_sequences, train_configs
            excluded_tuples = []

        print(f"  Excluded tuples: {len(excluded_tuples)}")

        if not excluded_tuples:
            print("  No excluded tuples at this percentage, skipping.")
            continue

        calibrated = calibrate_all_retained_tuples(
            filtered_train_seq, filtered_train_cfg, ar_order=ar_order
        )

        metrics = evaluate_nearest_tuple_baseline(
            calibrated, test_sequences, test_configs, excluded_tuples,
            axis_ranges, all_mcs, ar_order=ar_order,
        )
        print(f"  [nearest_tuple] Median KS: {metrics['overall_ks_median']:.4f}, "
              f"Median ACF RMSE: {metrics['overall_acf_median']:.4f}")
        results[pct] = metrics

    return results


if __name__ == '__main__':
    run_baseline_exclusion_analysis()
