"""Baseline 1 (Nearest-MCS) and Baseline 2 (Piecewise-Linear MCS
Interpolation) as defined in
self_review/EESM_LOG_AR_SPARSE_MCS_BASELINES.md, Sections 3-4.

Both operate purely on already-calibrated ARParams objects and never touch
raw sequences, so they cannot accidentally read excluded-MCS or test data.
"""
from typing import Dict, List

import numpy as np

from .calibration import ARParams
from .pacf_transform import ar_to_pacf, pacf_to_ar


def nearest_mcs_transfer(target_mcs: int, retained_params: Dict[int, ARParams]) -> ARParams:
    """Baseline 1: reuse the calibrated params of the numerically nearest
    retained MCS. Ties broken toward the lower MCS (Sec. 3, tie-breaking
    rule).

    Args:
        target_mcs: excluded MCS to approximate.
        retained_params: {mcs: ARParams} for retained MCS at this (slice, SNR).

    Returns:
        ARParams copied from the nearest retained MCS.
    """
    if not retained_params:
        raise ValueError("nearest_mcs_transfer called with no retained MCS params.")
    if target_mcs in retained_params:
        return retained_params[target_mcs]

    retained_mcs = sorted(retained_params.keys())
    distances = [(abs(target_mcs - m), m) for m in retained_mcs]
    # sorted() is stable and (abs_diff, mcs) tuples break distance ties by
    # the smaller mcs value automatically, matching "select the lower MCS".
    _, nearest = min(distances)
    return retained_params[nearest]


def _to_psi(params: ARParams) -> np.ndarray:
    """Map ARParams -> interpolation-space vector
    Psi = [mu, atanh(kappa_1), ..., atanh(kappa_p), log(sigma)]
    (Sec. 2)."""
    kappa = ar_to_pacf(params.phi)
    kappa = np.clip(kappa, -1.0 + 1e-9, 1.0 - 1e-9)
    return np.concatenate([[params.mu], np.arctanh(kappa), [np.log(params.sigma)]])


def _from_psi(psi: np.ndarray) -> ARParams:
    """Inverse of _to_psi: recover ARParams from an interpolated Psi
    vector (Sec. 2)."""
    mu = psi[0]
    kappa = np.tanh(psi[1:-1])
    sigma = np.exp(psi[-1])
    phi = pacf_to_ar(kappa)
    return ARParams.from_mu_phi_sigma(mu, phi, sigma)


def piecewise_linear_interpolate(target_mcs: int, retained_params: Dict[int, ARParams]) -> ARParams:
    """Baseline 2: piecewise-linear interpolation between the nearest
    bracketing retained MCS values, performed in the transformed
    (mu, atanh(kappa), log(sigma)) space (Sec. 4). Falls back to
    nearest-boundary transfer outside the retained MCS range -- no
    extrapolation (Sec. 4, Boundary Handling).

    Args:
        target_mcs: excluded MCS to approximate.
        retained_params: {mcs: ARParams} for retained MCS at this (slice, SNR).

    Returns:
        ARParams recovered from the interpolated (or boundary-transferred)
        Psi vector.
    """
    if not retained_params:
        raise ValueError("piecewise_linear_interpolate called with no retained MCS params.")
    if target_mcs in retained_params:
        return retained_params[target_mcs]

    retained_mcs = sorted(retained_params.keys())

    if target_mcs < retained_mcs[0]:
        return retained_params[retained_mcs[0]]
    if target_mcs > retained_mcs[-1]:
        return retained_params[retained_mcs[-1]]

    # Find bracketing m_L < target_mcs < m_R among retained MCS.
    m_L = max(m for m in retained_mcs if m < target_mcs)
    m_R = min(m for m in retained_mcs if m > target_mcs)

    w = (target_mcs - m_L) / (m_R - m_L)

    psi_L = _to_psi(retained_params[m_L])
    psi_R = _to_psi(retained_params[m_R])
    psi_hat = (1.0 - w) * psi_L + w * psi_R

    return _from_psi(psi_hat)


def build_baseline_params_for_all_mcs(
    retained_params: Dict[int, ARParams],
    all_mcs: List[int],
    method: str,
) -> Dict[int, ARParams]:
    """Convenience wrapper: apply a baseline transfer rule to every MCS in
    all_mcs, reusing retained_params directly where target_mcs is retained.

    Args:
        retained_params: {mcs: ARParams} calibrated from retained MCS only.
        all_mcs: full MCS set to produce params for (typically 0-9).
        method: 'nearest' or 'interpolate'.

    Returns:
        {mcs: ARParams} for every mcs in all_mcs.
    """
    if method == 'nearest':
        transfer_fn = nearest_mcs_transfer
    elif method == 'interpolate':
        transfer_fn = piecewise_linear_interpolate
    else:
        raise ValueError(f"Unknown method '{method}'. Expected 'nearest' or 'interpolate'.")

    return {mcs: transfer_fn(mcs, retained_params) for mcs in all_mcs}


CONFIG_AXES = ('channel_model_id', 'BW', 'N_t', 'N_r', 'N_ss', 'MCS')


def _mcs_waterfall_rank(mcs: int, all_mcs: List[int]) -> int:
    """Rank of mcs within the sorted set of all MCS values (0-9), used as
    the MCS axis's distance unit in config_tuple_distance so MCS
    contributes on the same normalized footing as the other axes. Since
    MCS indices are already contiguous integers 0-9 in this dataset, rank
    coincides with the raw index, but this keeps the convention explicit
    and consistent with snr_rank's rank-not-raw-value philosophy.
    """
    return sorted(all_mcs).index(mcs)


def config_tuple_distance(
    target: Dict[str, float],
    candidate: Dict[str, float],
    axis_ranges: Dict[str, float],
    all_mcs: List[int],
) -> float:
    """Distance between two full (channel_model_id, MCS, N_t, N_r, N_ss,
    BW) configuration tuples, used by nearest_tuple_transfer to find the
    retained tuple closest to an excluded one.

    Every axis is normalized to [0, 1] by its observed range in the
    dataset (so BW's spread in MHz and N_t's spread in antenna count
    contribute comparably) and weighted equally, so CH, BW, N_t, N_r,
    N_ss, and MCS are all "equally important" axes of the tuple, per the
    experimental design (self_review discussion on generalizing the
    Sec. V-D baseline from MCS-only to full-tuple exclusion). The
    channel model is categorical (no natural ordering), so its
    contribution is a fixed 0/1 mismatch penalty rather than a normalized
    numeric difference. MCS uses its waterfall rank as the numeric value
    (see _mcs_waterfall_rank), matching the rank-based (not raw-value)
    convention already used for SNR elsewhere in this module.

    SNR is deliberately NOT one of these axes: SNR is a within-tuple
    measurement axis (every tuple in this dataset carries all 10 SNR
    points), not a configuration axis that gets excluded, so it plays no
    role in choosing the nearest tuple. Once the nearest tuple is chosen,
    its SNR points are matched to the target's by waterfall rank via
    snr_rank/params_at_matching_rank, exactly as in the MCS-only
    baselines.

    Args:
        target: dict with keys channel_model_id, BW, N_t, N_r, N_ss, MCS
            for the excluded tuple.
        candidate: same keys, for a retained tuple.
        axis_ranges: {axis: max-min observed range} for BW, N_t, N_r, N_ss
            (channel_model_id and MCS are handled separately, see above).
        all_mcs: full sorted MCS set (typically 0-9), used to compute
            waterfall rank for the MCS axis.

    Returns:
        Non-negative scalar distance; 0.0 iff target == candidate.
    """
    dist = 0.0

    dist += 1.0 if target['channel_model_id'] != candidate['channel_model_id'] else 0.0

    for axis in ('BW', 'N_t', 'N_r', 'N_ss'):
        axis_range = axis_ranges[axis]
        if axis_range == 0:
            continue
        dist += abs(target[axis] - candidate[axis]) / axis_range

    mcs_range = len(all_mcs) - 1
    if mcs_range > 0:
        target_rank = _mcs_waterfall_rank(target['MCS'], all_mcs)
        candidate_rank = _mcs_waterfall_rank(candidate['MCS'], all_mcs)
        dist += abs(target_rank - candidate_rank) / mcs_range

    return dist


def nearest_tuple_transfer(
    target_tuple: Dict[str, float],
    retained_tuples: List[Dict[str, float]],
    axis_ranges: Dict[str, float],
    all_mcs: List[int],
) -> Dict[str, float]:
    """Baseline (full-tuple generalization of nearest_mcs_transfer): find
    the retained (channel_model_id, MCS, N_t, N_r, N_ss, BW) tuple closest
    to target_tuple under config_tuple_distance.

    This is the Section V-D baseline used once exclusion operates over the
    full joint configuration space rather than MCS alone within a fixed
    slice: nearest_mcs_transfer only ever searched within one fixed
    (CH, N_t, N_r, BW, N_ss) slice for the nearest retained MCS.
    nearest_tuple_transfer instead searches over every retained tuple in
    the whole dataset, so it remains well-defined when the excluded
    tuple's entire slice (not just its MCS) is unseen. Ties are broken by
    the order retained_tuples are given in (stable min()), mirroring
    nearest_mcs_transfer's deterministic tie-breaking.

    Args:
        target_tuple: excluded tuple to approximate, keys
            channel_model_id, BW, N_t, N_r, N_ss, MCS.
        retained_tuples: candidate tuples (same keys) to search over.
        axis_ranges: {axis: max-min observed range} for BW, N_t, N_r, N_ss,
            passed through to config_tuple_distance.
        all_mcs: full sorted MCS set (typically 0-9).

    Returns:
        The retained tuple (dict) with minimum distance to target_tuple.
    """
    if not retained_tuples:
        raise ValueError("nearest_tuple_transfer called with no retained tuples.")

    distances = [
        (config_tuple_distance(target_tuple, candidate, axis_ranges, all_mcs), i, candidate)
        for i, candidate in enumerate(retained_tuples)
    ]
    _, _, nearest = min(distances, key=lambda d: (d[0], d[1]))
    return nearest


def snr_rank(target_snr: int, mcs_snr_grid) -> int:
    """Waterfall-rank index (0 = lowest SNR / most reliable operating
    point, ... 9 = highest SNR / edge-of-failure point) of target_snr
    within its own MCS's sorted SNR grid.

    Each MCS is calibrated at 10 SNR points chosen to sample that MCS's
    own PER-vs-SNR waterfall curve (Table II), so raw SNR (dB) is not
    comparable across MCS: MCS0's waterfall sits around -2..25 dB while
    MCS9's sits around 34..61 dB, a fixed offset reflecting how much SNR
    that MCS needs for a given PER. Rank position on each MCS's own
    waterfall, not absolute dB, is the comparable quantity across MCS.
    """
    sorted_snrs = sorted(mcs_snr_grid)
    return sorted_snrs.index(target_snr)


def params_at_matching_rank(
    calibrated_by_mcs_snr: Dict[int, Dict[int, ARParams]],
    target_rank: int,
) -> Dict[int, ARParams]:
    """Reduce {mcs: {snr: ARParams}} to {mcs: ARParams} by taking, for
    each retained MCS independently, the params at its own SNR whose
    waterfall rank matches target_rank.

    "Using retained MCS m's params for target MCS's i-th waterfall SNR"
    means m's own i-th waterfall SNR params -- not m's nearest SNR in raw
    dB, which would mix operating points from different parts of the PER
    curve since each MCS's waterfall is offset in dB from every other
    MCS's. This is the piece that makes MCS-transfer well-defined when
    retained and excluded MCS have disjoint SNR grids (every MCS in this
    dataset has exactly 10 SNR points, one per rank 0-9, so target_rank
    always has a match for every retained MCS).

    Args:
        calibrated_by_mcs_snr: {mcs: {snr: ARParams}} from
            calibrate_retained_mcs.
        target_rank: waterfall-rank index (0-9) of the test cell being
            approximated, computed on the target MCS's own SNR grid via
            snr_rank.

    Returns:
        {mcs: ARParams}, one entry per retained MCS, each using that MCS's
        params at its own rank-target_rank SNR.
    """
    result = {}
    for mcs, mcs_dict in calibrated_by_mcs_snr.items():
        if not mcs_dict:
            continue
        sorted_snrs = sorted(mcs_dict.keys())
        if target_rank >= len(sorted_snrs):
            continue
        result[mcs] = mcs_dict[sorted_snrs[target_rank]]
    return result
