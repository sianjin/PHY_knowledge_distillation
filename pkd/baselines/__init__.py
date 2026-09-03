"""Naive sparse-MCS baselines for EESM-log-AR: nearest-MCS transfer and
piecewise-linear MCS interpolation, evaluated against the same test set
and metrics used for PKD's Fig. 13 configuration-scalability experiment.

See self_review/EESM_LOG_AR_SPARSE_MCS_BASELINES.md for the full design.
"""
from .calibration import calibrate_ar_params, ARParams
from .pacf_transform import ar_to_pacf, pacf_to_ar
from .transfer import (
    nearest_mcs_transfer,
    piecewise_linear_interpolate,
    nearest_tuple_transfer,
    config_tuple_distance,
)
from .generate import generate_ar_sequence
from .evaluate_baselines import (
    run_baseline_exclusion_analysis,
    run_baseline_tuple_exclusion_analysis,
    calibrate_all_retained_tuples,
    evaluate_nearest_tuple_baseline,
)

__all__ = [
    'calibrate_ar_params',
    'ARParams',
    'ar_to_pacf',
    'pacf_to_ar',
    'nearest_mcs_transfer',
    'piecewise_linear_interpolate',
    'nearest_tuple_transfer',
    'config_tuple_distance',
    'generate_ar_sequence',
    'run_baseline_exclusion_analysis',
    'run_baseline_tuple_exclusion_analysis',
    'calibrate_all_retained_tuples',
    'evaluate_nearest_tuple_baseline',
]
