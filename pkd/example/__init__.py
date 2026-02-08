"""PKD Example Package - Refactored for better organization."""

from .sgn_cdf import sgn_pdf, sgn_cdf
from .data_loader import load_real_data
from .utils import (
    compute_acf, compute_psd, ljung_box_test,
    get_config_signature, infer_most_common_slice, filter_by_slice, format_slice_label
)
from .evaluation import (
    evaluate_marginal_distribution,
    evaluate_temporal_dependence,
    evaluate_innovation_structure
)
from .plotting import (
    evaluate_test_set,
    generate_figure1_per_mcs_metrics,
    generate_figure2_quantile_error,
    generate_figure3_ccdf_error
)
from .main import (
    example_training,
    example_evaluation,
    example_test_evaluation
)

__all__ = [
    # SGN CDF
    'sgn_pdf',
    'sgn_cdf',
    # Data loading
    'load_real_data',
    # Utilities
    'compute_acf',
    'compute_psd',
    'ljung_box_test',
    'get_config_signature',
    'infer_most_common_slice',
    'filter_by_slice',
    'format_slice_label',
    # Evaluation
    'evaluate_marginal_distribution',
    'evaluate_temporal_dependence',
    'evaluate_innovation_structure',
    'evaluate_test_set',
    # Plotting
    'generate_figure1_per_mcs_metrics',
    'generate_figure2_quantile_error',
    'generate_figure3_ccdf_error',
    # Main entry points
    'example_training',
    'example_evaluation',
    'example_test_evaluation',
]
