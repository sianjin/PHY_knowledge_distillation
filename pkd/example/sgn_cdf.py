"""SGN CDF implementation for PIT computation.

CRITICAL NOTE ON PIT (Probability Integral Transform):

PIT is used to assess conditional calibration of the student model.
If the model assumes innovations are z_t ~ SGN(0, 1, beta_t, lambda_t),
then the PIT values u_t = F_SGN(z_t; beta_t, lambda_t) should be Uniform(0,1).

WRONG: u_t = Phi(z_t)  [Gaussian CDF]
  - This is ONLY correct if innovations are Gaussian N(0,1)
  - Using Gaussian CDF with SGN innovations will cause PIT to fail uniformity tests
    even when the SGN model is perfectly calibrated!

CORRECT: u_t = F_SGN(z_t; beta_t, lambda_t)  [SGN CDF]
  - Must use the SAME distribution assumed by the model
  - For SGN innovations, this requires numerical computation of SGN CDF

Higher PIT pass rate is BETTER (pass = do NOT reject uniformity hypothesis)
"""

import numpy as np
from scipy import stats
from scipy.special import gamma as gamma_func
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d


def sgn_pdf(z, beta, lam):
    """
    Compute the standardized SGN PDF: f(z; beta, lambda) where SGN(0, 1, beta, lambda).

    The SGN distribution is defined as:
        f(z) = (2 / Gamma(1/beta)) * beta/2 * exp(-|z|^beta) * Phi(sqrt(2) * lambda * z)

    where Phi is the standard normal CDF.

    Args:
        z: Input values (can be array)
        beta: Shape parameter (beta > 0, beta=2 is Gaussian-like)
        lam: Skewness parameter (lambda=0 is symmetric)

    Returns:
        pdf_values: PDF evaluated at z
    """
    z = np.atleast_1d(z)

    # Generalized normal base PDF: phi(z; beta)
    gamma_val = gamma_func(1.0 / beta)
    base_pdf = (beta / (2.0 * gamma_val)) * np.exp(-np.abs(z) ** beta)

    # Standard normal CDF for skewing
    skew_term = stats.norm.cdf(np.sqrt(2) * lam * z)

    # SGN PDF
    pdf_values = 2.0 * base_pdf * skew_term

    return pdf_values


def sgn_cdf(z, beta, lam, z_min=-12, z_max=12, n_points=2000):
    """
    Compute the standardized SGN CDF: F(z; beta, lambda) for SGN(0, 1, beta, lambda).

    Since there is no closed-form CDF for SGN, we compute it numerically:
    1. Create a fine grid over [z_min, z_max]
    2. Evaluate the PDF on the grid
    3. Integrate using trapezoidal rule to get CDF
    4. Interpolate to evaluate at arbitrary z values

    Args:
        z: Input values (scalar or array) where we want to evaluate CDF
        beta: Shape parameter (beta > 0)
        lam: Skewness parameter
        z_min: Lower bound for integration grid
        z_max: Upper bound for integration grid
        n_points: Number of points in integration grid

    Returns:
        cdf_values: CDF evaluated at z (in [0, 1])

    Notes:
        - For PIT computation, this must use the SAME distribution assumed by the model
        - Using Gaussian CDF (stats.norm.cdf) is WRONG when innovations are SGN
        - This function is cached per (beta, lambda) pair for efficiency
    """
    z = np.atleast_1d(z)
    is_scalar = (z.ndim == 0) or (len(z) == 1)

    # Create integration grid
    z_grid = np.linspace(z_min, z_max, n_points)

    # Evaluate PDF on grid
    pdf_grid = sgn_pdf(z_grid, beta, lam)

    # Integrate to get unnormalized CDF
    cdf_grid = np.zeros_like(z_grid)
    for i in range(1, len(z_grid)):
        cdf_grid[i] = cdf_grid[i-1] + trapezoid(pdf_grid[i-1:i+1], z_grid[i-1:i+1])

    # Normalize to ensure CDF(+inf) = 1.0
    total_mass = cdf_grid[-1]
    if total_mass > 0:
        cdf_grid /= total_mass
    else:
        # Fallback: use uniform CDF if integration fails
        cdf_grid = np.linspace(0, 1, len(z_grid))

    # Clamp to [0, 1]
    cdf_grid = np.clip(cdf_grid, 0, 1)

    # Interpolate to evaluate at requested z values
    cdf_interp = interp1d(z_grid, cdf_grid, kind='linear',
                          bounds_error=False, fill_value=(0.0, 1.0))
    cdf_values = cdf_interp(z)

    # Ensure output is in [0, 1]
    cdf_values = np.clip(cdf_values, 0, 1)

    if is_scalar:
        return float(cdf_values[0])
    return cdf_values
