"""Classical EESM-log-AR calibration: fit AR(p) + Gaussian innovation
parameters Phi(m, s) = {mu, phi_1:p, sigma} from teacher sequences at a
fixed (configuration slice, MCS, SNR) cell.

Mirrors the ML estimation in the EESM-log-AR paper (Sec. 3.1): the log
domain effective SINR X_t = c + sum_i phi_i X_{t-i} + eps_t, eps_t ~
N(0, sigma^2). Under Gaussian innovations, conditional ML is equivalent to
conditional least squares, so we fit via OLS on the same lag structure
PKD uses (p=10, order [X_{t-1}, ..., X_{t-p}]).

Calibration must only ever be given training-split sequences at retained
MCS values -- see self_review/EESM_LOG_AR_SPARSE_MCS_BASELINES.md Rule 2
and the caller in evaluate_baselines.py for how that is enforced.
"""
from dataclasses import dataclass
from typing import List

import numpy as np

from .pacf_transform import ar_to_pacf


@dataclass
class ARParams:
    """Calibrated (or transferred/interpolated) EESM-log-AR parameters
    Phi(m, s) = {mu, phi_1:p, sigma} at a fixed AR order p."""
    mu: float          # stationary mean E[X_t]
    phi: np.ndarray    # (p,) AR coefficients
    sigma: float       # innovation standard deviation
    c: float           # intercept, c = (1 - sum(phi)) * mu

    @staticmethod
    def from_mu_phi_sigma(mu: float, phi: np.ndarray, sigma: float) -> "ARParams":
        c = (1.0 - np.sum(phi)) * mu
        return ARParams(mu=mu, phi=np.asarray(phi, dtype=np.float64), sigma=sigma, c=c)


def calibrate_ar_params(sequences: List[np.ndarray], ar_order: int = 10) -> ARParams:
    """Fit AR(p) + Gaussian innovation parameters from a pool of teacher
    sequences sharing the same (configuration slice, MCS, SNR).

    All sequences are pooled into a single design matrix so the estimate
    uses every available training sample at this (MCS, SNR) cell, rather
    than fitting per-sequence and averaging.

    Args:
        sequences: list of 1-D log-domain effective SINR sequences
            (X_t = ln(gamma_eff,t)), all from the same (slice, MCS, SNR).
        ar_order: AR order p (must match PKD's ar_order, default 10).

    Returns:
        ARParams with mu, phi (p,), sigma, c.

    Raises:
        ValueError: if there is not enough data to fit an AR(p) model, or
            the fit is not stable (unlikely for real teacher data, but
            checked so a degenerate calibration cannot silently propagate
            into the interpolation baseline's PACF transform).
    """
    if len(sequences) == 0:
        raise ValueError("calibrate_ar_params called with no sequences.")

    X_design = []
    y = []
    for seq in sequences:
        seq = np.asarray(seq, dtype=np.float64)
        T = len(seq)
        if T <= ar_order:
            continue
        for t in range(ar_order, T):
            # [X_{t-1}, ..., X_{t-p}], matching PKD's training order
            # (pkd/train.py: np.flip(X_seq[t-ar_order:t], axis=0))
            X_design.append(seq[t - ar_order:t][::-1])
            y.append(seq[t])

    if len(y) < ar_order + 2:
        raise ValueError(
            f"Not enough samples to calibrate AR({ar_order}): got {len(y)} "
            f"regression rows from {len(sequences)} sequences."
        )

    X_design = np.asarray(X_design, dtype=np.float64)  # (N, p)
    y = np.asarray(y, dtype=np.float64)                # (N,)

    X_with_intercept = np.column_stack([np.ones(len(X_design)), X_design])
    beta, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)

    c = beta[0]
    phi = beta[1:]

    residuals = y - X_with_intercept @ beta
    sigma = np.std(residuals, ddof=ar_order + 1)

    phi_sum = np.sum(phi)
    if abs(1.0 - phi_sum) < 1e-8:
        raise ValueError(
            f"AR({ar_order}) fit has sum(phi)={phi_sum:.6f} ~= 1 "
            f"(near-unit-root); cannot recover a finite stationary mean."
        )
    mu = c / (1.0 - phi_sum)

    # Stability check mirrors pkd/infer.py's companion-matrix guard, and
    # also guarantees ar_to_pacf's step-down recursion won't diverge.
    _assert_stable(phi, ar_order)

    return ARParams.from_mu_phi_sigma(mu, phi, sigma)


def _assert_stable(phi: np.ndarray, ar_order: int) -> None:
    F = np.zeros((ar_order, ar_order))
    F[0, :] = phi
    F[1:, :-1] = np.eye(ar_order - 1)
    rho = np.max(np.abs(np.linalg.eigvals(F)))
    if rho >= 1.0:
        raise ValueError(
            f"Calibrated AR({ar_order}) is unstable (spectral radius="
            f"{rho:.6f} >= 1.0). This should not happen for a converged "
            f"OLS fit on real teacher data; check the input sequences."
        )
