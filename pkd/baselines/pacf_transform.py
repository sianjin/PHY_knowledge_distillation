"""AR(p) <-> PACF (reflection coefficient) conversion in numpy.

pkd/model/ld.py already implements the PACF -> AR direction (Levinson-Durbin
recursion) as a differentiable torch layer for the neural model. The
interpolation baseline additionally needs the inverse direction, AR -> PACF
(the "step-down" recursion), to move fitted AR coefficients into the
interpolation-safe space described in Section 2 of
self_review/EESM_LOG_AR_SPARSE_MCS_BASELINES.md.
"""
import numpy as np


def pacf_to_ar(kappa: np.ndarray) -> np.ndarray:
    """Convert PACF coefficients to AR coefficients via Levinson-Durbin.

    Numpy port of pkd.model.ld.levinson_durbin_from_pacf, kept independent
    so the classical baselines have no torch dependency.

    Args:
        kappa: (p,) reflection coefficients, |kappa_i| < 1.

    Returns:
        phi: (p,) AR coefficients, guaranteed stable if |kappa_i| < 1.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    p = kappa.shape[-1]

    phi = np.zeros(p, dtype=np.float64)
    for k in range(p):
        kappa_k = kappa[k]
        if k == 0:
            phi_new = np.array([kappa_k])
        else:
            prev = phi[:k]
            prev_rev = prev[::-1]
            phi_new = np.concatenate([prev - kappa_k * prev_rev, [kappa_k]])
        phi[:k + 1] = phi_new

    return phi


def ar_to_pacf(phi: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Convert AR coefficients to PACF coefficients via the Levinson-Durbin
    step-down recursion (inverse of pacf_to_ar).

    Args:
        phi: (p,) AR coefficients of a stable AR(p) process.
        eps: numerical clamp to keep intermediate reflection coefficients
            strictly inside (-1, 1) even under mild estimation noise.

    Returns:
        kappa: (p,) PACF (reflection) coefficients.

    Raises:
        ValueError: if the step-down recursion encounters |kappa_k| >= 1,
            which indicates the input phi does not correspond to a stable
            AR(p) process (should not happen for a converged ML/OLS fit,
            but calibration noise on short sequences can occasionally
            produce a marginally unstable estimate).
    """
    phi = np.asarray(phi, dtype=np.float64).copy()
    p = phi.shape[-1]

    kappa = np.zeros(p, dtype=np.float64)
    phi_k = phi.copy()

    for k in range(p - 1, -1, -1):
        kappa_k = phi_k[k]
        if abs(kappa_k) >= 1.0:
            raise ValueError(
                f"Unstable AR fit encountered during AR->PACF step-down "
                f"(|kappa_{k}|={abs(kappa_k):.6f} >= 1.0). "
                f"The calibrated AR coefficients do not correspond to a "
                f"stationary process."
            )
        kappa[k] = kappa_k

        if k == 0:
            break

        prev = phi_k[:k]
        prev_rev = prev[::-1]
        denom = 1.0 - kappa_k ** 2
        phi_k = (prev + kappa_k * prev_rev) / denom

    return np.clip(kappa, -1.0 + eps, 1.0 - eps)
