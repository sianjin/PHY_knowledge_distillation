"""Generate synthetic effective-SINR sequences from calibrated/transferred/
interpolated EESM-log-AR parameters.

Mirrors pkd/infer.py's PKDInference free-running generation exactly (same
state ordering, same mean-initialized burn-in, same dB conversion) so any
KS/ACF differences between PKD and the baselines reflect the parameter
values, not a different generation procedure.
"""
import numpy as np

from .calibration import ARParams


def generate_ar_sequence(
    params: ARParams,
    length: int,
    ar_order: int = 10,
    burn_in: int = 50,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Free-running AR(p) + Gaussian-innovation sample path in the natural
    log domain, X_t = ln(gamma_eff,t).

    Args:
        params: ARParams (mu, phi, sigma, c) to simulate from.
        length: number of output samples (post burn-in).
        ar_order: AR order p (must match params.phi length).
        burn_in: number of discarded warm-up steps, mirroring
            PKDInference.cold_start's burn_in=50 default.
        rng: numpy Generator for reproducibility; a fresh default_rng() is
            used if not provided.

    Returns:
        X: (length,) natural-log-domain sequence.
    """
    if rng is None:
        rng = np.random.default_rng()

    phi = params.phi
    if len(phi) != ar_order:
        raise ValueError(f"params.phi has length {len(phi)}, expected ar_order={ar_order}.")

    # State buffer holds [X_{t-p}, ..., X_{t-1}] (oldest to newest), matching
    # pkd.infer.PKDInference's deque convention, initialized to the mean.
    state = [params.mu] * ar_order

    def step():
        # Reverse to [X_{t-1}, ..., X_{t-p}] for the dot product with phi,
        # matching PKDInference.step.
        state_array = np.asarray(state[::-1])
        mu_t = params.c + np.dot(phi, state_array)
        eps = rng.normal(0.0, params.sigma)
        X_t = mu_t + eps

        # Same emergency clamp as PKDInference.step.
        X_t = np.clip(X_t, -50.0, 50.0)

        state.pop(0)
        state.append(X_t)
        return X_t

    for _ in range(burn_in):
        step()

    X = np.empty(length, dtype=np.float64)
    for i in range(length):
        X[i] = step()

    return X
