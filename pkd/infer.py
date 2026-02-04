"""Inference engine with time-skipping via configuration caching."""
import torch
import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CachedParams:
    """Cached process parameters for time-skipping."""
    m: float
    phi: np.ndarray  # (p,)
    c: float
    innov_params: np.ndarray  # innovation parameters (sigma for gaussian, psi for flow)

    @staticmethod
    def from_torch(params_dict):
        """Convert torch tensors to numpy for caching."""
        # Handle innovation parameters (can be scalar sigma or vector psi)
        innov = params_dict['innov_params']
        if innov.ndim == 0:
            innov_np = innov.item()  # Scalar sigma
        else:
            innov_np = innov.cpu().numpy()  # Vector psi

        return CachedParams(
            m=params_dict['m'].item(),
            phi=params_dict['phi'].cpu().numpy(),
            c=params_dict['c'].item(),
            innov_params=innov_np
        )


class PKDInference:
    """
    PKD inference engine with configuration-driven parameter caching.

    Implements Algorithm 2 from paper with time-skipping mechanism.
    """

    def __init__(self,
                 model,
                 per_lut,
                 ar_order=10,
                 burn_in=50,
                 snr_quant=0.1,
                 device='cuda'):
        """
        Args:
            model: trained PKDModel
            per_lut: AWGNPERLookup instance
            ar_order: AR order
            burn_in: burn-in period for cold start
            snr_quant: SNR quantization for caching (dB)
            device: computation device
        """
        self.model = model.to(device)
        self.model.eval()
        self.per_lut = per_lut
        self.ar_order = ar_order
        self.burn_in = burn_in
        self.snr_quant = snr_quant
        self.device = device

        # State
        self.state_buffer = deque(maxlen=ar_order)
        self.cache = {}
        self.current_key = None
        self.current_params = None

    def _make_config_key(self, config_dict) -> Tuple:
        """Create hashable configuration key."""
        # Extract and quantize SNR
        snr_bar = config_dict['SNR_bar'].item() if torch.is_tensor(config_dict['SNR_bar']) else config_dict['SNR_bar']
        snr_key = np.round(snr_bar / self.snr_quant) * self.snr_quant

        # Create key from all config components
        key = (
            config_dict['channel_model_id'].item() if torch.is_tensor(config_dict['channel_model_id']) else config_dict['channel_model_id'],
            config_dict['N_t'].item() if torch.is_tensor(config_dict['N_t']) else config_dict['N_t'],
            config_dict['N_r'].item() if torch.is_tensor(config_dict['N_r']) else config_dict['N_r'],
            config_dict['BW'].item() if torch.is_tensor(config_dict['BW']) else config_dict['BW'],
            snr_key,
            config_dict['MCS'].item() if torch.is_tensor(config_dict['MCS']) else config_dict['MCS'],
            config_dict['N_ss'].item() if torch.is_tensor(config_dict['N_ss']) else config_dict['N_ss'],
        )

        return key

    def _evaluate_network(self, config_dict):
        """Evaluate network to get process parameters."""

        # Convert config to batch format
        config_batch = {}
        for key, val in config_dict.items():
            if not torch.is_tensor(val):
                val = torch.tensor([val])
            elif val.ndim == 0:
                val = val.unsqueeze(0)
            config_batch[key] = val.to(self.device)

        with torch.no_grad():
            params = self.model(config_batch, X_hist=None)

        # Extract single-sample parameters
        params_np = CachedParams.from_torch({k: v[0] for k, v in params.items()})
        
        # Stability sanity check: verify AR polynomial roots
        def ar_is_stable_companion(phi):
            phi = np.asarray(phi).reshape(-1)
            p = len(phi)
            F = np.zeros((p, p), dtype=float)
            F[0, :] = phi
            F[1:, :-1] = np.eye(p-1)
            rho = np.max(np.abs(np.linalg.eigvals(F)))
            return rho < 1.0, rho

        stable, rho = ar_is_stable_companion(params_np.phi)
        if not stable:
            import warnings
            warnings.warn(
                f"AR recursion unstable: spectral radius={rho:.6f} >= 1.0"
            )

        return params_np

    def _get_params(self, config_dict) -> CachedParams:
        """Get parameters with caching (time-skipping mechanism)."""
        key = self._make_config_key(config_dict)

        # Check if key changed
        if key != self.current_key:
            # Check cache
            if key in self.cache:
                self.current_params = self.cache[key]
            else:
                # Evaluate network
                self.current_params = self._evaluate_network(config_dict)
                self.cache[key] = self.current_params

            self.current_key = key

        return self.current_params

    def cold_start(self, initial_config):
        """Initialize state with burn-in."""
        # Get initial mean
        params = self._get_params(initial_config)
        m_init = params.m

        # Initialize state buffer
        self.state_buffer.clear()
        for _ in range(self.ar_order):
            self.state_buffer.append(m_init)

        # Burn-in
        for _ in range(self.burn_in):
            _ = self.step(initial_config, return_per=False)

    def step(self, config_dict, return_per=True):
        """
        Single inference step.

        Args:
            config_dict: configuration at time t
            return_per: if True, also compute PER

        Returns:
            gamma_eff: effective SINR (linear scale)
            per: packet error rate (if return_per=True)
            error_event: packet error (if return_per=True)
        """
        # Get parameters (cached if config unchanged)
        params = self._get_params(config_dict)

        # Current state as numpy array [X_{t-1}, ..., X_{t-p}]
        # CRITICAL FIX: deque stores [X_{t-p}, ..., X_{t-1}] (oldest to newest)
        # Training expects [X_{t-1}, ..., X_{t-p}] (newest to oldest)
        # So we must reverse!
        state_array = np.array(list(self.state_buffer))[::-1]

        # Compute AR mean
        mu = params.c + np.dot(params.phi, state_array)

        # Sample innovation (works for both gaussian and flow)
        eps = self.model.innovation.sample_numpy(params.innov_params)

        # Generate X_t
        X_t = mu + eps

        # Optional emergency guard (very rare): clamp only for numerical extremes
        # This should almost never trigger if model is stable (PACF constrained + reasonable sigma)
        if X_t > 50.0:
            X_t = 50.0
        elif X_t < -50.0:
            X_t = -50.0

        # Fail-fast check for numerical issues
        if not np.isfinite(X_t):
            raise RuntimeError(f"X_t became non-finite: {X_t} (mu={mu:.4f}, eps={eps:.4f})")

        # Convert to linear domain
        gamma_eff = np.exp(X_t)

        # Update state
        self.state_buffer.append(X_t)

        if not return_per:
            return gamma_eff

        # Compute PER
        mcs = config_dict['MCS'].item() if torch.is_tensor(config_dict['MCS']) else config_dict['MCS']
        per = self.per_lut.lookup(gamma_eff, mcs)

        # Sample error event
        error_event = np.random.rand() < per

        return gamma_eff, per, error_event

    def run_sequence(self, config_trajectory, return_details=True):
        """
        Run inference over a configuration trajectory.

        Args:
            config_trajectory: list of config dicts (one per packet)
            return_details: if True, return full details

        Returns:
            results: dict with gamma_eff, per, errors arrays
        """
        # Cold start with first config
        self.cold_start(config_trajectory[0])

        # Run sequence
        gamma_eff_list = []
        per_list = []
        error_list = []

        for config in config_trajectory:
            gamma_eff, per, error = self.step(config, return_per=True)
            gamma_eff_list.append(gamma_eff)
            per_list.append(per)
            error_list.append(error)

        results = {
            'gamma_eff': np.array(gamma_eff_list),
            'per': np.array(per_list),
            'errors': np.array(error_list)
        }

        return results

    def run_with_time_skipping(self, config_windows):
        """
        Run inference with explicit time-skipping windows.

        Args:
            config_windows: list of (config, num_packets) tuples
                           Each tuple represents a window with constant config

        Returns:
            results: dict with gamma_eff, per, errors arrays
        """
        # Cold start
        first_config, _ = config_windows[0]
        self.cold_start(first_config)

        gamma_eff_list = []
        per_list = []
        error_list = []

        for config, num_packets in config_windows:
            # Parameters evaluated once per window
            params = self._get_params(config)
            mcs = config['MCS'].item() if torch.is_tensor(config['MCS']) else config['MCS']

            for _ in range(num_packets):
                # AR evolution with cached params (cheap O(p) per packet)
                # CRITICAL FIX: Reverse state array to match training order
                state_array = np.array(list(self.state_buffer))[::-1]
                mu = params.c + np.dot(params.phi, state_array)

                # Sample innovation (works for both gaussian and flow)
                eps = self.model.innovation.sample_numpy(params.innov_params)

                X_t = mu + eps

                # Optional emergency guard (very rare): clamp only for numerical extremes
                if X_t > 50.0:
                    X_t = 50.0
                elif X_t < -50.0:
                    X_t = -50.0

                if not np.isfinite(X_t):
                    raise RuntimeError(f"X_t became non-finite: {X_t}")

                gamma_eff = np.exp(X_t)

                self.state_buffer.append(X_t)

                # PER and error
                per = self.per_lut.lookup(gamma_eff, mcs)
                error = np.random.rand() < per

                gamma_eff_list.append(gamma_eff)
                per_list.append(per)
                error_list.append(error)

        return {
            'gamma_eff': np.array(gamma_eff_list),
            'per': np.array(per_list),
            'errors': np.array(error_list)
        }
