"""Complete PKD student model."""
import torch
import torch.nn as nn
from .encoder import CompositionalEncoder
from .heads import MeanHead, PACFHead
from .ld import PACFToAR
from .innovation import create_innovation_model


class PKDModel(nn.Module):
    """
    PHY Knowledge Distillation student model.

    Implements conditional time-varying log-AR(p) process with:
    - Stable AR dynamics via PACF parameterization
    - Flexible innovation distributions: SGN (default), Gaussian, or Flow
    - Compositional conditioning for generalization

    SGN (Skew Generalized Normal) innovations provide:
    - Configuration-adaptive variance, skewness, and tail behavior
    - Generalization of Gaussian (lambda=0, beta=2 case)
    - Better marginal calibration across diverse PHY conditions
    """

    def __init__(self,
                 num_channel_models: int,
                 num_mcs: int,
                 num_nss: int,
                 ar_order: int = 10,
                 hidden_dim: int = 128,
                 kappa_max: float = 0.95,
                 innovation_type: str = 'sgn',
                 **innovation_kwargs):
        """
        Args:
            num_channel_models: Number of channel models
            num_mcs: Number of MCS levels (0-indexed: 0 to num_mcs-1)
            num_nss: Number of spatial streams
            ar_order: AR order (p)
            hidden_dim: Hidden dimension for encoder
            kappa_max: Maximum PACF coefficient magnitude (0.90-0.98)
                      Lower = more stable, higher = more flexible
            innovation_type: 'sgn' (default), 'gaussian', or 'flow'
            **innovation_kwargs: Additional arguments for innovation model
                For sgn: min_sigma, min_beta, max_beta (defaults: 0.1, 0.5, 4.0)
                For gaussian: min_sigma (default: 0.1)
                For flow: num_flow_bins, flow_tail_bound, min_bin_size, min_derivative
        """
        super().__init__()

        self.ar_order = ar_order
        self.innovation_type = innovation_type

        # Conditioning encoder
        self.encoder = CompositionalEncoder(
            num_channel_models=num_channel_models,
            num_mcs=num_mcs,
            num_nss=num_nss,
            hidden_dim=hidden_dim
        )

        # Parameter heads
        self.mean_head = MeanHead(hidden_dim)
        self.pacf_head = PACFHead(hidden_dim, ar_order)

        # PACF to AR conversion with stability constraint
        self.pacf_to_ar = PACFToAR(kappa_max=kappa_max)

        # Innovation distribution model
        self.innovation = create_innovation_model(
            innovation_type=innovation_type,
            hidden_dim=hidden_dim,
            **innovation_kwargs
        )

        print(f"PKDModel initialized with {innovation_type} innovation")

    def encode_config(self, config_dict):
        """Encode configuration to conditioning representation."""
        return self.encoder(config_dict)

    def generate_params(self, h):
        """
        Generate stochastic process parameters from conditioning.

        Args:
            h: (batch, hidden_dim) conditioning representation

        Returns:
            dict with keys:
                - m: (batch,) conditional mean
                - u: (batch, p) unconstrained PACF
                - kappa: (batch, p) constrained PACF
                - phi: (batch, p) AR coefficients
                - c: (batch,) AR offset
                - innov_params: innovation parameters (sigma for gaussian, psi for flow)
        """
        # Generate raw parameters with explicit shape control
        m = self.mean_head(h)  # Already returns (batch,) from squeeze in head
        u = self.pacf_head(h)  # (batch, p)

        # Validate shapes
        assert m.ndim == 1, f"Mean should be (batch,), got {m.shape}"
        assert u.ndim == 2 and u.shape[-1] == self.ar_order, f"PACF should be (batch, {self.ar_order}), got {u.shape}"

        # Convert PACF to AR (guaranteed stable if |kappa| < kappa_max < 1)
        phi, kappa = self.pacf_to_ar(u)

        # Compute AR offset for mean consistency
        # c is chosen so that E[X_t] = m when process is stationary
        phi_sum = phi.sum(dim=-1)  # (batch,)
        c = (1.0 - phi_sum) * m  # (batch,)

        # Generate innovation parameters
        innov_params = self.innovation(h)

        return {
            'm': m,
            'u': u,
            'kappa': kappa,
            'phi': phi,
            'c': c,
            'innov_params': innov_params
        }

    def compute_ar_mean(self, c, phi, X_hist):
        """
        Compute AR conditional mean.

        Args:
            c: (batch,) AR offset
            phi: (batch, p) AR coefficients
            X_hist: (batch, p) history [X_{t-1}, ..., X_{t-p}]

        Returns:
            mu: (batch,) conditional mean
        """
        # mu = c + sum_i phi_i * X_{t-i}
        mu = c + (phi * X_hist).sum(dim=-1)
        return mu

    def compute_log_likelihood(self, X, X_hist, config_dict):
        """
        Compute conditional log-likelihood for training.

        Args:
            X: (batch,) target log-effective-SINR
            X_hist: (batch, p) history
            config_dict: configuration dictionary

        Returns:
            log_q: (batch,) log-likelihood
            info: dict with intermediate values for diagnostics
        """
        # Encode configuration
        h = self.encode_config(config_dict)

        # Generate parameters
        params = self.generate_params(h)

        # Compute AR mean
        mu = self.compute_ar_mean(params['c'], params['phi'], X_hist)

        # Compute innovation
        eps = X - mu

        # Compute log-likelihood using innovation model
        log_q = self.innovation.log_prob(eps, params['innov_params'])

        # Info for diagnostics
        info = {
            'mu': mu,
            'eps': eps,
            'kappa': params['kappa'],
            'phi': params['phi'],
            'm': params['m'],
            'innov_params': params['innov_params']
        }

        return log_q, info

    def sample_innovation(self, innov_params, num_samples=1):
        """
        Sample innovations from conditional distribution.

        Args:
            innov_params: innovation parameters (sigma for gaussian, psi for flow)
            num_samples: number of samples per parameter set

        Returns:
            eps: innovation samples
        """
        return self.innovation.sample(innov_params, num_samples=num_samples)

    def forward(self, config_dict, X_hist=None, return_params=False):
        """
        Forward pass for inference.

        Args:
            config_dict: configuration dictionary
            X_hist: (batch, p) history (if None, returns only params)
            return_params: if True, return full parameter dict

        Returns:
            If X_hist is None: parameter dict
            Otherwise: X (sampled), params (if return_params=True)
        """
        # Encode and generate parameters
        h = self.encode_config(config_dict)
        params = self.generate_params(h)

        if X_hist is None:
            return params

        # Sample innovation
        eps = self.sample_innovation(params['innov_params'], num_samples=1)

        # Compute AR mean and generate X
        mu = self.compute_ar_mean(params['c'], params['phi'], X_hist)
        X = mu + eps

        if return_params:
            return X, params
        else:
            return X
