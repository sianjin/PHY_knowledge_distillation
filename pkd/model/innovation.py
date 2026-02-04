"""Innovation distribution models for AR process."""
import torch
import torch.nn as nn
import numpy as np


class GaussianInnovation(nn.Module):
    """
    Gaussian innovation distribution with learnable variance.

    Models innovations as: eps ~ N(0, sigma^2(h))
    where sigma^2 is predicted from conditioning.
    """

    def __init__(self, hidden_dim: int = 128, min_sigma: float = 0.1):
        """
        Args:
            hidden_dim: Input conditioning dimension
            min_sigma: Minimum standard deviation for numerical stability
        """
        super().__init__()
        self.min_sigma = min_sigma

        # MLP to predict log(sigma) for numerical stability
        self.sigma_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Initialize to predict sigma ≈ 1.0 initially
        nn.init.zeros_(self.sigma_net[-1].weight)
        nn.init.zeros_(self.sigma_net[-1].bias)

    def forward(self, h):
        """
        Predict innovation parameters from conditioning.

        Args:
            h: (batch, hidden_dim) conditioning representation

        Returns:
            sigma: (batch,) predicted standard deviation
        """
        # Predict log(sigma) and transform to sigma
        log_sigma = self.sigma_net(h).squeeze(-1)
        sigma = torch.exp(log_sigma) + self.min_sigma

        return sigma

    def log_prob(self, eps, sigma):
        """
        Compute log probability of innovations.

        Args:
            eps: (batch,) innovation values
            sigma: (batch,) standard deviations

        Returns:
            log_p: (batch,) log probability
        """
        # log N(eps | 0, sigma^2)
        log_p = -0.5 * ((eps / sigma) ** 2 +
                        2 * torch.log(sigma) +
                        torch.log(torch.tensor(2 * np.pi, device=eps.device)))
        return log_p

    def sample(self, sigma, num_samples=1):
        """
        Sample innovations from Gaussian.

        Args:
            sigma: (...,) standard deviations
            num_samples: number of samples per sigma

        Returns:
            eps: (..., num_samples) innovation samples
        """
        shape = sigma.shape + (num_samples,) if num_samples > 1 else sigma.shape
        z = torch.randn(shape, device=sigma.device)

        if num_samples == 1:
            eps = sigma * z
        else:
            eps = sigma.unsqueeze(-1) * z

        return eps

    def sample_numpy(self, sigma):
        """
        Sample single innovation (for numpy inference).

        Args:
            sigma: float, standard deviation

        Returns:
            eps: float, innovation sample
        """
        return np.random.randn() * sigma


class FlowInnovation(nn.Module):
    """
    Wrapper for flow-based innovation distribution.

    Maintains same interface as GaussianInnovation for easy swapping.
    """

    def __init__(self, hidden_dim: int = 128, num_bins: int = 16,
                 tail_bound: float = 5.0, min_bin_size: float = 1e-3,
                 min_derivative: float = 1e-3):
        """
        Args:
            hidden_dim: Input conditioning dimension
            num_bins: Number of spline bins
            tail_bound: Tail bound for spline
            min_bin_size: Minimum bin size
            min_derivative: Minimum derivative
        """
        super().__init__()

        from .flow import MonotoneSplineFlow1D
        from .heads import FlowHead

        self.flow_head = FlowHead(hidden_dim, num_bins)
        self.flow = MonotoneSplineFlow1D(
            num_bins=num_bins,
            tail_bound=tail_bound,
            min_bin_size=min_bin_size,
            min_derivative=min_derivative
        )

    def forward(self, h):
        """
        Predict flow parameters from conditioning.

        Args:
            h: (batch, hidden_dim) conditioning representation

        Returns:
            psi: (batch, flow_param_dim) flow parameters
        """
        return self.flow_head(h)

    def log_prob(self, eps, psi):
        """
        Compute log probability of innovations through flow.

        Args:
            eps: (batch,) innovation values
            psi: (batch, flow_param_dim) flow parameters

        Returns:
            log_p: (batch,) log probability
        """
        # Flow inverse: eps -> z
        z, logabsdet = self.flow.inverse(eps, psi)

        # Standard normal log-likelihood
        log_normal = -0.5 * (z ** 2 + torch.log(torch.tensor(2 * np.pi, device=z.device)))

        # Total log probability
        log_p = log_normal + logabsdet
        return log_p

    def sample(self, psi, num_samples=1):
        """
        Sample innovations from flow.

        Args:
            psi: (..., flow_param_dim) flow parameters
            num_samples: number of samples per parameter set

        Returns:
            eps: (..., num_samples) innovation samples
        """
        batch_shape = psi.shape[:-1]

        # Sample base distribution
        z = torch.randn(*batch_shape, num_samples, device=psi.device)

        # Transform through flow
        eps_list = []
        for i in range(num_samples):
            eps_i, _ = self.flow.forward(z[..., i], psi)
            eps_list.append(eps_i)

        if num_samples == 1:
            return eps_list[0]
        else:
            return torch.stack(eps_list, dim=-1)

    def sample_numpy(self, psi):
        """
        Sample single innovation (for numpy inference).

        Args:
            psi: numpy array, flow parameters

        Returns:
            eps: float, innovation sample
        """
        # Convert to torch
        z = np.random.randn()
        psi_torch = torch.from_numpy(psi).unsqueeze(0).float()
        z_torch = torch.tensor([z]).float()

        with torch.no_grad():
            eps, _ = self.flow.forward(z_torch, psi_torch)

        return eps.item()


def create_innovation_model(innovation_type: str, hidden_dim: int = 128, **kwargs):
    """
    Factory function to create innovation models.

    Args:
        innovation_type: 'gaussian' or 'flow'
        hidden_dim: Hidden dimension for conditioning
        **kwargs: Additional arguments for specific innovation types

    Returns:
        innovation_model: GaussianInnovation or FlowInnovation
    """
    if innovation_type == 'gaussian':
        min_sigma = kwargs.get('min_sigma', 0.1)
        return GaussianInnovation(hidden_dim, min_sigma)

    elif innovation_type == 'flow':
        num_bins = kwargs.get('num_flow_bins', 16)
        tail_bound = kwargs.get('flow_tail_bound', 5.0)
        min_bin_size = kwargs.get('min_bin_size', 1e-3)
        min_derivative = kwargs.get('min_derivative', 1e-3)
        return FlowInnovation(hidden_dim, num_bins, tail_bound,
                             min_bin_size, min_derivative)

    else:
        raise ValueError(f"Unknown innovation type: {innovation_type}. "
                        f"Choose 'gaussian' or 'flow'.")
