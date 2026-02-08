"""Innovation distribution models for AR process."""
import torch
import torch.nn as nn
import numpy as np
from scipy.special import gamma as gamma_func


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


class SGNInnovation(nn.Module):
    """
    Skew Generalized Normal (SGN) innovation distribution.

    Models innovations as: eps ~ SGN(0, sigma, beta, lambda)
    where sigma (scale), beta (shape), lambda (skewness) are predicted from conditioning.

    The SGN distribution generalizes the Gaussian:
    - When lambda = 0 and beta = 2, recovers standard Gaussian
    - lambda controls skewness
    - beta controls tail heaviness (beta=2 is Gaussian-like)
    """

    def __init__(self, hidden_dim: int = 128, min_sigma: float = 0.1,
                 min_beta: float = 0.5, max_beta: float = 4.0):
        """
        Args:
            hidden_dim: Input conditioning dimension
            min_sigma: Minimum scale for numerical stability
            min_beta: Minimum shape parameter
            max_beta: Maximum shape parameter
        """
        super().__init__()
        self.min_sigma = min_sigma
        self.min_beta = min_beta
        self.max_beta = max_beta

        # MLP to predict SGN parameters
        self.param_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # [sigma, beta, lambda]
        )

        # Initialize to predict near-Gaussian distribution initially
        # [log(sigma) ≈ 0, beta ≈ 2, lambda ≈ 0]
        with torch.no_grad():
            self.param_net[-1].weight.mul_(0.1)
            self.param_net[-1].bias.copy_(torch.tensor([0.0, 2.0, 0.0]))

    def forward(self, h):
        """
        Predict innovation parameters from conditioning.

        Args:
            h: (batch, hidden_dim) conditioning representation

        Returns:
            params: dict with keys 'sigma', 'beta', 'lambda'
                sigma: (batch,) scale parameter > 0
                beta: (batch,) shape parameter > 0
                lambda: (batch,) skewness parameter (unconstrained)
        """
        raw_params = self.param_net(h)  # (batch, 3)

        # sigma: use softplus to ensure > 0, add min for stability
        sigma = torch.nn.functional.softplus(raw_params[:, 0]) + self.min_sigma

        # beta: use softplus and clamp to reasonable range
        beta = torch.nn.functional.softplus(raw_params[:, 1])
        beta = torch.clamp(beta, self.min_beta, self.max_beta)

        # lambda: unconstrained skewness
        lam = raw_params[:, 2]

        return {'sigma': sigma, 'beta': beta, 'lambda': lam}

    def _gn_base_pdf(self, u, beta):
        """Generalized normal base PDF: phi(u; beta) with location 0, scale 1."""
        # phi(u;beta) = beta / (2 * Gamma(1/beta)) * exp(-|u|^beta)
        gamma_val = torch.tensor(gamma_func(1.0 / beta.cpu().numpy()),
                                 device=beta.device, dtype=beta.dtype)
        coef = beta / (2.0 * gamma_val)
        return coef * torch.exp(-torch.abs(u) ** beta)

    def _sgn_pdf(self, eps, sigma, beta, lam):
        """
        SGN probability density function.

        f(eps) = (2/sigma) * phi(z; beta) * Phi(sqrt(2) * lambda * z)
        where z = eps/sigma
        """
        z = eps / sigma

        # Generalized normal base PDF
        phi = self._gn_base_pdf(z, beta)

        # Standard normal CDF for skewing
        skew_term = torch.distributions.Normal(0, 1).cdf(np.sqrt(2) * lam * z)

        # SGN PDF
        pdf = (2.0 / sigma) * phi * skew_term

        return pdf

    def log_prob(self, eps, params):
        """
        Compute log probability of innovations.

        Args:
            eps: (batch,) innovation values
            params: dict with 'sigma', 'beta', 'lambda'

        Returns:
            log_p: (batch,) log probability
        """
        sigma = params['sigma']
        beta = params['beta']
        lam = params['lambda']

        z = eps / sigma

        # log phi(z; beta) = log(beta) - log(2) - log(Gamma(1/beta)) - |z|^beta
        gamma_val = torch.tensor([gamma_func(1.0 / b.item()) for b in beta],
                                 device=beta.device, dtype=beta.dtype)

        log_phi = (torch.log(beta) - torch.log(torch.tensor(2.0, device=beta.device))
                   - torch.log(gamma_val) - torch.abs(z) ** beta)

        # log Phi(sqrt(2) * lambda * z) - use log_cdf for numerical stability
        # Phi(x) can be very small (near 0), so we use log space to avoid log(0) = -inf
        skew_arg = np.sqrt(2) * lam * z
        normal_dist = torch.distributions.Normal(0, 1)

        # Clamp CDF to avoid log(0) issues
        cdf_val = normal_dist.cdf(skew_arg)
        cdf_val = torch.clamp(cdf_val, min=1e-10, max=1-1e-10)
        log_skew = torch.log(cdf_val)

        # log f(eps) = log(2) - log(sigma) + log_phi + log_skew
        log_p = (torch.log(torch.tensor(2.0, device=eps.device))
                 - torch.log(sigma) + log_phi + log_skew)

        return log_p

    def sample(self, params, num_samples=1):
        """
        Sample innovations from SGN using rejection sampling.

        Args:
            params: dict with 'sigma', 'beta', 'lambda'
            num_samples: number of samples per parameter set

        Returns:
            eps: (..., num_samples) innovation samples
        """
        sigma = params['sigma']
        beta = params['beta']
        lam = params['lambda']

        batch_shape = sigma.shape

        # For each batch element, use rejection sampling
        samples = []
        for i in range(batch_shape[0]):
            s_i = sigma[i].item()
            b_i = beta[i].item()
            l_i = lam[i].item()

            batch_samples = []
            for _ in range(num_samples):
                # Use rejection sampling with proposal = GN(0, 1, beta)
                accepted = False
                max_tries = 1000
                tries = 0

                while not accepted and tries < max_tries:
                    # Sample from generalized normal base
                    u = np.random.randn()
                    z = np.sign(u) * (np.abs(u) ** (1.0/b_i))

                    # Accept with probability Phi(sqrt(2) * lambda * z) / 0.5
                    # (using 0.5 as upper bound for CDF)
                    from scipy.stats import norm
                    accept_prob = 2.0 * norm.cdf(np.sqrt(2) * l_i * z)

                    if np.random.rand() < accept_prob:
                        accepted = True
                        eps_val = s_i * z
                        batch_samples.append(eps_val)

                    tries += 1

                if not accepted:
                    # Fallback to Gaussian if rejection fails
                    batch_samples.append(s_i * np.random.randn())

            samples.append(batch_samples)

        # Convert to tensor
        samples_tensor = torch.tensor(samples, device=sigma.device, dtype=sigma.dtype)

        if num_samples == 1:
            return samples_tensor.squeeze(-1)
        else:
            return samples_tensor

    def sample_numpy(self, params):
        """
        Sample single innovation (for numpy inference).

        Args:
            params: dict with 'sigma' (float), 'beta' (float), 'lambda' (float)

        Returns:
            eps: float, innovation sample
        """
        sigma = params['sigma']
        beta = params['beta']
        lam = params['lambda']

        # Simple rejection sampling
        accepted = False
        max_tries = 1000
        tries = 0

        while not accepted and tries < max_tries:
            # Sample from generalized normal base
            u = np.random.randn()
            z = np.sign(u) * (np.abs(u) ** (1.0/beta))

            # Accept with probability Phi(sqrt(2) * lambda * z) / 0.5
            from scipy.stats import norm
            accept_prob = 2.0 * norm.cdf(np.sqrt(2) * lam * z)

            if np.random.rand() < accept_prob:
                accepted = True
                return sigma * z

            tries += 1

        # Fallback to Gaussian
        return sigma * np.random.randn()


def create_innovation_model(innovation_type: str, hidden_dim: int = 128, **kwargs):
    """
    Factory function to create innovation models.

    Args:
        innovation_type: 'sgn', 'gaussian', or 'flow'
        hidden_dim: Hidden dimension for conditioning
        **kwargs: Additional arguments for specific innovation types

    Returns:
        innovation_model: SGNInnovation, GaussianInnovation, or FlowInnovation
    """
    if innovation_type == 'sgn':
        min_sigma = kwargs.get('min_sigma', 0.1)
        min_beta = kwargs.get('min_beta', 0.5)
        max_beta = kwargs.get('max_beta', 4.0)
        return SGNInnovation(hidden_dim, min_sigma, min_beta, max_beta)

    elif innovation_type == 'gaussian':
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
                        f"Choose 'sgn', 'gaussian', or 'flow'.")
