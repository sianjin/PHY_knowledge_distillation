"""Parameter heads for stochastic process generation."""
import torch
import torch.nn as nn


class MeanHead(nn.Module):
    """Predict conditional mean of log-effective-SINR."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, h):
        """
        Args:
            h: (batch, hidden_dim)

        Returns:
            m: (batch,) conditional mean
        """
        return self.mlp(h).squeeze(-1)


class PACFHead(nn.Module):
    """Predict unconstrained PACF parameters."""

    def __init__(self, hidden_dim: int = 128, ar_order: int = 10):
        super().__init__()
        self.ar_order = ar_order

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, ar_order)
        )

    def forward(self, h):
        """
        Args:
            h: (batch, hidden_dim)

        Returns:
            u: (batch, p) unconstrained PACF parameters
        """
        return self.mlp(h)


class FlowHead(nn.Module):
    """Predict parameters for conditional monotone spline flow."""

    def __init__(self, hidden_dim: int = 128, num_bins: int = 16):
        super().__init__()

        # Flow parameter dimension: 3*K + 1
        param_dim = 3 * num_bins + 1

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, param_dim)
        )

    def forward(self, h):
        """
        Args:
            h: (batch, hidden_dim)

        Returns:
            psi: (batch, param_dim) flow parameters
        """
        return self.mlp(h)
