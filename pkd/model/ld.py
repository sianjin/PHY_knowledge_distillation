"""Levinson-Durbin recursion for stable AR coefficient construction."""
import torch
import torch.nn as nn


def levinson_durbin_from_pacf(kappa: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Convert PACF (reflection coefficients) to stable AR coefficients
    using full Levinson–Durbin recursion.

    Guarantees stability if |kappa| < 1.
    """
    kappa = kappa.clamp(-1.0 + eps, 1.0 - eps)
    p = kappa.shape[-1]

    phi = []
    for k in range(p):
        kappa_k = kappa[..., k]
        if k == 0:
            phi_k = kappa_k.unsqueeze(-1)
        else:
            prev = phi[-1]
            prev_rev = torch.flip(prev, dims=[-1])
            phi_k = torch.cat(
                [
                    prev - kappa_k.unsqueeze(-1) * prev_rev,
                    kappa_k.unsqueeze(-1),
                ],
                dim=-1,
            )
        phi.append(phi_k)

    return phi[-1]


class PACFToAR(nn.Module):
    """Module for converting PACF to AR coefficients with guaranteed stability."""

    def __init__(self, kappa_max: float = 0.95):
        """
        Args:
            kappa_max: Maximum absolute value for PACF coefficients.
                      Typical values: 0.90-0.98
                      Lower values = more stable but less flexible
                      Higher values = more flexible but closer to unit root
        """
        super().__init__()
        self.kappa_max = kappa_max
        print(f"PACFToAR initialized with kappa_max={kappa_max:.3f} for stability")

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (..., p) unconstrained PACF parameters

        Returns:
            phi: (..., p) stable AR coefficients
            kappa: (..., p) constrained PACF coefficients
        """
        # Map to (-kappa_max, kappa_max) using tanh
        # This ensures |kappa| < kappa_max < 1, guaranteeing stationarity
        kappa = self.kappa_max * torch.tanh(u)

        # Convert to AR coefficients using Levinson-Durbin
        phi = levinson_durbin_from_pacf(kappa)

        return phi, kappa
