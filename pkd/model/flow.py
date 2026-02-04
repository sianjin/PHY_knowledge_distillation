"""1D Monotone Spline Flow for non-Gaussian innovation distribution."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MonotoneSplineFlow1D(nn.Module):
    """
    1D monotone spline flow using rational quadratic splines.
    Implements conditional transformation for innovation distribution.
    """

    def __init__(self,
                 num_bins: int = 16,
                 tail_bound: float = 5.0,
                 min_bin_size: float = 1e-3,
                 min_derivative: float = 1e-3):
        super().__init__()
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.min_bin_size = min_bin_size
        self.min_derivative = min_derivative

        # Parameter dimension: widths + heights + derivatives
        self.param_dim = 3 * num_bins + 1

    def _process_params(self, params: torch.Tensor):
        """
        Convert raw parameters to valid spline parameters.

        Args:
            params: (..., param_dim) raw parameters

        Returns:
            widths, heights, derivatives with constraints enforced
        """
        K = self.num_bins

        # Split parameters
        widths_raw = params[..., :K]
        heights_raw = params[..., K:2*K]
        derivs_raw = params[..., 2*K:]

        # Widths: epsilon-mixed softmax for minimum bin size
        widths_soft = F.softmax(widths_raw, dim=-1)
        widths = self.min_bin_size + (1.0 - K * self.min_bin_size) * widths_soft

        # Heights: same treatment
        heights_soft = F.softmax(heights_raw, dim=-1)
        heights = self.min_bin_size + (1.0 - K * self.min_bin_size) * heights_soft

        # Derivatives: positive with minimum
        derivatives = F.softplus(derivs_raw) + self.min_derivative

        return widths, heights, derivatives

    def _rational_quadratic_spline(self, x, widths, heights, derivatives, inverse=False):
        """
        Rational quadratic spline transformation.

        Args:
            x: input values
            widths, heights, derivatives: spline parameters
            inverse: if True, compute inverse transformation

        Returns:
            y: transformed values
            logabsdet: log absolute determinant of Jacobian
        """
        # Get batch dimensions
        batch_shape = x.shape[:-1] if x.ndim > 1 else ()

        # Cumulative widths and heights (bin boundaries)
        cum_widths = torch.cumsum(widths, dim=-1)
        cum_widths = F.pad(cum_widths, (1, 0), value=0.0)
        cum_widths = cum_widths * 2 * self.tail_bound - self.tail_bound

        cum_heights = torch.cumsum(heights, dim=-1)
        cum_heights = F.pad(cum_heights, (1, 0), value=0.0)
        cum_heights = cum_heights * 2 * self.tail_bound - self.tail_bound

        # Find which bin each x falls into
        if inverse:
            bin_idx = self._searchsorted(cum_heights, x)
        else:
            bin_idx = self._searchsorted(cum_widths, x)

        # Clamp to valid range (avoid in-place operation)
        bin_idx = bin_idx.clamp(0, self.num_bins - 1)

        # Get bin parameters (unsqueeze bin_idx for gather operation)
        bin_idx_expanded = bin_idx.unsqueeze(-1)
        input_cum_widths = cum_widths.gather(-1, bin_idx_expanded)
        input_bin_widths = widths.gather(-1, bin_idx_expanded)

        input_cum_heights = cum_heights.gather(-1, bin_idx_expanded)
        input_bin_heights = heights.gather(-1, bin_idx_expanded)

        input_delta = derivatives.gather(-1, bin_idx_expanded)
        input_delta_plus = derivatives.gather(-1, bin_idx_expanded + 1)

        if inverse:
            # Inverse transformation
            theta = (x - input_cum_heights) / (input_bin_heights.clamp(min=1e-8))
            theta = torch.clamp(theta, 0.0, 1.0)  # Ensure theta is in [0,1]
            theta_one_minus_theta = theta * (1 - theta)

            denominator = input_delta + input_delta_plus - 2 * input_delta * theta + theta_one_minus_theta * (input_delta_plus - input_delta)
            numerator = input_bin_heights * (input_delta * theta.pow(2) + input_delta_plus * theta_one_minus_theta)

            xi = numerator / (denominator.clamp(min=1e-8))
            y = input_cum_widths + xi

            # Derivative
            derivative_numerator = (input_delta_plus * theta.pow(2) +
                                   2 * input_delta * theta_one_minus_theta +
                                   input_delta * (1 - theta).pow(2))

            # Clamp to avoid log of zero/negative
            derivative_numerator = torch.clamp(derivative_numerator, min=1e-8)
            denominator_clamped = torch.clamp(denominator, min=1e-8)

            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator_clamped) + \
                       torch.log(input_bin_heights.clamp(min=1e-8)) - torch.log(input_bin_widths.clamp(min=1e-8))
        else:
            # Forward transformation
            xi = (x - input_cum_widths) / (input_bin_widths.clamp(min=1e-8))
            xi = torch.clamp(xi, 0.0, 1.0)  # Ensure xi is in [0,1]
            xi_one_minus_xi = xi * (1 - xi)

            numerator = input_bin_heights * (input_delta * xi.pow(2) + input_delta_plus * xi_one_minus_xi)
            denominator = input_delta + (input_delta_plus - input_delta) * xi_one_minus_xi

            theta = numerator / (denominator.clamp(min=1e-8))
            y = input_cum_heights + theta

            # Derivative
            derivative_numerator = (input_delta_plus * xi.pow(2) +
                                   2 * input_delta * xi_one_minus_xi +
                                   input_delta * (1 - xi).pow(2))

            # Clamp to avoid log of zero/negative
            derivative_numerator = torch.clamp(derivative_numerator, min=1e-8)
            denominator_clamped = torch.clamp(denominator, min=1e-8)

            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator_clamped) + \
                       torch.log(input_bin_widths.clamp(min=1e-8)) - torch.log(input_bin_heights.clamp(min=1e-8))

        return y.squeeze(-1), logabsdet.squeeze(-1)

    def _searchsorted(self, bin_locations, inputs):
        """Find which bin inputs fall into."""
        # inputs shape: (..., 1)
        # bin_locations shape: (..., num_bins+1)

        # Remove the last dimension from inputs for comparison
        inputs_squeezed = inputs.squeeze(-1)

        # Expand inputs for comparison with bin boundaries
        inputs_expanded = inputs_squeezed.unsqueeze(-1)

        # Compare with bin boundaries
        return torch.sum(inputs_expanded >= bin_locations[..., :-1], dim=-1) - 1

    def forward(self, z: torch.Tensor, params: torch.Tensor):
        """
        Forward transformation: z -> eps

        Args:
            z: (...,) standard normal samples
            params: (..., param_dim) flow parameters

        Returns:
            eps: transformed samples
            logabsdet: log|deps/dz|
        """
        widths, heights, derivatives = self._process_params(params)

        # Add dimension for compatibility
        z_expanded = z.unsqueeze(-1)

        eps, logabsdet = self._rational_quadratic_spline(
            z_expanded, widths, heights, derivatives, inverse=False
        )

        return eps, logabsdet

    def inverse(self, eps: torch.Tensor, params: torch.Tensor):
        """
        Inverse transformation: eps -> z

        Args:
            eps: innovation samples
            params: (..., param_dim) flow parameters

        Returns:
            z: base distribution samples
            logabsdet: log|dz/deps|
        """
        widths, heights, derivatives = self._process_params(params)

        # Add dimension for compatibility
        eps_expanded = eps.unsqueeze(-1)

        z, logabsdet = self._rational_quadratic_spline(
            eps_expanded, widths, heights, derivatives, inverse=True
        )

        return z, logabsdet
