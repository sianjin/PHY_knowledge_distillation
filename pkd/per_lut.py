"""AWGN PER lookup table interface."""
import numpy as np
import torch
from scipy.interpolate import interp1d


class AWGNPERLookup:
    """AWGN packet error rate lookup table."""

    def __init__(self, lut_dict):
        """
        Args:
            lut_dict: dict mapping MCS -> (snr_db_array, per_array)
        """
        self.lut_dict = {}

        # Create interpolators for each MCS
        for mcs, (snr_db, per) in lut_dict.items():
            # Use log-linear interpolation (better for PER)
            # Clamp PER to avoid log(0)
            per_clamped = np.clip(per, 1e-10, 1.0)

            # Create interpolator
            interpolator = interp1d(
                snr_db,
                per_clamped,
                kind='linear',
                bounds_error=False,
                fill_value=(1.0, 0.0)  # PER=1 below range, PER=0 above
            )

            self.lut_dict[mcs] = interpolator

    def lookup(self, gamma_eff, mcs):
        """
        Look up PER for given effective SINR and MCS.

        Args:
            gamma_eff: effective SINR (linear scale, can be array or scalar)
            mcs: MCS index (scalar)

        Returns:
            per: packet error rate in [0, 1]
        """
        # Convert to dB
        gamma_eff_np = gamma_eff.cpu().numpy() if torch.is_tensor(gamma_eff) else gamma_eff
        snr_db = 10 * np.log10(gamma_eff_np + 1e-10)

        # Lookup
        if mcs not in self.lut_dict:
            raise ValueError(f"MCS {mcs} not in lookup table")

        per = self.lut_dict[mcs](snr_db)

        # Convert back to torch if needed
        if torch.is_tensor(gamma_eff):
            per = torch.from_numpy(per).to(gamma_eff.device).float()

        return per

    @staticmethod
    def create_dummy_lut(num_mcs=10, snr_range=(-10, 30), num_points=100):
        """Create a dummy LUT for testing."""
        lut_dict = {}

        for mcs in range(num_mcs):
            snr_db = np.linspace(snr_range[0], snr_range[1], num_points)

            # Sigmoid-like PER curve, shifted based on MCS
            snr_threshold = -5 + 2.5 * mcs  # Higher MCS needs higher SNR
            per = 1.0 / (1.0 + np.exp((snr_db - snr_threshold) / 2.0))

            lut_dict[mcs] = (snr_db, per)

        return AWGNPERLookup(lut_dict)
