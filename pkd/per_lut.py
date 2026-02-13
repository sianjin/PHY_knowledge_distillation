"""AWGN PER lookup table interface."""
import numpy as np
import torch
from scipy.interpolate import interp1d


# LDPC PER Lookup Table Data
# Extracted from TGax evaluation methodology
# Shape: 12 MCS × 12 SNR points × 2 [SNR_dB, PER]
# Specified for packet length L0=1458 bytes
LDPC_PER_TABLE = [
    # MCS 0
    [
        [-1.50, 1.0000000000e+00],
        [-1.25, 9.7945205479e-01],
        [-1.00, 6.0483383686e-01],
        [-0.75, 1.7049906319e-01],
        [-0.50, 3.3183053769e-02],
        [-0.25, 5.3400000000e-03],
        [0.00, 8.5000000000e-04],
        [0.25, 2.2000000000e-04],
        [0.50, 4.0000000000e-05],
        [0.75, 0.0000000000e+00],
        [1.00, 0.0000000000e+00],
        [1.25, 0.0000000000e+00],
    ],
    # MCS 1
    [
        [1.50, 1.0000000000e+00],
        [1.75, 9.7468354430e-01],
        [2.00, 6.2328767123e-01],
        [2.25, 1.8585220943e-01],
        [2.50, 3.3972509757e-02],
        [2.75, 5.5200000000e-03],
        [3.00, 8.3000000000e-04],
        [3.25, 1.5000000000e-04],
        [3.50, 3.0000000000e-05],
        [3.75, 0.0000000000e+00],
        [4.00, 0.0000000000e+00],
        [4.25, 0.0000000000e+00],
    ],
    # MCS 2
    [
        [4.00, 1.0000000000e+00],
        [4.25, 9.8717948718e-01],
        [4.50, 6.2562500000e-01],
        [4.75, 1.5801104972e-01],
        [5.00, 2.0879831460e-02],
        [5.25, 2.4600000000e-03],
        [5.50, 3.4000000000e-04],
        [5.75, 3.0000000000e-05],
        [6.00, 0.0000000000e+00],
        [6.25, 0.0000000000e+00],
        [6.50, 0.0000000000e+00],
        [6.75, 0.0000000000e+00],
    ],
    # MCS 3
    [
        [7.00, 9.9800598205e-01],
        [7.25, 9.4344957587e-01],
        [7.50, 5.7894736842e-01],
        [7.75, 2.0643431635e-01],
        [8.00, 4.8425330173e-02],
        [8.25, 9.3200000000e-03],
        [8.50, 1.8500000000e-03],
        [8.75, 4.0000000000e-04],
        [9.00, 1.1000000000e-04],
        [9.25, 2.0000000000e-05],
        [9.50, 0.0000000000e+00],
        [9.75, 0.0000000000e+00],
    ],
    # MCS 4
    [
        [10.00, 1.0000000000e+00],
        [10.25, 9.9305555556e-01],
        [10.50, 7.0892351275e-01],
        [10.75, 2.4722153618e-01],
        [11.00, 4.7017379051e-02],
        [11.25, 5.8500000000e-03],
        [11.50, 9.1000000000e-04],
        [11.75, 1.6000000000e-04],
        [12.00, 3.0000000000e-05],
        [12.25, 0.0000000000e+00],
        [12.50, 0.0000000000e+00],
        [12.75, 0.0000000000e+00],
    ],
    # MCS 5
    [
        [14.00, 1.0000000000e+00],
        [14.25, 9.9701195219e-01],
        [14.50, 9.1834862385e-01],
        [14.75, 5.3788285868e-01],
        [15.00, 1.6611350813e-01],
        [15.25, 3.6896424622e-02],
        [15.50, 6.5200000000e-03],
        [15.75, 1.0100000000e-03],
        [16.00, 3.1000000000e-04],
        [16.25, 5.0000000000e-05],
        [16.50, 0.0000000000e+00],
        [16.75, 0.0000000000e+00],
    ],
    # MCS 6
    [
        [15.50, 1.0000000000e+00],
        [15.75, 9.8137254902e-01],
        [16.00, 7.3929098966e-01],
        [16.25, 3.3112801852e-01],
        [16.50, 8.1481481481e-02],
        [16.75, 1.6190861302e-02],
        [17.00, 2.7400000000e-03],
        [17.25, 5.2000000000e-04],
        [17.50, 5.0000000000e-05],
        [17.75, 3.0000000000e-05],
        [18.00, 0.0000000000e+00],
        [18.25, 0.0000000000e+00],
    ],
    # MCS 7
    [
        [17.00, 1.0000000000e+00],
        [17.25, 9.7753906250e-01],
        [17.50, 7.3983739837e-01],
        [17.75, 3.3189655172e-01],
        [18.00, 9.6370463079e-02],
        [18.25, 2.1824920964e-02],
        [18.50, 4.6700000000e-03],
        [18.75, 8.7000000000e-04],
        [19.00, 1.8000000000e-04],
        [19.25, 3.0000000000e-05],
        [19.50, 0.0000000000e+00],
        [19.75, 0.0000000000e+00],
    ],
    # MCS 8
    [
        [20.50, 1.0000000000e+00],
        [20.75, 9.9502982107e-01],
        [21.00, 8.9695340502e-01],
        [21.25, 5.6267566048e-01],
        [21.50, 2.0919540230e-01],
        [21.75, 5.5956174185e-02],
        [22.00, 1.1693925234e-02],
        [22.25, 2.4900000000e-03],
        [22.50, 3.8000000000e-04],
        [22.75, 1.3000000000e-04],
        [23.00, 4.0000000000e-05],
        [23.25, 1.2307692308e-05],
    ],
    # MCS 9
    [
        [22.25, 1.0000000000e+00],
        [22.50, 9.9900199601e-01],
        [22.75, 9.4078947368e-01],
        [23.00, 6.3595933926e-01],
        [23.25, 2.7193697365e-01],
        [23.50, 8.6960298845e-02],
        [23.75, 2.2134264992e-02],
        [24.00, 4.9700000000e-03],
        [24.25, 1.1000000000e-03],
        [24.50, 3.2000000000e-04],
        [24.75, 4.0000000000e-05],
        [25.00, 0.0000000000e+00],
    ],
    # MCS 10
    [
        [25.50, 1.0000000000e+00],
        [25.75, 1.0000000000e+00],
        [26.00, 9.4971537002e-01],
        [26.25, 6.8655692730e-01],
        [26.50, 3.2938466601e-01],
        [26.75, 1.1620617599e-01],
        [27.00, 3.4385627426e-02],
        [27.25, 8.7700000000e-03],
        [27.50, 2.1300000000e-03],
        [27.75, 5.4000000000e-04],
        [28.00, 9.0000000000e-05],
        [28.25, 2.0000000000e-05],
    ],
    # MCS 11
    [
        [27.50, 1.0000000000e+00],
        [27.75, 1.0000000000e+00],
        [28.00, 9.4881516588e-01],
        [28.25, 7.5263157895e-01],
        [28.50, 4.0233118971e-01],
        [28.75, 1.6207901554e-01],
        [29.00, 5.1486472585e-02],
        [29.25, 1.3095580732e-02],
        [29.50, 3.5900000000e-03],
        [29.75, 1.0000000000e-03],
        [30.00, 2.2000000000e-04],
        [30.25, 6.0000000000e-05],
    ],
]


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

    def lookup(self, gamma_eff_db, mcs, packet_length=1458):
        """
        Look up PER for given effective SINR and MCS.

        Args:
            gamma_eff_db: effective SINR in dB scale (can be array or scalar)
            mcs: MCS index (scalar)
            packet_length: data packet length in bytes (default: 1458)

        Returns:
            per: packet error rate in [0, 1], adjusted for packet length

        Note:
            The AWGN PER LUT is specified for L0=1458 bytes (TGax evaluation methodology).
            For different packet lengths, PER is adjusted using:
                per_adj = 1 - (1 - per_base)^(packet_length / L0)
        """
        # Input is already in dB scale
        gamma_eff_np = gamma_eff_db.cpu().numpy() if torch.is_tensor(gamma_eff_db) else gamma_eff_db
        snr_db = gamma_eff_np

        # Lookup base PER for L0=1458 bytes
        if mcs not in self.lut_dict:
            raise ValueError(f"MCS {mcs} not in lookup table")

        per_base = self.lut_dict[mcs](snr_db)

        # Adjust for packet length (L0 = 1458 bytes per TGax evaluation methodology)
        L0 = 1458
        if packet_length != L0:
            # per_adj = 1 - (1 - per_base)^(packet_length / L0)
            # Clamp per_base to avoid numerical issues with (1 - per_base)^exponent
            per_base_clamped = np.clip(per_base, 1e-10, 1.0 - 1e-10)
            exponent = packet_length / L0
            per = 1.0 - np.power(1.0 - per_base_clamped, exponent)
        else:
            per = per_base

        # Convert back to torch if needed
        if torch.is_tensor(gamma_eff_db):
            per = torch.from_numpy(per).to(gamma_eff_db.device).float()

        return per

    @staticmethod
    def load_ldpc_lut():
        """Load LDPC PER lookup table from embedded data.

        Returns:
            AWGNPERLookup instance with LDPC PER curves for MCS 0-11.

        Note:
            This LUT is specified for packet length L0=1458 bytes per TGax evaluation methodology.
            The data is embedded directly in this module (LDPC_PER_TABLE).
        """
        # Build dictionary mapping MCS -> (snr_db, per)
        lut_dict = {}

        for mcs, mcs_data in enumerate(LDPC_PER_TABLE):
            # Convert list of [SNR, PER] pairs to separate arrays
            mcs_array = np.array(mcs_data)
            snr_db = mcs_array[:, 0]  # Shape: (12,)
            per = mcs_array[:, 1]      # Shape: (12,)

            lut_dict[mcs] = (snr_db, per)

        return AWGNPERLookup(lut_dict)
