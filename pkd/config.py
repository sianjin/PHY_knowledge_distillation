"""Configuration structures and utilities."""
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class StaticConfig:
    """Static PHY configuration (unchanged during simulation run)."""
    channel_model: int  # categorical ID
    N_t: int  # number of transmit antennas
    N_r: int  # number of receive antennas
    BW: float  # bandwidth in MHz

    def to_tuple(self) -> Tuple:
        """Convert to hashable tuple for caching."""
        return (self.channel_model, self.N_t, self.N_r, self.BW)


@dataclass
class DynamicConfig:
    """Dynamic PHY configuration (may vary with rate adaptation)."""
    SNR_bar: float  # average SNR in dB
    MCS: int  # modulation and coding scheme ID
    N_ss: int  # number of spatial streams
    R_t: int  # resource allocation index (0=full-band, 1-7=future extensions)

    def to_tuple(self, snr_quant: float = 0.1) -> Tuple:
        """Convert to hashable tuple with SNR quantization."""
        snr_key = np.round(self.SNR_bar / snr_quant) * snr_quant
        return (snr_key, self.MCS, self.N_ss, self.R_t)


@dataclass
class Config:
    """Complete PHY configuration."""
    static: StaticConfig
    dynamic: DynamicConfig

    def make_key(self, snr_quant: float = 0.1) -> Tuple:
        """Create cache key for configuration-driven parameter caching."""
        return self.static.to_tuple() + self.dynamic.to_tuple(snr_quant)


def normalize_continuous_inputs(values: np.ndarray,
                                mean: Optional[np.ndarray] = None,
                                std: Optional[np.ndarray] = None) -> np.ndarray:
    """Normalize continuous configuration inputs."""
    if mean is None or std is None:
        return values
    return (values - mean) / (std + 1e-8)
