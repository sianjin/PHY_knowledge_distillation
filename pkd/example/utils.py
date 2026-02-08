"""Utility functions for PKD examples."""

import numpy as np
from scipy import stats
from scipy.signal import welch
from collections import Counter


def compute_acf(x, max_lag=50):
    """Compute autocorrelation function."""
    x = x - np.mean(x)
    acf = np.correlate(x, x, mode='full')
    acf = acf[len(acf)//2:]
    acf = acf / acf[0]
    return acf[:max_lag+1]


def compute_psd(x, fs=1.0):
    """Compute power spectral density."""
    freqs, psd = welch(x, fs=fs, nperseg=min(256, len(x)//4))
    return freqs, psd


def ljung_box_test(residuals, lags=20):
    """Ljung-Box test for autocorrelation."""
    n = len(residuals)
    acf_vals = compute_acf(residuals, max_lag=lags)

    # Ljung-Box statistic
    lb_stat = n * (n + 2) * np.sum(acf_vals[1:]**2 / (n - np.arange(1, lags+1)))
    p_value = 1 - stats.chi2.cdf(lb_stat, lags)

    return lb_stat, p_value


# ============ Configuration Slice Filtering ============

def get_config_signature(config, keys):
    """Extract a hashable signature tuple from config for specified keys.

    Args:
        config: Dictionary of configuration parameters
        keys: List of keys to include in signature

    Returns:
        Tuple of (key, value) pairs for the specified keys.
        Uses None for missing keys.
    """
    return tuple((k, config.get(k, None)) for k in keys)


def infer_most_common_slice(test_configs, keys):
    """Infer the most common configuration slice from test configs.

    Args:
        test_configs: List of config dictionaries
        keys: List of keys defining the slice (excluding MCS and SNR_bar)

    Returns:
        Dictionary representing the most common slice specification
    """
    # Count signatures
    signatures = [get_config_signature(cfg, keys) for cfg in test_configs]
    counter = Counter(signatures)

    # Get most common
    most_common_sig, count = counter.most_common(1)[0]

    # Convert back to dict
    slice_spec = {k: v for k, v in most_common_sig}

    print(f"\nInferred most common slice ({count}/{len(test_configs)} sequences):")
    for k, v in slice_spec.items():
        print(f"  {k}: {v}")

    return slice_spec


def filter_by_slice(test_sequences, test_configs, slice_spec):
    """Filter test data to keep only sequences matching the slice specification.

    Args:
        test_sequences: List of test sequences
        test_configs: List of test config dicts
        slice_spec: Dictionary specifying fixed configuration values

    Returns:
        Tuple of (filtered_sequences, filtered_configs, kept_indices)
    """
    kept_indices = []

    for i, config in enumerate(test_configs):
        # Check if this config matches the slice
        matches = True
        for key, value in slice_spec.items():
            if config.get(key, None) != value:
                matches = False
                break

        if matches:
            kept_indices.append(i)

    # Filter sequences and configs
    filtered_sequences = [test_sequences[i] for i in kept_indices]
    filtered_configs = [test_configs[i] for i in kept_indices]

    return filtered_sequences, filtered_configs, kept_indices


def format_slice_label(slice_spec):
    """Format slice specification as a compact label for plot titles.

    Args:
        slice_spec: Dictionary of fixed configuration values

    Returns:
        Compact string representation (e.g., "4x2x2, BW=20, Model-D")
    """
    parts = []

    # MIMO configuration
    if 'N_t' in slice_spec and 'N_r' in slice_spec and 'N_ss' in slice_spec:
        parts.append(f"{slice_spec['N_t']}x{slice_spec['N_r']}x{slice_spec['N_ss']}")

    # Bandwidth
    if 'BW' in slice_spec:
        parts.append(f"BW={int(slice_spec['BW'])}MHz")

    # Channel model
    # MATLAB PHY simulator uses 1-based indexing for 802.11n TGn channel models
    if 'channel_model_id' in slice_spec:
        ch_id = slice_spec['channel_model_id']
        # 1-based indexing (MATLAB convention): A=1, B=2, C=3, D=4, E=5, F=6
        ch_names = {
            1: 'Model-A',
            2: 'Model-B',
            3: 'Model-C',
            4: 'Model-D',
            5: 'Model-E',
            6: 'Model-F'
        }
        parts.append(ch_names.get(ch_id, f'Ch={ch_id}'))

    # Packet length
    if 'packet_length' in slice_spec and slice_spec['packet_length'] != 1000:
        parts.append(f"L={slice_spec['packet_length']}B")

    return ', '.join(parts)
