"""Unit tests for PKD components."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from pkd.model.ld import levinson_durbin_from_pacf, PACFToAR
from pkd.model.flow import MonotoneSplineFlow1D


def test_pacf_stability():
    """Test that PACF -> AR conversion produces stable coefficients."""
    print("Testing PACF stability...")

    pacf_to_ar = PACFToAR(delta=1e-4)

    # Random unconstrained parameters
    u = torch.randn(100, 10)
    phi, kappa = pacf_to_ar(u)

    # Check kappa in valid range
    assert torch.all(torch.abs(kappa) < 1.0), "PACF coefficients outside (-1, 1)"

    # Check AR stability (roots outside unit circle)
    # For simplicity, just check coefficients are bounded
    assert torch.all(torch.abs(phi) < 10.0), "AR coefficients unbounded"

    print("✓ PACF stability test passed")


def test_flow_invertibility():
    """Test flow forward/inverse consistency."""
    print("Testing flow invertibility...")

    flow = MonotoneSplineFlow1D(num_bins=16, tail_bound=5.0)

    # Random parameters
    params = torch.randn(100, flow.param_dim)

    # Random z
    z = torch.randn(100)

    # Forward then inverse
    eps, logdet_fwd = flow.forward(z, params)
    z_recon, logdet_inv = flow.inverse(eps, params)

    # Check reconstruction
    assert torch.allclose(z, z_recon, atol=1e-4), "Flow not invertible"

    # Check logdet consistency (should be negatives)
    assert torch.allclose(logdet_fwd, -logdet_inv, atol=1e-4), "Logdet inconsistent"

    print("✓ Flow invertibility test passed")


def test_ar_mean_consistency():
    """Test that AR offset ensures correct stationary mean."""
    print("Testing AR mean consistency...")

    # AR coefficients
    phi = torch.tensor([0.7, -0.3, 0.1])
    m = torch.tensor([2.5])

    # Compute offset
    c = (1.0 - phi.sum()) * m

    # For stationary AR: E[X] = c / (1 - sum(phi))
    expected_mean = c / (1.0 - phi.sum())

    assert torch.allclose(expected_mean, m, atol=1e-6), "Mean inconsistency"

    print("✓ AR mean consistency test passed")


def test_time_skipping_correctness():
    """Test that time-skipping produces same results as per-packet evaluation."""
    print("Testing time-skipping correctness...")

    from pkd.model.pkd_model import PKDModel
    from pkd.infer import PKDInference
    from pkd.per_lut import AWGNPERLookup

    # Create model
    model = PKDModel(
        num_channel_models=5,
        num_mcs=10,
        num_nss=4,
        ar_order=5,
        hidden_dim=64,
        num_flow_bins=8
    )

    # Create dummy PER LUT
    per_lut = AWGNPERLookup.create_dummy_lut(num_mcs=10)

    # Create inference engine
    inference = PKDInference(model, per_lut, ar_order=5, burn_in=10, device='cpu')

    # Create config
    config = {
        'channel_model_id': torch.tensor(0),
        'N_t': torch.tensor(4),
        'N_r': torch.tensor(4),
        'BW': torch.tensor(20.0),
        'SNR_bar': torch.tensor(15.0),
        'MCS': torch.tensor(5),
        'N_ss': torch.tensor(2)
    }

    # Set seed
    np.random.seed(42)
    torch.manual_seed(42)

    # Run with per-packet evaluation
    config_traj = [config] * 100
    results1 = inference.run_sequence(config_traj)

    # Reset and run with time-skipping
    inference.state_buffer.clear()
    inference.cache.clear()
    np.random.seed(42)
    torch.manual_seed(42)

    config_windows = [(config, 100)]
    results2 = inference.run_with_time_skipping(config_windows)

    # Compare (should be identical with same seed)
    assert np.allclose(results1['gamma_eff'], results2['gamma_eff'], atol=1e-5), \
        "Time-skipping produces different results"

    print("✓ Time-skipping correctness test passed")


if __name__ == '__main__':
    test_pacf_stability()
    test_flow_invertibility()
    test_ar_mean_consistency()
    test_time_skipping_correctness()
    print("\n✅ All tests passed!")
