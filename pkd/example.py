"""Example usage of PKD model."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import welch
from model.pkd_model import PKDModel
from per_lut import AWGNPERLookup
from infer import PKDInference
from train import train_pkd


# ============ Training Example ============
def example_training():
    """Example training workflow."""

    # Create model with Gaussian innovation
    model = PKDModel(
        num_channel_models=5,
        num_mcs=10,
        num_nss=4,
        ar_order=10,
        hidden_dim=128,
        kappa_max=0.95,  # Keep PACF away from ±1 for stability
        innovation_type='gaussian',  # Use Gaussian innovation with learnable sigma
        min_sigma=0.1  # Minimum sigma for numerical stability
    )

    # Generate dummy teacher data (replace with real PHY simulator output)
    def generate_dummy_sequence(length=1000, mu=2.0, phi=0.9, sigma=0.5):
        """Generate effective SINR sequence using log-AR(1) process.

        log(gamma_eff[t]) = mu + phi * (log(gamma_eff[t-1]) - mu) + epsilon[t]
        where epsilon[t] ~ N(0, sigma^2)

        Args:
            length: Sequence length
            mu: Long-term mean of log(gamma_eff)
            phi: AR(1) coefficient (autocorrelation), |phi| < 1 for stationarity
            sigma: Standard deviation of innovation noise
        """
        log_gamma = np.zeros(length)
        # Initialize from stationary distribution: N(mu, sigma^2 / (1 - phi^2))
        log_gamma[0] = np.random.randn() * sigma / np.sqrt(1 - phi**2) + mu

        # Generate AR(1) process in log-space
        for t in range(1, length):
            epsilon = np.random.randn() * sigma
            log_gamma[t] = mu + phi * (log_gamma[t-1] - mu) + epsilon

        # Transform back to linear scale
        gamma_eff = np.exp(log_gamma)
        return gamma_eff

    # Training data
    train_sequences = [generate_dummy_sequence() for _ in range(100)]
    train_configs = [{
        'channel_model_id': 0,
        'N_t': 4,
        'N_r': 4,
        'BW': 20.0,
        'SNR_bar': 15.0 + np.random.randn() * 2,
        'MCS': np.random.randint(1, 11),  # 1-10
        'N_ss': np.random.randint(1, 5)   # 1-4
    } for _ in range(100)]

    # Validation data
    val_sequences = [generate_dummy_sequence() for _ in range(20)]
    val_configs = [{
        'channel_model_id': 0,
        'N_t': 4,
        'N_r': 4,
        'BW': 20.0,
        'SNR_bar': 15.0 + np.random.randn() * 2,
        'MCS': np.random.randint(1, 11),  # 1-10
        'N_ss': np.random.randint(1, 5)   # 1-4
    } for _ in range(20)]

    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_config = {
        'num_channel_models': 5,
        'num_mcs': 10,
        'num_nss': 4,
        'ar_order': 10,
        'hidden_dim': 128,
        'kappa_max': 0.95,
        'innovation_type': 'gaussian',
        'min_sigma': 0.1
    }

    trained_model = train_pkd(
        model,
        train_sequences,
        train_configs,
        val_sequences,
        val_configs,
        num_epochs=10,
        batch_size=256,
        lr=1e-3,
        device=device,
        early_stopping_patience=3,
        model_config=model_config
    )

    print(f"Training complete. Best model saved to pkd_model.pt")

    return trained_model


# ============ Evaluation Functions ============
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


def evaluate_marginal_distribution(teacher_seq, student_seq, save_path='eval_marginal.png'):
    """Evaluate marginal distribution fidelity.

    Generates:
    - (a) CCDF comparison
    - (b) QQ plot
    - (c) Quantile error plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (a) CCDF
    teacher_sorted = np.sort(teacher_seq)
    student_sorted = np.sort(student_seq)
    teacher_ccdf = 1 - np.arange(len(teacher_sorted)) / len(teacher_sorted)
    student_ccdf = 1 - np.arange(len(student_sorted)) / len(student_sorted)

    axes[0].semilogy(teacher_sorted, teacher_ccdf, 'b-', label='Teacher', alpha=0.7)
    axes[0].semilogy(student_sorted, student_ccdf, 'r--', label='Student', alpha=0.7)
    axes[0].set_xlabel('log(SINR)')
    axes[0].set_ylabel('CCDF')
    axes[0].set_title('(a) Marginal CCDF')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # (b) QQ plot
    quantiles = np.linspace(0, 1, 100)
    teacher_q = np.quantile(teacher_seq, quantiles)
    student_q = np.quantile(student_seq, quantiles)

    axes[1].plot(teacher_q, student_q, 'o', alpha=0.5)
    axes[1].plot([teacher_q.min(), teacher_q.max()],
                 [teacher_q.min(), teacher_q.max()], 'k--', label='y=x')
    axes[1].set_xlabel('Teacher Quantiles')
    axes[1].set_ylabel('Student Quantiles')
    axes[1].set_title('(b) QQ Plot')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # (c) Quantile error
    quantile_error = student_q - teacher_q
    axes[2].plot(quantiles, quantile_error, 'g-', linewidth=2)
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[2].fill_between(quantiles, quantile_error, 0, alpha=0.3)
    axes[2].set_xlabel('Quantile α')
    axes[2].set_ylabel('Quantile Error')
    axes[2].set_title('(c) Quantile Error')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved marginal distribution evaluation to {save_path}")
    plt.close()

    # Compute KS statistic
    ks_stat, ks_pval = stats.ks_2samp(teacher_seq, student_seq)
    print(f"Kolmogorov-Smirnov test: statistic={ks_stat:.4f}, p-value={ks_pval:.4f}")

    return {'ks_stat': ks_stat, 'ks_pval': ks_pval, 'quantile_error': quantile_error}


def evaluate_temporal_dependence(teacher_seq, student_seq, save_path='eval_temporal.png'):
    """Evaluate temporal correlation structure.

    Generates:
    - (a) ACF comparison
    - (b) PSD comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # (a) ACF
    max_lag = min(50, len(teacher_seq) // 10)
    teacher_acf = compute_acf(teacher_seq, max_lag)
    student_acf = compute_acf(student_seq, max_lag)

    lags = np.arange(len(teacher_acf))
    axes[0].plot(lags, teacher_acf, 'b-o', label='Teacher', markersize=4)
    axes[0].plot(lags, student_acf, 'r--s', label='Student', markersize=3)
    axes[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF')
    axes[0].set_title('(a) Autocorrelation Function')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # (b) PSD
    teacher_freqs, teacher_psd = compute_psd(teacher_seq)
    student_freqs, student_psd = compute_psd(student_seq)

    axes[1].semilogy(teacher_freqs, teacher_psd, 'b-', label='Teacher', alpha=0.7)
    axes[1].semilogy(student_freqs, student_psd, 'r--', label='Student', alpha=0.7)
    axes[1].set_xlabel('Frequency')
    axes[1].set_ylabel('PSD')
    axes[1].set_title('(b) Power Spectral Density')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved temporal dependence evaluation to {save_path}")
    plt.close()

    # Compute ACF RMSE
    acf_rmse = np.sqrt(np.mean((teacher_acf[1:] - student_acf[1:])**2))
    print(f"ACF RMSE (excluding lag 0): {acf_rmse:.4f}")

    return {'acf_rmse': acf_rmse, 'teacher_acf': teacher_acf, 'student_acf': student_acf}


def evaluate_innovation_structure(model, inference, teacher_seq, config, device='cpu',
                                  save_path='eval_innovations.png'):
    """Evaluate innovation structure and PIT calibration.

    Generates:
    - Innovation diagnostics table
    - PIT histogram and ACF
    """
    # Generate student sequence with same config
    config_traj = [config] * len(teacher_seq)

    # Convert teacher sequence to log domain
    X_teacher = np.log(teacher_seq)

    # Get model predictions for teacher data
    model.eval()
    with torch.no_grad():
        # Build history windows
        ar_order = model.ar_order
        innovations = []
        pit_values = []

        for t in range(ar_order, len(X_teacher)):
            # Use .copy() to handle negative strides
            X_hist_array = X_teacher[t-ar_order:t][::-1].copy()
            X_hist = torch.tensor(X_hist_array, dtype=torch.float32).unsqueeze(0).to(device)
            X_t = torch.tensor(X_teacher[t], dtype=torch.float32).unsqueeze(0).to(device)

            # Prepare config
            config_dict = {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else torch.tensor([v]).to(device)
                          for k, v in config.items()}

            # Compute log likelihood and get parameters
            log_q, info = model.compute_log_likelihood(X_t, X_hist, config_dict)

            # Standardized innovation
            eps = info['eps'].cpu().numpy()[0]
            innovations.append(eps)

            # PIT: Transform innovation to uniform via standard normal CDF
            # For Gaussian innovation: eps ~ N(0, sigma^2), so z = eps/sigma ~ N(0,1)
            # For Flow innovation: would need to compute z via flow inverse
            innov_params = info['innov_params'].cpu().numpy()

            # Check if it's a scalar (Gaussian) or vector (Flow)
            # Gaussian: shape is () or (1,), Flow: shape is (K,) where K > 1
            if innov_params.shape == () or (innov_params.shape == (1,)):
                # Gaussian innovation: innov_params is scalar sigma
                sigma = innov_params.item() if innov_params.shape == () else innov_params[0]
                z = eps / sigma
            else:
                # Flow innovation: innov_params is vector psi
                # For proper PIT with flow, we'd need to compute the flow inverse
                # For now, assume unit variance as approximation
                z = eps

            u = stats.norm.cdf(z)
            pit_values.append(u)

    innovations = np.array(innovations)
    pit_values = np.array(pit_values)

    # Innovation diagnostics
    print("\n=== Innovation Diagnostics ===")
    lb_stat, lb_pval = ljung_box_test(innovations, lags=20)
    lb_stat_sq, lb_pval_sq = ljung_box_test(innovations**2, lags=20)

    print(f"Ljung-Box test (ε_t):     statistic={lb_stat:.2f}, p-value={lb_pval:.4f}")
    print(f"Ljung-Box test (ε_t²):    statistic={lb_stat_sq:.2f}, p-value={lb_pval_sq:.4f}")
    print(f"Mean of innovations:      {np.mean(innovations):.4f}")
    print(f"Variance of innovations:  {np.var(innovations):.4f}")

    innov_acf = compute_acf(innovations, max_lag=20)
    max_acf = np.max(np.abs(innov_acf[1:]))
    print(f"Max |ACF| (lags 1-20):    {max_acf:.4f}")

    # PIT evaluation
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # (a) PIT histogram
    axes[0].hist(pit_values, bins=20, density=True, alpha=0.7, edgecolor='black')
    axes[0].axhline(y=1.0, color='r', linestyle='--', label='Uniform(0,1)', linewidth=2)
    axes[0].set_xlabel('PIT value')
    axes[0].set_ylabel('Density')
    axes[0].set_title('(a) PIT Histogram')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # (b) ACF of centered PIT
    pit_centered = pit_values - 0.5
    pit_acf = compute_acf(pit_centered, max_lag=20)
    lags = np.arange(len(pit_acf))

    axes[1].stem(lags, pit_acf, basefmt=' ')
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1].axhline(y=1.96/np.sqrt(len(pit_values)), color='r', linestyle='--', alpha=0.5)
    axes[1].axhline(y=-1.96/np.sqrt(len(pit_values)), color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('ACF')
    axes[1].set_title('(b) ACF of Centered PIT')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved innovation evaluation to {save_path}")
    plt.close()

    # KS test for uniformity
    ks_stat, ks_pval = stats.kstest(pit_values, 'uniform')
    print(f"PIT uniformity KS test:   statistic={ks_stat:.4f}, p-value={ks_pval:.4f}")

    return {
        'lb_pval': lb_pval,
        'lb_pval_sq': lb_pval_sq,
        'mean_innov': np.mean(innovations),
        'var_innov': np.var(innovations),
        'max_acf': max_acf,
        'pit_ks_pval': ks_pval
    }


def example_evaluation():
    """Run comprehensive evaluation of student model fidelity."""
    print("=" * 50)
    print("PKD Model Fidelity Evaluation")
    print("=" * 50)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load('pkd_model.pt', map_location=device)

    print(f"\nLoaded checkpoint from pkd_model.pt")
    print(f"  Checkpoint keys: {list(checkpoint.keys())}")
    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch']+1} epochs")
    if 'val_loss' in checkpoint:
        print(f"  Best validation loss: {checkpoint['val_loss']:.4f}")
    if 'train_loss' in checkpoint:
        print(f"  Training loss at best epoch: {checkpoint['train_loss']:.4f}")

    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        # Add kappa_max if not present (for backward compatibility)
        if 'kappa_max' not in model_config:
            print("  Note: kappa_max not in checkpoint, using default 0.95")
            model_config['kappa_max'] = 0.95
        # Add innovation_type if not present (for backward compatibility)
        if 'innovation_type' not in model_config:
            print("  Note: innovation_type not in checkpoint, using default 'gaussian'")
            model_config['innovation_type'] = 'gaussian'
            model_config['min_sigma'] = 0.1
    else:
        print("  Warning: model_config not found in checkpoint, using defaults")
        model_config = {
            'num_channel_models': 5,
            'num_mcs': 10,
            'num_nss': 4,
            'ar_order': 10,
            'hidden_dim': 128,
            'kappa_max': 0.95,
            'innovation_type': 'gaussian',
            'min_sigma': 0.1
        }

    model = PKDModel(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Debug: Check model parameters to see if they're trained
    print("\nModel parameter diagnostics:")
    with torch.no_grad():
        # Check mean head output for a sample config
        sample_config = {
            'channel_model_id': torch.tensor([0], device=device),
            'N_t': torch.tensor([4], device=device),
            'N_r': torch.tensor([4], device=device),
            'BW': torch.tensor([20.0], device=device),
            'SNR_bar': torch.tensor([15.0], device=device),
            'MCS': torch.tensor([5], device=device),
            'N_ss': torch.tensor([2], device=device)
        }
        h = model.encode_config(sample_config)
        params = model.generate_params(h)
        print(f"  Sample mean (m): {params['m'].item():.4f}")
        print(f"  Sample AR coeffs (phi): {params['phi'][0, :5].cpu().numpy()}")
        print(f"  Sample AR offset (c): {params['c'].item():.4f}")

    # Create inference engine
    per_lut = AWGNPERLookup.create_dummy_lut(num_mcs=10)
    inference = PKDInference(model, per_lut, ar_order=model_config['ar_order'], device=device)

    # Generate teacher sequence (log-AR(1) process)
    def generate_teacher_sequence(length=2000, mu=2.0, phi=0.9, sigma=0.5):
        log_gamma = np.zeros(length)
        log_gamma[0] = np.random.randn() * sigma / np.sqrt(1 - phi**2) + mu
        for t in range(1, length):
            epsilon = np.random.randn() * sigma
            log_gamma[t] = mu + phi * (log_gamma[t-1] - mu) + epsilon
        return np.exp(log_gamma)

    teacher_seq = generate_teacher_sequence()

    # Generate student sequence
    config = {
        'channel_model_id': torch.tensor(0),
        'N_t': torch.tensor(4),
        'N_r': torch.tensor(4),
        'BW': torch.tensor(20.0),
        'SNR_bar': torch.tensor(15.0),
        'MCS': torch.tensor(5),
        'N_ss': torch.tensor(2)
    }

    config_traj = [config] * len(teacher_seq)
    student_results = inference.run_sequence(config_traj)
    student_seq = student_results['gamma_eff']

    # CRITICAL VALIDATION: Check student sequence validity
    print(f"\nStudent sequence validation:")
    student_arr = np.asarray(student_seq)

    # Check for non-finite values
    finite_mask = np.isfinite(student_arr)
    if not np.all(finite_mask):
        print(f"  ERROR: Student sequence contains non-finite values!")
        print(f"  Finite ratio: {finite_mask.mean():.4f}")
        print(f"  NaN count: {np.isnan(student_arr).sum()}")
        print(f"  Inf count: {np.isinf(student_arr).sum()}")
        raise ValueError("Student sequence contains non-finite values")

    # Check for non-positive values
    if not np.all(student_arr > 0):
        print(f"  ERROR: Student sequence contains non-positive values!")
        print(f"  Positive ratio: {(student_arr > 0).mean():.4f}")
        print(f"  Min value: {student_arr.min():.4e}")
        raise ValueError("Student sequence contains non-positive gamma_eff")

    # Diagnostics: check if student sequence is degenerate
    student_unique = len(np.unique(np.round(student_seq, 6)))
    print(f"  ✓ All values finite and positive")
    print(f"  Unique values: {student_unique} / {len(student_seq)}")
    if student_unique < len(student_seq) * 0.9:
        print(f"  WARNING: Low diversity! Expected ~{len(student_seq)}, got {student_unique}")
    print(f"  Min: {np.min(student_seq):.4f}, Max: {np.max(student_seq):.4f}")
    print(f"  Mean: {np.mean(student_seq):.4f}, Std: {np.std(student_seq):.4f}")

    # Convert to log domain for analysis
    # Note: validation above ensures all values are finite and positive
    teacher_log = np.log(teacher_seq)
    student_log = np.log(student_seq)

    print("\n--- 1. Marginal Distribution Fidelity ---")
    marginal_metrics = evaluate_marginal_distribution(teacher_log, student_log)

    print("\n--- 2. Temporal Dependence ---")
    temporal_metrics = evaluate_temporal_dependence(teacher_log, student_log)

    print("\n--- 3. Innovation Structure ---")
    innovation_metrics = evaluate_innovation_structure(model, inference, teacher_seq, config, device)

    print("\n" + "=" * 50)
    print("Evaluation complete! Check generated PNG files.")
    print("=" * 50)

    return {
        'marginal': marginal_metrics,
        'temporal': temporal_metrics,
        'innovation': innovation_metrics
    }


if __name__ == '__main__':
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else 'eval'

    if mode == 'train':
        # Run training example
        print("=" * 50)
        print("PKD Example - Training")
        print("=" * 50)
        trained_model = example_training()

    else:
        # Run evaluation (default mode)
        example_evaluation()
