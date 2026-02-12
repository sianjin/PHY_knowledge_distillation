"""Evaluation functions for PKD model (PKD v1: Gaussian innovation)."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from .utils import compute_acf, compute_psd, ljung_box_test

def evaluate_marginal_distribution(teacher_seq, student_seq, save_path='eval_marginal.png'):
    """Evaluate marginal distribution fidelity (PKD v1: Gaussian innovation).

    PKD v1: Restricts quantile analysis to α ∈ [0.05, 0.95] to avoid
    over-penalizing extreme tail mismatch.

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

    # (b) QQ plot - restrict to [0.05, 0.95]
    quantiles = np.linspace(0.05, 0.95, 100)
    teacher_q = np.quantile(teacher_seq, quantiles)
    student_q = np.quantile(student_seq, quantiles)

    axes[1].plot(teacher_q, student_q, 'o', alpha=0.5)
    axes[1].plot([teacher_q.min(), teacher_q.max()],
                 [teacher_q.min(), teacher_q.max()], 'k--', label='y=x')
    axes[1].set_xlabel('Teacher Quantiles')
    axes[1].set_ylabel('Student Quantiles')
    axes[1].set_title('(b) QQ Plot (α ∈ [0.05, 0.95])')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # (c) Quantile error
    quantile_error = student_q - teacher_q
    axes[2].plot(quantiles, quantile_error, 'g-', linewidth=2)
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[2].fill_between(quantiles, quantile_error, 0, alpha=0.3)
    axes[2].set_xlabel('Quantile α')
    axes[2].set_ylabel('Quantile Error')
    axes[2].set_title('(c) Quantile Error (α ∈ [0.05, 0.95])')
    axes[2].grid(True, alpha=0.3)

    # Add super-title
    fig.suptitle('Marginal Distribution Evaluation (Gaussian Innovation)',
                 fontsize=12, fontweight='bold', y=1.00)

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
    """Evaluate innovation structure and PIT calibration (PKD v1: Gaussian innovation).

    PKD v1 uses Gaussian innovations only. This function:
    - Computes standardized innovations z_t = ε_t / σ_t
    - Tests z_t for independence via Ljung-Box
    - Evaluates PIT calibration using Gaussian CDF

    Args:
        teacher_seq: Teacher sequence in natural log scale (already converted from dB)

    Generates:
    - Innovation diagnostics table
    - PIT histogram and ACF
    """
    # Teacher sequence is already in natural log scale (converted in data_loader.py)
    X_teacher = teacher_seq

    # Get model predictions for teacher data (teacher-forced)
    model.eval()
    with torch.no_grad():
        # Build history windows
        ar_order = model.ar_order
        standardized_innovations = []  # z_t = ε_t / σ_t
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

            # Extract raw innovation and sigma
            eps = info['eps'].cpu().numpy()[0]  # ε_t (raw innovation)
            innov_params = info['innov_params']

            # PKD v1: Gaussian innovation only
            # innov_params is scalar sigma (standard deviation)
            if isinstance(innov_params, dict):
                raise ValueError("PKD v1 supports Gaussian innovation only. Found SGN parameters.")

            innov_params_np = innov_params.cpu().numpy()
            sigma = innov_params_np.item() if innov_params_np.ndim == 0 else innov_params_np[0]

            # Standardize: z_t = ε_t / σ_t ~ N(0, 1)
            z = eps / sigma
            standardized_innovations.append(z)

            # PIT: u_t = Φ(z_t) should be Uniform(0,1) if model is well-calibrated
            u = stats.norm.cdf(z)
            pit_values.append(u)

    standardized_innovations = np.array(standardized_innovations)
    pit_values = np.array(pit_values)

    # Innovation diagnostics (on STANDARDIZED innovations)
    print(f"\n=== Innovation Diagnostics (Gaussian) ===")
    lb_stat, lb_pval = ljung_box_test(standardized_innovations, lags=20)
    lb_stat_sq, lb_pval_sq = ljung_box_test(standardized_innovations**2, lags=20)

    print(f"Ljung-Box test (z_t):     statistic={lb_stat:.2f}, p-value={lb_pval:.4f}")
    print(f"Ljung-Box test (z_t²):    statistic={lb_stat_sq:.2f}, p-value={lb_pval_sq:.4f}")
    print(f"Mean of standardized innovations:      {np.mean(standardized_innovations):.4f}")
    print(f"Variance of standardized innovations:  {np.var(standardized_innovations):.4f}")

    innov_acf = compute_acf(standardized_innovations, max_lag=20)
    max_acf = np.max(np.abs(innov_acf[1:]))
    print(f"Max |ACF| (lags 1-20):    {max_acf:.4f}")

    # PIT evaluation
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # (a) PIT histogram
    axes[0].hist(pit_values, bins=20, density=True, alpha=0.7, edgecolor='black')
    axes[0].axhline(y=1.0, color='r', linestyle='--', label='Uniform(0,1)', linewidth=2)
    axes[0].set_xlabel('PIT value')
    axes[0].set_ylabel('Density')
    axes[0].set_title('(a) PIT Histogram (Gaussian Innovation)')
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
    axes[1].set_title('(b) ACF of Centered PIT (Gaussian)')
    axes[1].grid(True, alpha=0.3)

    # Add overall title
    fig.suptitle('Innovation Structure Evaluation (Gaussian Innovation)',
                 fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved innovation evaluation to {save_path}")
    plt.close()

    # KS test for uniformity
    ks_stat, ks_pval = stats.kstest(pit_values, 'uniform')
    print(f"PIT uniformity KS test (Gaussian CDF): statistic={ks_stat:.4f}, p-value={ks_pval:.4f}")

    return {
        'lb_pval': lb_pval,
        'lb_pval_sq': lb_pval_sq,
        'mean_innov': np.mean(standardized_innovations),
        'var_innov': np.var(standardized_innovations),
        'max_acf': max_acf,
        'pit_ks_pval': ks_pval
    }

