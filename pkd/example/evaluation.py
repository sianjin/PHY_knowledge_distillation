"""Evaluation functions for PKD model (PKD v1: Gaussian innovation)."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from .utils import compute_acf, compute_psd, ljung_box_test

def evaluate_marginal_distribution(teacher_seq, student_seq, save_prefix='figures/eval_marginal'):
    """Evaluate marginal distribution fidelity (PKD v1: Gaussian innovation).

    PKD v1: Restricts quantile analysis to α ∈ [0.05, 0.95] to avoid
    over-penalizing extreme tail mismatch.

    Generates 3 separate files:
    - eval_marginal_ccdf.png: CCDF comparison
    - eval_marginal_qq.png: QQ plot
    - eval_marginal_quantile_error.png: Quantile error plot
    """
    # Prepare data for all plots
    teacher_sorted = np.sort(teacher_seq)
    student_sorted = np.sort(student_seq)
    teacher_ccdf = 1 - np.arange(len(teacher_sorted)) / len(teacher_sorted)
    student_ccdf = 1 - np.arange(len(student_sorted)) / len(student_sorted)

    quantiles = np.linspace(0.05, 0.95, 100)
    teacher_q = np.quantile(teacher_seq, quantiles)
    student_q = np.quantile(student_seq, quantiles)
    quantile_error = student_q - teacher_q

    # Plot 1: CCDF
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.semilogy(teacher_sorted, teacher_ccdf, 'b-', label='Teacher', alpha=0.7)
    ax.semilogy(student_sorted, student_ccdf, 'r--', label='Student', alpha=0.7)
    ax.set_xlabel('log(SINR)')
    ax.set_ylabel('CCDF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_ccdf.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_ccdf.png")
    plt.close()

    # Plot 2: QQ plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(teacher_q, student_q, 'o', alpha=0.5)
    ax.plot([teacher_q.min(), teacher_q.max()],
            [teacher_q.min(), teacher_q.max()], 'k--', label='y=x')
    ax.set_xlabel('Teacher Quantiles')
    ax.set_ylabel('Student Quantiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_qq.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_qq.png")
    plt.close()

    # Plot 3: Quantile error
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(quantiles, quantile_error, 'g-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.fill_between(quantiles, quantile_error, 0, alpha=0.3)
    ax.set_xlabel('Quantile α')
    ax.set_ylabel('Quantile Error')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_quantile_error.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_quantile_error.png")
    plt.close()

    # Compute KS statistic
    ks_stat, ks_pval = stats.ks_2samp(teacher_seq, student_seq)
    print(f"Kolmogorov-Smirnov test: statistic={ks_stat:.4f}, p-value={ks_pval:.4f}")

    return {'ks_stat': ks_stat, 'ks_pval': ks_pval, 'quantile_error': quantile_error}


def evaluate_temporal_dependence(teacher_seq, student_seq, save_prefix='figures/eval_temporal'):
    """Evaluate temporal correlation structure.

    Generates 2 separate files:
    - eval_temporal_acf.png: ACF comparison
    - eval_temporal_psd.png: PSD comparison
    """
    # Prepare data
    max_lag = min(50, len(teacher_seq) // 10)
    teacher_acf = compute_acf(teacher_seq, max_lag)
    student_acf = compute_acf(student_seq, max_lag)
    lags = np.arange(len(teacher_acf))

    teacher_freqs, teacher_psd = compute_psd(teacher_seq)
    student_freqs, student_psd = compute_psd(student_seq)

    # Plot 1: ACF
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(lags, teacher_acf, 'b-o', label='Teacher', markersize=4)
    ax.plot(lags, student_acf, 'r--s', label='Student', markersize=3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_acf.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_acf.png")
    plt.close()

    # Plot 2: PSD
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.semilogy(teacher_freqs, teacher_psd, 'b-', label='Teacher', alpha=0.7)
    ax.semilogy(student_freqs, student_psd, 'r--', label='Student', alpha=0.7)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('PSD')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_psd.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_psd.png")
    plt.close()

    # Compute ACF RMSE
    acf_rmse = np.sqrt(np.mean((teacher_acf[1:] - student_acf[1:])**2))
    print(f"ACF RMSE (excluding lag 0): {acf_rmse:.4f}")

    return {'acf_rmse': acf_rmse, 'teacher_acf': teacher_acf, 'student_acf': student_acf}


def evaluate_innovation_structure(model, inference, teacher_seq, config, device='cpu',
                                  save_prefix='figures/eval_innovations'):
    """Evaluate innovation structure and PIT calibration (PKD v1: Gaussian innovation).

    PKD v1 uses Gaussian innovations only. This function:
    - Computes standardized innovations z_t = ε_t / σ_t
    - Tests z_t for independence via Ljung-Box
    - Evaluates PIT calibration using Gaussian CDF

    Args:
        teacher_seq: Teacher sequence in natural log scale (already converted from dB)

    Generates 2 separate files:
    - eval_innovations_pit_hist.png: PIT histogram
    - eval_innovations_pit_acf.png: ACF of centered PIT
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

    # PIT evaluation - Plot 1: Histogram
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(pit_values, bins=20, density=True, alpha=0.7, edgecolor='black')
    ax.axhline(y=1.0, color='r', linestyle='--', label='Uniform(0,1)', linewidth=2)
    ax.set_xlabel('PIT value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_pit_hist.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_pit_hist.png")
    plt.close()

    # Plot 2: ACF of centered PIT
    pit_centered = pit_values - 0.5
    pit_acf = compute_acf(pit_centered, max_lag=20)
    lags = np.arange(len(pit_acf))

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.stem(lags, pit_acf, basefmt=' ')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axhline(y=1.96/np.sqrt(len(pit_values)), color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=-1.96/np.sqrt(len(pit_values)), color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_pit_acf.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_pit_acf.png")
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


def evaluate_teacher_baseline_ar(teacher_seq, ar_order=5, save_prefix='figures/eval_teacher_baseline'):
    """
    Fit classical AR(p) + constant σ to teacher sequence as baseline diagnostic.

    This checks if Ljung-Box rejection on z_t² is inherent to teacher data
    or a student model issue.

    Args:
        teacher_seq: Teacher sequence in natural log scale
        ar_order: AR order for baseline (default: 5)
        save_prefix: Output file prefix (without extension)

    Generates 4 separate files:
    - eval_teacher_baseline_residuals.png: Standardized residuals time series
    - eval_teacher_baseline_acf_zt.png: ACF of z_t
    - eval_teacher_baseline_acf_zt2.png: ACF of z_t²
    - eval_teacher_baseline_pit_hist.png: PIT histogram

    Returns:
        dict with baseline diagnostics
    """
    print(f"\n{'='*60}")
    print(f"BASELINE DIAGNOSTIC: Classical AR({ar_order}) on Teacher Data")
    print(f"{'='*60}")

    X = teacher_seq
    T = len(X)

    # Build design matrix for AR(p) regression
    # X_t = c + phi_1*X_{t-1} + ... + phi_p*X_{t-p} + eps_t
    X_design = []
    y = []

    for t in range(ar_order, T):
        # History: [X_{t-1}, ..., X_{t-p}]
        X_hist = X[t-ar_order:t][::-1]  # Reverse to get [X_{t-1}, ..., X_{t-p}]
        X_design.append(X_hist)
        y.append(X[t])

    X_design = np.array(X_design)  # (T-p, p)
    y = np.array(y)  # (T-p,)

    # Fit AR model with OLS using numpy
    # Add intercept column to design matrix
    X_design_with_intercept = np.column_stack([np.ones(len(X_design)), X_design])  # (T-p, p+1)

    # Solve normal equations: (X^T X) beta = X^T y
    # beta = [c, phi_1, ..., phi_p]
    beta = np.linalg.lstsq(X_design_with_intercept, y, rcond=None)[0]

    c = beta[0]  # Intercept
    phi = beta[1:]  # AR coefficients (p,)

    # Compute residuals
    y_pred = X_design_with_intercept @ beta
    residuals = y - y_pred  # eps_t

    # Estimate constant sigma (MLE for Gaussian)
    sigma_hat = np.std(residuals, ddof=ar_order+1)  # ddof accounts for p+1 parameters

    # Standardized residuals
    z = residuals / sigma_hat

    print(f"\nFitted AR({ar_order}) parameters:")
    print(f"  Intercept (c): {c:.4f}")
    print(f"  AR coefficients (phi): {phi}")
    print(f"  Residual std (σ̂): {sigma_hat:.4f}")
    print(f"  Sum of AR coeffs: {np.sum(phi):.4f}")

    # Diagnostics on standardized residuals
    print(f"\n=== Teacher Baseline Diagnostics ===")
    lb_stat, lb_pval = ljung_box_test(z, lags=20)
    lb_stat_sq, lb_pval_sq = ljung_box_test(z**2, lags=20)

    print(f"Ljung-Box test (z_t):     statistic={lb_stat:.2f}, p-value={lb_pval:.4f}")
    print(f"Ljung-Box test (z_t²):    statistic={lb_stat_sq:.2f}, p-value={lb_pval_sq:.4f}")
    print(f"Mean of z_t:              {np.mean(z):.4f}")
    print(f"Variance of z_t:          {np.var(z):.4f}")

    z_acf = compute_acf(z, max_lag=20)
    z_sq_acf = compute_acf(z**2, max_lag=20)
    max_acf_z = np.max(np.abs(z_acf[1:]))
    max_acf_z_sq = np.max(np.abs(z_sq_acf[1:]))
    print(f"Max |ACF(z_t)| (lags 1-20):   {max_acf_z:.4f}")
    print(f"Max |ACF(z_t²)| (lags 1-20):  {max_acf_z_sq:.4f}")

    # PIT test
    pit_values = stats.norm.cdf(z)
    ks_stat_pit, ks_pval_pit = stats.kstest(pit_values, 'uniform')
    print(f"PIT uniformity KS test:   statistic={ks_stat_pit:.4f}, p-value={ks_pval_pit:.4f}")

    # Plot diagnostics - 4 separate files
    lags = np.arange(len(z_acf))

    # Plot 1: Standardized residuals z_t
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(z, linewidth=0.5, alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(y=2, color='r', linestyle='--', alpha=0.3)
    ax.axhline(y=-2, color='r', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time')
    ax.set_ylabel('z_t')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_residuals.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_residuals.png")
    plt.close()

    # Plot 2: ACF of z_t
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.stem(lags, z_acf, basefmt=' ')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axhline(y=1.96/np.sqrt(len(z)), color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=-1.96/np.sqrt(len(z)), color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_acf_zt.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_acf_zt.png")
    plt.close()

    # Plot 3: ACF of z_t²
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.stem(lags, z_sq_acf, basefmt=' ')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axhline(y=1.96/np.sqrt(len(z)), color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=-1.96/np.sqrt(len(z)), color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_acf_zt2.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_acf_zt2.png")
    plt.close()

    # Plot 4: PIT histogram
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(pit_values, bins=20, density=True, alpha=0.7, edgecolor='black')
    ax.axhline(y=1.0, color='r', linestyle='--', label='Uniform(0,1)', linewidth=2)
    ax.set_xlabel('PIT value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_pit_hist.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_pit_hist.png")
    plt.close()

    return {
        'ar_order': ar_order,
        'phi': phi,
        'c': c,
        'sigma': sigma_hat,
        'lb_pval_z': lb_pval,
        'lb_pval_z_sq': lb_pval_sq,
        'mean_z': np.mean(z),
        'var_z': np.var(z),
        'max_acf_z': max_acf_z,
        'max_acf_z_sq': max_acf_z_sq,
        'pit_ks_pval': ks_pval_pit
    }

