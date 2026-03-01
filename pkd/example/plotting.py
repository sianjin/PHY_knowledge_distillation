"""Plotting functions for PKD test set evaluation (PKD v1: Gaussian innovation)."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from collections import defaultdict

from .utils import compute_acf, compute_psd, ljung_box_test

def evaluate_test_set(model, test_sequences, test_configs, device='cpu'):
    """Evaluate model on entire test set and compute aggregate metrics.

    Args:
        model: Trained PKDModel
        test_sequences: List of test gamma_eff sequences (in natural log scale)
        test_configs: List of test config dicts
        device: Device to run evaluation on

    Returns:
        Dictionary of aggregate metrics across all test sequences
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on {len(test_sequences)} test sequences")
    print(f"{'='*60}")

    from train import PKDDataset, collate_fn
    from torch.utils.data import DataLoader

    # Create test dataset (DataLoader handles batching automatically)
    test_dataset = PKDDataset(test_sequences, test_configs, model.ar_order)
    test_loader = DataLoader(test_dataset, batch_size=256,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=4)

    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_log_q = []

    print("\nComputing test set metrics...")
    with torch.no_grad():
        for X_t, X_hist, config_dict in tqdm(test_loader, desc="Evaluating"):
            X_t = X_t.to(device)
            X_hist = X_hist.to(device)
            config_dict = {k: v.to(device) for k, v in config_dict.items()}

            log_q, _ = model.compute_log_likelihood(X_t, X_hist, config_dict)

            # Check for non-finite values
            if torch.isfinite(log_q).all():
                batch_size = X_t.shape[0]
                total_loss += (-log_q.mean().item()) * batch_size
                total_samples += batch_size
                all_log_q.extend(log_q.cpu().numpy().tolist())

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    avg_log_likelihood = np.mean(all_log_q)

    print(f"\n{'='*60}")
    print(f"Test Set Results")
    print(f"{'='*60}")
    print(f"Number of test sequences: {len(test_sequences)}")
    print(f"Total test samples evaluated: {total_samples:,}")
    print(f"Average test loss (NLL): {avg_loss:.4f}")
    print(f"Average log-likelihood: {avg_log_likelihood:.4f}")
    print(f"{'='*60}")

    return {
        'test_loss': avg_loss,
        'avg_log_likelihood': avg_log_likelihood,
        'num_sequences': len(test_sequences),
        'num_samples': total_samples
    }


def generate_figure1_per_mcs_metrics(model, test_sequences, test_configs, device='cpu',
                                      save_path='test_metrics', slice_label=None):
    """Generate Figure 1: Per-MCS metrics vs SNR (6 separate files) - Gaussian innovation.

    PKD v1: All metrics assume Gaussian innovation N(0, σ²).

    Output files:
    - test_metrics_pit_pass_rate.png: PIT pass rate vs SNR (teacher-forced, Gaussian CDF)
    - test_metrics_lb_pass_rate_zt.png: Ljung-Box pass rate (z_t) vs SNR (teacher-forced)
    - test_metrics_lb_pass_rate_zt2.png: Ljung-Box pass rate (z_t²) vs SNR (teacher-forced, volatility)
    - test_metrics_acf_rmse.png: Median ACF RMSE vs SNR (free-running)
    - test_metrics_psd_rmse.png: Median PSD RMSE vs SNR (free-running)
    - test_metrics_ks_stat.png: Median KS statistic vs SNR (free-running)

    Args:
        model: Trained PKD model
        test_sequences: Filtered test sequences (single config slice)
        test_configs: Filtered test configs (single config slice)
        device: Computation device
        save_path: Output file prefix (without extension)
        slice_label: Configuration slice label for plot title
    """
    from collections import defaultdict

    print("\n" + "="*60)
    print("Generating Figure 1: Per-MCS Metrics vs SNR (Gaussian Innovation)")
    if slice_label:
        print(f"Configuration slice: {slice_label}")
    print("="*60)

    # Group sequences by (MCS, SNR)
    grouped = defaultdict(list)
    for i, config in enumerate(test_configs):
        mcs = config['MCS']
        snr = int(config['SNR_bar'])
        grouped[(mcs, snr)].append(i)

    # Storage for metrics
    metrics_by_mcs_snr = defaultdict(lambda: {
        'pit_pass': [], 'lb_pass': [], 'lb_pass_sq': [], 'acf_rmse': [], 'psd_rmse': [], 'ks_stat': []
    })

    model.eval()
    ar_order = model.ar_order

    # Create inference engine for free-running
    from per_lut import AWGNPERLookup
    from infer import PKDInference
    # Load LDPC PER LUT (embedded data)
    per_lut = AWGNPERLookup.load_ldpc_lut()
    inference = PKDInference(model, per_lut, ar_order=ar_order, device=device)

    print("\nProcessing sequences by (MCS, SNR)...")
    for (mcs, snr), seq_indices in tqdm(sorted(grouped.items()), desc="(MCS, SNR) groups"):
        for seq_idx in seq_indices:
            teacher_seq = test_sequences[seq_idx]
            config = test_configs[seq_idx]
            X_teacher = teacher_seq  # Already in natural log scale  # Log domain

            # === Teacher-Forced Metrics (PKD v1: Gaussian innovation) ===
            # OPTIMIZATION: Process all timesteps in a single batched forward pass
            # instead of looping (25x faster: ~990 forward passes → 1 forward pass)
            with torch.no_grad():
                T = len(X_teacher)
                num_samples = T - ar_order

                # Pre-allocate arrays
                X_t_batch = np.zeros(num_samples, dtype=np.float32)
                X_hist_batch = np.zeros((num_samples, ar_order), dtype=np.float32)

                # Build batch of all timesteps at once
                for i, t in enumerate(range(ar_order, T)):
                    X_t_batch[i] = X_teacher[t]
                    X_hist_batch[i] = X_teacher[t-ar_order:t][::-1]

                # Convert to tensors once
                X_t_tensor = torch.tensor(X_t_batch, dtype=torch.float32).to(device)
                X_hist_tensor = torch.tensor(X_hist_batch, dtype=torch.float32).to(device)

                # Prepare config dict once (broadcast to all timesteps)
                config_dict = {}
                for k, v in config.items():
                    if isinstance(v, (int, float)):
                        config_dict[k] = torch.full((num_samples,), v, dtype=torch.float32 if isinstance(v, float) else torch.long).to(device)
                    else:
                        config_dict[k] = torch.tensor([v] * num_samples).to(device)

                # Single batched forward pass
                log_q, info = model.compute_log_likelihood(X_t_tensor, X_hist_tensor, config_dict)

                # Extract innovations and sigma
                eps_batch = info['eps'].cpu().numpy()  # (num_samples,)
                innov_params = info['innov_params']

                # PKD v1: Gaussian innovation only
                if isinstance(innov_params, dict):
                    raise ValueError("PKD v1 supports Gaussian innovation only.")

                sigma_batch = innov_params.cpu().numpy()  # (num_samples,)

                # Standardize: z_t = ε_t / σ_t ~ N(0, 1)
                standardized_innovations = eps_batch / sigma_batch

                # PIT: u_t = Φ(z_t) ~ Uniform(0,1) if calibrated
                pit_values_seq = stats.norm.cdf(standardized_innovations)

                # PIT test
                ks_stat_pit, p_pit = stats.kstest(pit_values_seq, 'uniform')
                pit_pass = 1 if p_pit > 0.05 else 0

                # Ljung-Box test (on standardized innovations z_t)
                lb_stat, p_lb = ljung_box_test(standardized_innovations, lags=20)
                lb_pass = 1 if p_lb > 0.05 else 0

                # Ljung-Box test (on squared standardized innovations z_t²) for volatility clustering
                lb_stat_sq, p_lb_sq = ljung_box_test(standardized_innovations**2, lags=20)
                lb_pass_sq = 1 if p_lb_sq > 0.05 else 0

            # === Free-Running Metrics ===
            # Generate student sequence
            config_traj = [config] * len(teacher_seq)
            student_results = inference.run_sequence(config_traj)
            student_seq_db = np.array(student_results['gamma_eff'])  # Inference outputs dB scale

            # Convert from dB to natural log scale
            student_seq = student_seq_db * np.log(10) / 10

            if np.all(np.isfinite(student_seq)):
                X_student = student_seq  # Now in natural log scale

                # ACF RMSE
                teacher_acf = compute_acf(X_teacher, max_lag=50)
                student_acf = compute_acf(X_student, max_lag=50)
                acf_rmse = np.sqrt(np.mean((teacher_acf[1:] - student_acf[1:])**2))

                # PSD RMSE
                teacher_freqs, teacher_psd = compute_psd(X_teacher)
                student_freqs, student_psd = compute_psd(X_student)
                psd_rmse = np.sqrt(np.mean((teacher_psd - student_psd)**2))

                # KS test
                ks_stat_marg, _ = stats.ks_2samp(X_teacher, X_student)
            else:
                acf_rmse = np.nan
                psd_rmse = np.nan
                ks_stat_marg = np.nan

            # Store metrics
            metrics_by_mcs_snr[(mcs, snr)]['pit_pass'].append(pit_pass)
            metrics_by_mcs_snr[(mcs, snr)]['lb_pass'].append(lb_pass)
            metrics_by_mcs_snr[(mcs, snr)]['lb_pass_sq'].append(lb_pass_sq)
            metrics_by_mcs_snr[(mcs, snr)]['acf_rmse'].append(acf_rmse)
            metrics_by_mcs_snr[(mcs, snr)]['psd_rmse'].append(psd_rmse)
            metrics_by_mcs_snr[(mcs, snr)]['ks_stat'].append(ks_stat_marg)

    # Aggregate metrics
    results = defaultdict(lambda: {'snr': [], 'pit_rate': [], 'lb_rate': [], 'lb_rate_sq': [],
                                    'acf_rmse_med': [], 'psd_rmse_med': [], 'ks_med': []})

    for (mcs, snr), metrics in sorted(metrics_by_mcs_snr.items()):
        pit_rate = np.mean(metrics['pit_pass'])
        lb_rate = np.mean(metrics['lb_pass'])
        lb_rate_sq = np.mean(metrics['lb_pass_sq'])
        acf_rmse_med = np.nanmedian(metrics['acf_rmse'])
        psd_rmse_med = np.nanmedian(metrics['psd_rmse'])
        ks_med = np.nanmedian(metrics['ks_stat'])

        results[mcs]['snr'].append(snr)
        results[mcs]['pit_rate'].append(pit_rate)
        results[mcs]['lb_rate'].append(lb_rate)
        results[mcs]['lb_rate_sq'].append(lb_rate_sq)
        results[mcs]['acf_rmse_med'].append(acf_rmse_med)
        results[mcs]['psd_rmse_med'].append(psd_rmse_med)
        results[mcs]['ks_med'].append(ks_med)

    # Plot each metric as a separate figure
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # 1. PIT Pass Rate
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for mcs in sorted(results.keys()):
        data = results[mcs]
        snr = np.array(data['snr'])
        ax.plot(snr, data['pit_rate'], 'o-', label=f'MCS {mcs}', color=colors[mcs])
    ax.set_ylabel('PIT Pass Rate')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylim([0, 1.05])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}_pit_pass_rate.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}_pit_pass_rate.png")
    plt.close()

    # 2. Ljung-Box Pass Rate (z_t)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for mcs in sorted(results.keys()):
        data = results[mcs]
        snr = np.array(data['snr'])
        ax.plot(snr, data['lb_rate'], 'o-', label=f'MCS {mcs}', color=colors[mcs])
    ax.set_ylabel('Ljung-Box Pass Rate (z_t)')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylim([0, 1.05])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}_lb_pass_rate_zt.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}_lb_pass_rate_zt.png")
    plt.close()

    # 3. Ljung-Box Pass Rate (z_t²) - volatility clustering
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for mcs in sorted(results.keys()):
        data = results[mcs]
        snr = np.array(data['snr'])
        ax.plot(snr, data['lb_rate_sq'], 'o-', label=f'MCS {mcs}', color=colors[mcs])
    ax.set_ylabel('Ljung-Box Pass Rate (z_t²)')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylim([0, 1.05])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}_lb_pass_rate_zt2.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}_lb_pass_rate_zt2.png")
    plt.close()

    # 4. ACF RMSE
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for mcs in sorted(results.keys()):
        data = results[mcs]
        snr = np.array(data['snr'])
        ax.plot(snr, data['acf_rmse_med'], 'o-', label=f'MCS {mcs}', color=colors[mcs])
    ax.set_ylabel('Median ACF RMSE')
    ax.set_xlabel('SNR (dB)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}_acf_rmse.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}_acf_rmse.png")
    plt.close()

    # 5. PSD RMSE
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for mcs in sorted(results.keys()):
        data = results[mcs]
        snr = np.array(data['snr'])
        ax.plot(snr, data['psd_rmse_med'], 'o-', label=f'MCS {mcs}', color=colors[mcs])
    ax.set_ylabel('Median PSD RMSE')
    ax.set_xlabel('SNR (dB)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}_psd_rmse.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}_psd_rmse.png")
    plt.close()

    # 6. KS Statistic
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for mcs in sorted(results.keys()):
        data = results[mcs]
        snr = np.array(data['snr'])
        ax.plot(snr, data['ks_med'], 'o-', label=f'MCS {mcs}', color=colors[mcs])
    ax.set_ylabel('Median KS Statistic')
    ax.set_xlabel('SNR (dB)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}_ks_stat.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}_ks_stat.png")
    plt.close()

    return results


def generate_figure2_quantile_error(model, test_sequences, test_configs, device='cpu',
                                     save_path='fig2_quantile_error.png', slice_label=None):
    """Generate Figure 2: Aggregated quantile error curve (per MCS, pooled over SNR).

    PKD v1: Restricts to α ∈ [0.05, 0.95] to avoid over-penalizing extreme tail mismatch.

    Args:
        model: Trained PKD model
        test_sequences: Filtered test sequences (single config slice)
        test_configs: Filtered test configs (single config slice)
        device: Computation device
        save_path: Output file path
        slice_label: Configuration slice label for plot title
    """
    print("\n" + "="*60)
    print("Generating Figure 2: Quantile Error Curves (Gaussian Innovation)")
    if slice_label:
        print(f"Configuration slice: {slice_label}")
    print("="*60)

    # Group by MCS (pool over SNR)
    grouped_by_mcs = defaultdict(list)
    for i, config in enumerate(test_configs):
        mcs = config['MCS']
        grouped_by_mcs[mcs].append(i)

    # Create inference engine
    from per_lut import AWGNPERLookup
    from infer import PKDInference
    # Load LDPC PER LUT (embedded data)
    per_lut = AWGNPERLookup.load_ldpc_lut()
    inference = PKDInference(model, per_lut, ar_order=model.ar_order, device=device)

    # PKD v1: Restrict to [0.05, 0.95] to avoid extreme tail over-penalization
    quantile_levels = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])

    results = {}

    print("\nProcessing sequences by MCS...")
    for mcs in sorted(grouped_by_mcs.keys()):
        print(f"\nMCS {mcs}: {len(grouped_by_mcs[mcs])} sequences")
        seq_indices = grouped_by_mcs[mcs]

        quantile_errors = []

        for seq_idx in tqdm(seq_indices, desc=f"MCS {mcs}"):
            teacher_seq = test_sequences[seq_idx]
            config = test_configs[seq_idx]
            X_teacher = teacher_seq  # Already in natural log scale

            # Generate student
            config_traj = [config] * len(teacher_seq)
            student_results = inference.run_sequence(config_traj)
            student_seq_db = np.array(student_results['gamma_eff'])  # Inference outputs dB scale

            # Convert from dB to natural log scale
            student_seq = student_seq_db * np.log(10) / 10

            if np.all(np.isfinite(student_seq)):
                X_student = student_seq  # Now in natural log scale

                # Compute quantiles
                q_teacher = np.quantile(X_teacher, quantile_levels)
                q_student = np.quantile(X_student, quantile_levels)

                # Quantile error
                q_error = q_student - q_teacher
                quantile_errors.append(q_error)

        quantile_errors = np.array(quantile_errors)

        # Aggregate: median and 10-90 percentiles
        q_error_median = np.median(quantile_errors, axis=0)
        q_error_10 = np.percentile(quantile_errors, 10, axis=0)
        q_error_90 = np.percentile(quantile_errors, 90, axis=0)

        results[mcs] = {
            'median': q_error_median,
            'p10': q_error_10,
            'p90': q_error_90
        }

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for mcs in sorted(results.keys()):
        data = results[mcs]
        ax.plot(quantile_levels, data['median'], 'o-', label=f'MCS {mcs}', color=colors[mcs], linewidth=2)
        ax.fill_between(quantile_levels, data['p10'], data['p90'], alpha=0.2, color=colors[mcs])

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Quantile Level α')
    ax.set_ylabel('Quantile Error (log domain)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved Figure 2 to {save_path}")
    plt.close()

    return results


def generate_figure3_ccdf_error(model, test_sequences, test_configs, device='cpu',
                                 save_path='fig3_ccdf_error.png', slice_label=None):
    """Generate Figure 3: Aggregated CCDF error curve (per MCS, pooled over SNR).

    PKD v1: Restricts CCDF thresholds to [5%, 95%] quantiles to reduce amplification
    of decoding-cliff artifacts near extreme regimes.

    Args:
        model: Trained PKD model
        test_sequences: Filtered test sequences (single config slice)
        test_configs: Filtered test configs (single config slice)
        device: Computation device
        save_path: Output file path
        slice_label: Configuration slice label for plot title
    """
    print("\n" + "="*60)
    print("Generating Figure 3: CCDF Error Curves (Gaussian Innovation)")
    if slice_label:
        print(f"Configuration slice: {slice_label}")
    print("="*60)

    # Group by MCS (pool over SNR)
    grouped_by_mcs = defaultdict(list)
    for i, config in enumerate(test_configs):
        mcs = config['MCS']
        grouped_by_mcs[mcs].append(i)

    # Create inference engine
    from per_lut import AWGNPERLookup
    from infer import PKDInference
    # Load LDPC PER LUT (embedded data)
    per_lut = AWGNPERLookup.load_ldpc_lut()
    inference = PKDInference(model, per_lut, ar_order=model.ar_order, device=device)

    results = {}

    print("\nProcessing sequences by MCS...")
    for mcs in sorted(grouped_by_mcs.keys()):
        print(f"\nMCS {mcs}: {len(grouped_by_mcs[mcs])} sequences")
        seq_indices = grouped_by_mcs[mcs]

        # Collect all teacher values to determine threshold grid
        all_teacher_values = []
        for seq_idx in seq_indices:
            X_teacher = test_sequences[seq_idx]  # Already in natural log scale
            all_teacher_values.extend(X_teacher)

        all_teacher_values = np.array(all_teacher_values)
        # PKD v1: Restrict to [5%, 95%] to avoid decoding-cliff artifact amplification
        tau_min = np.percentile(all_teacher_values, 5)
        tau_max = np.percentile(all_teacher_values, 95)
        thresholds = np.linspace(tau_min, tau_max, 100)

        ccdf_errors = []

        for seq_idx in tqdm(seq_indices, desc=f"MCS {mcs}"):
            teacher_seq = test_sequences[seq_idx]
            config = test_configs[seq_idx]
            X_teacher = teacher_seq  # Already in natural log scale

            # Generate student
            config_traj = [config] * len(teacher_seq)
            student_results = inference.run_sequence(config_traj)
            student_seq_db = np.array(student_results['gamma_eff'])  # Inference outputs dB scale

            # Convert from dB to natural log scale
            student_seq = student_seq_db * np.log(10) / 10

            if np.all(np.isfinite(student_seq)):
                X_student = student_seq  # Now in natural log scale

                # Compute CCDF for each threshold
                ccdf_error = []
                for tau in thresholds:
                    ccdf_teacher = np.mean(X_teacher > tau)
                    ccdf_student = np.mean(X_student > tau)
                    ccdf_error.append(np.abs(ccdf_student - ccdf_teacher))

                ccdf_errors.append(ccdf_error)

        ccdf_errors = np.array(ccdf_errors)

        # Aggregate: median and 10-90 percentiles
        ccdf_error_median = np.median(ccdf_errors, axis=0)
        ccdf_error_10 = np.percentile(ccdf_errors, 10, axis=0)
        ccdf_error_90 = np.percentile(ccdf_errors, 90, axis=0)

        results[mcs] = {
            'thresholds': thresholds,
            'median': ccdf_error_median,
            'p10': ccdf_error_10,
            'p90': ccdf_error_90
        }

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for mcs in sorted(results.keys()):
        data = results[mcs]
        ax.plot(data['thresholds'], data['median'], '-', label=f'MCS {mcs}',
                color=colors[mcs], linewidth=2)
        ax.fill_between(data['thresholds'], data['p10'], data['p90'],
                        alpha=0.2, color=colors[mcs])

    ax.set_xlabel('Threshold τ (log-SINR)')
    ax.set_ylabel('CCDF Absolute Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved Figure 3 to {save_path}")
    plt.close()

    return results

