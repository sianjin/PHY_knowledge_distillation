"""Main entry points for PKD examples."""

import torch
import numpy as np
import os

from model.pkd_model import PKDModel
from per_lut import AWGNPERLookup
from infer import PKDInference
from train import train_pkd

from .data_loader import load_real_data
from .evaluation import (
    evaluate_marginal_distribution,
    evaluate_temporal_dependence,
    evaluate_innovation_structure,
    evaluate_teacher_baseline_ar
)
from .plotting import (
    evaluate_test_set,
    generate_figure1_per_mcs_metrics,
    generate_figure2_quantile_error,
    generate_figure3_ccdf_error
)

# ============================================================================
# CONFIGURATION: Innovation Type
# ============================================================================
# To switch between innovation types, simply change INNOVATION_TYPE below.
# All training and evaluation functions will automatically use the specified type.
#
# Available options:
#   - 'gaussian': Simple Gaussian innovation (fastest, fewer parameters)
#   - 'sgn': Skew Generalized Normal (more flexible, models heavy tails)
#   - 'flow': Normalizing flow (most flexible, slow)
# ============================================================================

INNOVATION_TYPE = 'gaussian'  # <-- CHANGE THIS TO SWITCH INNOVATION TYPE

# Innovation-specific parameters (automatically selected based on INNOVATION_TYPE)
INNOVATION_PARAMS = {
    'gaussian': {
        'min_sigma': 0.1
    },
    'sgn': {
        'min_sigma': 0.1,
        'min_beta': 0.5,
        'max_beta': 4.0
    },
    'flow': {
        'min_sigma': 0.1,
        'num_flow_layers': 4
    }
}
# ============================================================================

def example_training(data_dir='data', max_files=None, exclusion_config=None):
    """Example training workflow with real PHY simulator data.

    Args:
        data_dir: Directory containing .mat files
        max_files: Maximum number of .mat files to load (None = load all)
        exclusion_config: Path to exclusion config file (None = no filtering)
    """

    # Create model using configured innovation type
    model_params = {
        'num_channel_models': 5,
        'num_mcs': 10,
        'num_nss': 4,
        'ar_order': 10,
        'hidden_dim': 128,
        'kappa_max': 0.95,  # Keep PACF away from ±1 for stability
        'innovation_type': INNOVATION_TYPE,
    }
    # Add innovation-specific parameters
    model_params.update(INNOVATION_PARAMS[INNOVATION_TYPE])

    model = PKDModel(**model_params)

    # Load real PHY simulator data
    print("Loading real data from", data_dir)
    train_sequences, train_configs, val_sequences, val_configs, test_sequences, test_configs = load_real_data(
        data_dir=data_dir,
        train_ratio=0.7,
        val_ratio=0.1,
        max_files=max_files
    )
    print(f"Loaded {len(train_sequences)} training, {len(val_sequences)} validation, {len(test_sequences)} test sequences")

    # Apply exclusion filter if config provided
    if exclusion_config is not None:
        print(f"\nApplying exclusion filter from {exclusion_config}")
        from .exclusion_filter import filter_dataset_with_exclusions

        (train_sequences, train_configs,
         val_sequences, val_configs,
         test_sequences, test_configs,
         filter_stats) = filter_dataset_with_exclusions(
            train_sequences, train_configs,
            val_sequences, val_configs,
            test_sequences, test_configs,
            exclusion_config_path=exclusion_config
        )

    print(f"\nFinal counts: {len(train_sequences)} training, {len(val_sequences)} validation, {len(test_sequences)} test sequences")

    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_config = {
        'num_channel_models': 5,
        'num_mcs': 10,  # MCS 0-9
        'num_nss': 4,
        'ar_order': 10,
        'hidden_dim': 128,
        'kappa_max': 0.95,
        'innovation_type': INNOVATION_TYPE,
    }
    # Add innovation-specific parameters
    model_config.update(INNOVATION_PARAMS[INNOVATION_TYPE])

    trained_model = train_pkd(
        model,
        train_sequences,
        train_configs,
        val_sequences,
        val_configs,
        num_epochs=10,
        batch_size=1024,
        lr=1e-3,
        device=device,
        early_stopping_patience=3,
        model_config=model_config
    )

    print(f"Training complete. Best model saved to pkd_model.pt")

    return trained_model


# ============ Evaluation Functions ============

def example_test_evaluation(data_dir='data', max_files=None, slice_spec=None):
    """Evaluate trained model on the held-out test set.

    This performs rigorous quantitative evaluation on all test sequences
    from the 20% held-out test split and generates three comprehensive figures.

    PKD v1: Filters to a single configuration slice (fixed Ntx/Nrx/Nss/BW/channel)
    to ensure figures show clean regime behavior without mixing heterogeneous configs.

    Args:
        data_dir: Directory containing .mat files
        max_files: Maximum number of files to load (should match training)
        slice_spec: Optional dict specifying fixed config values (e.g., {'N_t': 4, 'N_r': 2}).
                    If None, automatically selects the most common slice.
    """
    print("=" * 60)
    print("PKD Model - Test Set Evaluation (PKD v1: Gaussian Innovation)")
    print("=" * 60)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load('pkd_model.pt', map_location=device)

    print(f"\nLoaded checkpoint from pkd_model.pt")
    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch']+1} epochs")
    if 'val_loss' in checkpoint:
        print(f"  Best validation loss: {checkpoint['val_loss']:.4f}")

    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        if 'kappa_max' not in model_config:
            model_config['kappa_max'] = 0.95
        if 'innovation_type' not in model_config:
            model_config['innovation_type'] = INNOVATION_TYPE
            model_config.update(INNOVATION_PARAMS[INNOVATION_TYPE])
    else:
        model_config = {
            'num_channel_models': 5,
            'num_mcs': 10,
            'num_nss': 4,
            'ar_order': 10,
            'hidden_dim': 128,
            'kappa_max': 0.95,
            'innovation_type': INNOVATION_TYPE,
        }
        model_config.update(INNOVATION_PARAMS[INNOVATION_TYPE])

    model = PKDModel(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load test data (using same split as training)
    print(f"\nLoading test data from {data_dir}")
    _, _, _, _, test_sequences, test_configs = load_real_data(
        data_dir=data_dir,
        train_ratio=0.7,
        val_ratio=0.1,
        max_files=max_files,
        random_seed=42  # MUST use same seed as training!
    )

    print(f"Loaded {len(test_sequences)} test sequences")

    # ===== Configuration Slice Filtering =====
    # Define fixed keys (exclude MCS and SNR_bar which are the varying dimensions)
    from .utils import infer_most_common_slice, filter_by_slice, format_slice_label
    from collections import defaultdict

    fixed_keys = ['channel_model_id', 'N_t', 'N_r', 'BW', 'N_ss', 'packet_length']

    # Determine slice specification
    if slice_spec is None:
        print("\n" + "="*60)
        print("Auto-selecting configuration slice (most common in test set)")
        print("="*60)
        slice_spec = infer_most_common_slice(test_configs, fixed_keys)
    else:
        print("\n" + "="*60)
        print("Using user-specified configuration slice:")
        print("="*60)
        for k, v in slice_spec.items():
            print(f"  {k}: {v}")

    # Filter test set to this slice
    print(f"\nFiltering test set to slice...")
    test_sequences_orig = test_sequences
    test_configs_orig = test_configs
    test_sequences, test_configs, kept_indices = filter_by_slice(
        test_sequences, test_configs, slice_spec
    )

    print(f"Kept {len(test_sequences)}/{len(test_sequences_orig)} sequences " +
          f"({100*len(test_sequences)/len(test_sequences_orig):.1f}%)")

    # Verify sufficient coverage
    print("\nVerifying (MCS, SNR) coverage in filtered slice:")
    mcs_snr_counts = defaultdict(int)
    for cfg in test_configs:
        mcs = cfg['MCS']
        snr = int(cfg['SNR_bar'])
        mcs_snr_counts[(mcs, snr)] += 1

    print(f"Total (MCS, SNR) combinations: {len(mcs_snr_counts)}")
    min_count = min(mcs_snr_counts.values()) if mcs_snr_counts else 0
    max_count = max(mcs_snr_counts.values()) if mcs_snr_counts else 0
    print(f"Sequences per (MCS, SNR): min={min_count}, max={max_count}")

    if min_count < 10:
        print(f"WARNING: Some (MCS, SNR) bins have <10 sequences. Consider using more data.")

    # Format slice label for plots
    slice_label = format_slice_label(slice_spec)
    print(f"\nSlice label for plots: {slice_label}")

    # Basic test set evaluation
    test_metrics = evaluate_test_set(model, test_sequences, test_configs, device)

    # Generate three comprehensive figures (with slice label)
    print("\n" + "="*60)
    print("Generating Comprehensive Test Set Figures")
    print("="*60)

    fig1_results = generate_figure1_per_mcs_metrics(
        model, test_sequences, test_configs, device,
        save_path='figures/test_metrics', slice_label=slice_label
    )
    fig2_results = generate_figure2_quantile_error(
        model, test_sequences, test_configs, device,
        save_path='figures/test_quantile_error.png', slice_label=slice_label
    )
    fig3_results = generate_figure3_ccdf_error(
        model, test_sequences, test_configs, device,
        save_path='figures/test_ccdf_error.png', slice_label=slice_label
    )

    print("\n" + "="*60)
    print("Test Set Evaluation Complete!")
    print("="*60)
    print("Generated files in figures/:")
    print("  Test metrics (6 files):")
    print("    - figures/test_metrics_pit_pass_rate.png")
    print("    - figures/test_metrics_lb_pass_rate_zt.png")
    print("    - figures/test_metrics_lb_pass_rate_zt2.png")
    print("    - figures/test_metrics_acf_rmse.png")
    print("    - figures/test_metrics_psd_rmse.png")
    print("    - figures/test_metrics_ks_stat.png")
    print("  Quantile error:")
    print("    - figures/test_quantile_error.png")
    print("  CCDF error:")
    print("    - figures/test_ccdf_error.png")
    print(f"\nConfiguration slice: {slice_label}")
    print(f"Evaluated on {len(test_sequences)} test sequences")

    return {
        'test_metrics': test_metrics,
        'fig1_results': fig1_results,
        'fig2_results': fig2_results,
        'fig3_results': fig3_results,
        'slice_spec': slice_spec,
        'slice_label': slice_label
    }



def example_evaluation(data_dir='data', test_idx=None, max_files=None, slice_spec=None, slice_idx=None):
    """Run comprehensive evaluation of student model fidelity on a single test sequence.

    This is a qualitative analysis tool for detailed inspection of individual sequences
    from the held-out test set. For quantitative evaluation on the entire test set,
    use example_test_evaluation().

    Args:
        data_dir: Directory containing .mat files
        test_idx: Index within the test set (0 to num_test_sequences-1) - for backward compatibility
        max_files: Maximum number of files to load (should match training)
        slice_spec: Dict of config filters (e.g., {'N_t': 4, 'MCS': 7}) - new slice-based selection
        slice_idx: Index into filtered sequences (None = random selection with seed=42)

    Note:
        Two selection modes:
        1. Direct indexing (backward compatible): Provide test_idx
        2. Slice-based (new): Provide slice_spec, optionally with slice_idx

        The test set is the held-out 20% of data that the model never saw during training.
    """
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
            print(f"  Note: innovation_type not in checkpoint, using default '{INNOVATION_TYPE}'")
            model_config['innovation_type'] = INNOVATION_TYPE
            model_config.update(INNOVATION_PARAMS[INNOVATION_TYPE])
    else:
        print("  Warning: model_config not found in checkpoint, using defaults")
        model_config = {
            'num_channel_models': 5,
            'num_mcs': 10,  # MCS 0-9
            'num_nss': 4,
            'ar_order': 10,
            'hidden_dim': 128,
            'kappa_max': 0.95,
            'innovation_type': INNOVATION_TYPE,
        }
        model_config.update(INNOVATION_PARAMS[INNOVATION_TYPE])

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
            'N_ss': torch.tensor([2], device=device),
            'packet_length': torch.tensor([1000], device=device)  # Data packet length in bytes
        }
        h = model.encode_config(sample_config)
        params = model.generate_params(h)
        print(f"  Sample mean (m): {params['m'].item():.4f}")
        print(f"  Sample AR coeffs (phi): {params['phi'][0, :5].cpu().numpy()}")
        print(f"  Sample AR offset (c): {params['c'].item():.4f}")

    # Create inference engine
    # Load LDPC PER LUT (embedded data)
    per_lut = AWGNPERLookup.load_ldpc_lut()
    inference = PKDInference(model, per_lut, ar_order=model_config['ar_order'], device=device)

    # Load test set from the held-out data (20% split)
    print(f"\nLoading test data from {data_dir}")
    _, _, _, _, test_sequences, test_configs = load_real_data(
        data_dir=data_dir,
        train_ratio=0.7,
        val_ratio=0.1,
        max_files=max_files,
        random_seed=42  # MUST use same seed as training!
    )

    print(f"Loaded {len(test_sequences)} test sequences")

    # Sequence selection: three modes (direct index, slice+idx, slice+random)
    if test_idx is not None:
        # Backward compatibility mode: direct indexing
        if test_idx >= len(test_sequences):
            raise ValueError(f"test_idx={test_idx} out of range, only {len(test_sequences)} test sequences available")

        final_idx = test_idx
        teacher_seq = test_sequences[test_idx]
        config_dict = test_configs[test_idx]

        print(f"\nUsing test sequence {test_idx} (out of {len(test_sequences)} test sequences)")

    elif slice_spec is not None:
        # New slice-based mode
        from .utils import filter_by_slice, format_slice_label

        print("\n" + "="*60)
        print("Filtering test set by configuration slice:")
        print("="*60)
        for k, v in slice_spec.items():
            print(f"  {k}: {v}")

        # Filter test set
        filtered_sequences, filtered_configs, kept_indices = filter_by_slice(
            test_sequences, test_configs, slice_spec
        )

        print(f"\nMatching sequences: {len(filtered_sequences)}/{len(test_sequences)} " +
              f"({100*len(filtered_sequences)/len(test_sequences):.1f}%)")

        # Handle empty filter result
        if len(filtered_sequences) == 0:
            print("\nERROR: No sequences match the specified slice.")
            print("\nSlice specification:")
            for k, v in slice_spec.items():
                print(f"  {k}: {v}")
            print("\nSuggestions:")
            print("  1. Check that config values are valid (e.g., N_t, N_r, MCS ranges)")
            print("  2. Try a less restrictive slice (remove some parameters)")
            print("  3. Use 'python example.py test --slice ...' to see available configs")
            raise ValueError("No sequences match the specified slice")

        # Select from filtered sequences
        if slice_idx is not None:
            # Deterministic selection by index
            if slice_idx >= len(filtered_sequences):
                raise ValueError(
                    f"--idx {slice_idx} out of range for filtered sequences. "
                    f"Only {len(filtered_sequences)} sequences match the slice."
                )
            selection_idx = slice_idx
            print(f"\nUsing --idx {slice_idx} (deterministic selection)")
        else:
            # Random selection with fixed seed for reproducibility
            np.random.seed(42)
            selection_idx = np.random.randint(0, len(filtered_sequences))
            print(f"\nRandom selection (seed=42): index {selection_idx}/{len(filtered_sequences)-1}")

        # Get the selected sequence
        teacher_seq = filtered_sequences[selection_idx]
        config_dict = filtered_configs[selection_idx]
        final_idx = kept_indices[selection_idx]  # Original index in full test set

        # Display selection details
        slice_label = format_slice_label(slice_spec)
        print(f"\nSelected sequence details:")
        print(f"  Slice: {slice_label}")
        print(f"  Index within filtered set: {selection_idx}/{len(filtered_sequences)-1}")
        print(f"  Original test set index: {final_idx}/{len(test_sequences)-1}")

    else:
        # Neither test_idx nor slice_spec provided - use default (first sequence)
        final_idx = 0
        teacher_seq = test_sequences[0]
        config_dict = test_configs[0]
        print(f"\nUsing default test sequence 0 (out of {len(test_sequences)} test sequences)")

    print(f"Configuration: channel_model_id={config_dict['channel_model_id']}, "
          f"N_t={config_dict['N_t']}, N_r={config_dict['N_r']}, BW={config_dict['BW']}, "
          f"SNR={config_dict['SNR_bar']}, MCS={config_dict['MCS']}, N_ss={config_dict['N_ss']}")
    print(f"Sequence length: {len(teacher_seq)}")

    # Convert config dict to tensors for student model
    config = {
        'channel_model_id': torch.tensor(config_dict['channel_model_id']),
        'N_t': torch.tensor(config_dict['N_t']),
        'N_r': torch.tensor(config_dict['N_r']),
        'BW': torch.tensor(config_dict['BW']),
        'SNR_bar': torch.tensor(config_dict['SNR_bar']),
        'MCS': torch.tensor(config_dict['MCS']),
        'N_ss': torch.tensor(config_dict['N_ss']),
        'packet_length': torch.tensor(config_dict.get('packet_length', 1000))  # Default to 1000 bytes
    }

    config_traj = [config] * len(teacher_seq)
    student_results = inference.run_sequence(config_traj)
    student_seq_db = student_results['gamma_eff']  # Inference outputs dB scale

    # Convert student sequence from dB to natural log scale for comparison
    # dB scale: gamma_dB = 10*log10(gamma_linear)
    # Natural log: ln(gamma_linear) = gamma_dB * ln(10) / 10
    student_seq = student_seq_db * np.log(10) / 10

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

    # Diagnostics: check if student sequence is degenerate
    student_unique = len(np.unique(np.round(student_seq, 6)))
    print(f"  ✓ All values finite")
    print(f"  Unique values: {student_unique} / {len(student_seq)}")
    if student_unique < len(student_seq) * 0.9:
        print(f"  WARNING: Low diversity! Expected ~{len(student_seq)}, got {student_unique}")
    print(f"  Min: {np.min(student_seq):.4f}, Max: {np.max(student_seq):.4f}")
    print(f"  Mean: {np.mean(student_seq):.4f}, Std: {np.std(student_seq):.4f}")

    # Both sequences now in natural log scale for fair comparison
    teacher_log = teacher_seq  # Already in natural log scale (from data_loader)
    student_log = student_seq  # Converted from dB to natural log scale

    print("\n--- 1. Marginal Distribution Fidelity ---")
    marginal_metrics = evaluate_marginal_distribution(teacher_log, student_log, save_prefix='figures/eval_marginal')

    print("\n--- 2. Temporal Dependence ---")
    temporal_metrics = evaluate_temporal_dependence(teacher_log, student_log, save_prefix='figures/eval_temporal')

    print("\n--- 3. Innovation Structure ---")
    innovation_metrics = evaluate_innovation_structure(model, inference, teacher_seq, config, device, save_prefix='figures/eval_innovations')

    print("\n--- 4. Teacher Baseline (Classical AR) ---")
    print(f"Running classical AR({model.ar_order}) baseline on teacher data for comparison...")
    baseline_metrics = evaluate_teacher_baseline_ar(teacher_seq, ar_order=model.ar_order, save_prefix='figures/eval_teacher_baseline')

    print("\n" + "=" * 50)
    print("Evaluation complete!")
    print("=" * 50)
    print("Generated files in figures/ (11 total):")
    print("  Marginal distribution (3 files):")
    print("    - figures/eval_marginal_ccdf.png")
    print("    - figures/eval_marginal_qq.png")
    print("    - figures/eval_marginal_quantile_error.png")
    print("  Temporal dependence (2 files):")
    print("    - figures/eval_temporal_acf.png")
    print("    - figures/eval_temporal_psd.png")
    print("  Innovation structure (2 files):")
    print("    - figures/eval_innovations_pit_hist.png")
    print("    - figures/eval_innovations_pit_acf.png")
    print("  Teacher baseline (4 files):")
    print("    - figures/eval_teacher_baseline_residuals.png")
    print("    - figures/eval_teacher_baseline_acf_zt.png")
    print("    - figures/eval_teacher_baseline_acf_zt2.png")
    print("    - figures/eval_teacher_baseline_pit_hist.png")
    print("=" * 50)

    return {
        'marginal': marginal_metrics,
        'temporal': temporal_metrics,
        'innovation': innovation_metrics,
        'baseline': baseline_metrics
    }


if __name__ == '__main__':
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else 'eval'

    # Determine data directory path relative to this script
    # Script is in pkd/example/main.py, data is in ../../data/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkd_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(pkd_dir)
    data_dir = os.path.join(project_root, 'data')

    if mode == 'train':
        # Run training with real data
        print("=" * 50)
        print("PKD Example - Training with Real Data")
        print("=" * 50)
        max_files = int(sys.argv[2]) if len(sys.argv) > 2 else None
        trained_model = example_training(data_dir=data_dir, max_files=max_files)

    elif mode == 'test':
        # Run test set evaluation
        print("=" * 50)
        print("PKD Example - Test Set Evaluation")
        print("=" * 50)
        max_files = int(sys.argv[2]) if len(sys.argv) > 2 else None
        test_metrics = example_test_evaluation(data_dir=data_dir, max_files=max_files)

    elif mode == 'eval':
        # Run evaluation with real data (single test sequence)
        test_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        max_files = int(sys.argv[3]) if len(sys.argv) > 3 else None
        example_evaluation(data_dir=data_dir, test_idx=test_idx, max_files=max_files)

    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: train, test, eval")
        sys.exit(1)