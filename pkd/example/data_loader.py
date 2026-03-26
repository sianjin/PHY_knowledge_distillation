"""Data loading utilities for PKD examples."""

import numpy as np
import h5py
import os
import glob


def load_real_data(data_dir='data', train_ratio=0.7, val_ratio=0.1, max_files=None, random_seed=42):
    """Load real PHY simulator data from .mat files.

    Args:
        data_dir: Directory containing .mat files
        train_ratio: Ratio of sequences to use for training (default: 0.7)
        val_ratio: Ratio of sequences to use for validation (default: 0.1)
        max_files: Maximum number of files to load (None = load all)
        random_seed: Random seed for reproducible shuffling (None = no shuffling)

    Returns:
        train_sequences: List of gamma_eff sequences for training (in natural log scale)
        train_configs: List of config dicts for training
        val_sequences: List of gamma_eff sequences for validation (in natural log scale)
        val_configs: List of config dicts for validation
        test_sequences: List of gamma_eff sequences for testing (in natural log scale)
        test_configs: List of config dicts for testing

    Note:
        The .mat files contain gamma_eff in dB scale (10*log10 of SINR).
        This function converts to natural log scale: ln(gamma_linear) = gamma_dB * ln(10) / 10
        The model training and inference code work with natural log scale values.

        Sequences are randomly shuffled before splitting to ensure train/val/test sets
        have representative samples from all configurations.

        Default split is 70% train, 10% val, 20% test.
    """
    # Find all .mat files
    mat_files = sorted(glob.glob(os.path.join(data_dir, '*.mat')))

    if max_files is not None:
        mat_files = mat_files[:max_files]

    print(f"Found {len(mat_files)} .mat files")

    all_sequences = []
    all_configs = []

    for mat_file in mat_files:
        print(f"Loading {os.path.basename(mat_file)}...")

        with h5py.File(mat_file, 'r') as f:
            # Load configuration (shape: (7, 100) where each column is config for one sequence)
            config_array = f['config'][:]  # Shape: (7, 100)

            # Load gamma_eff sequences (shape: (1000, 100) where each column is one sequence)
            gamma_eff = f['gamma_eff'][:]  # Shape: (1000, 100)

            # Number of sequences in this file
            num_sequences = gamma_eff.shape[1]

            # Process each sequence
            for i in range(num_sequences):
                # Extract config for this sequence (column i)
                config_col = config_array[:, i]

                channel_model_id = int(config_col[0])
                N_t = int(config_col[1])
                N_r = int(config_col[2])
                BW = float(config_col[3])
                SNR_bar = float(config_col[4])
                MCS = int(config_col[5])  # Already 0-indexed in data
                N_ss = int(config_col[6])

                # Create config dict
                config_dict = {
                    'channel_model_id': channel_model_id,
                    'N_t': N_t,
                    'N_r': N_r,
                    'BW': BW,
                    'SNR_bar': SNR_bar,
                    'MCS': MCS,
                    'N_ss': N_ss,
                    'R_t': 0,  # Resource allocation: 0 = full-band (all data uses this)
                    'packet_length': 1000  # Data packet length in bytes
                }

                # Extract sequence for this index (column i)
                # Data is in dB scale from .mat files, convert to natural log scale
                gamma_dB = gamma_eff[:, i]  # Shape: (1000,), in dB scale
                # Convert: ln(gamma_linear) = gamma_dB * ln(10) / 10
                sequence = gamma_dB * np.log(10) / 10  # Convert to natural log scale

                all_sequences.append(sequence)
                all_configs.append(config_dict)

    print(f"\nTotal loaded: {len(all_sequences)} sequences")

    # Randomly shuffle all sequences before splitting
    if random_seed is not None:
        print(f"Shuffling data with random seed {random_seed}...")
        np.random.seed(random_seed)
        indices = np.random.permutation(len(all_sequences))
        all_sequences = [all_sequences[i] for i in indices]
        all_configs = [all_configs[i] for i in indices]

    # Split into train/val/test
    num_train = int(len(all_sequences) * train_ratio)
    num_val = int(len(all_sequences) * val_ratio)

    train_sequences = all_sequences[:num_train]
    train_configs = all_configs[:num_train]
    val_sequences = all_sequences[num_train:num_train + num_val]
    val_configs = all_configs[num_train:num_train + num_val]
    test_sequences = all_sequences[num_train + num_val:]
    test_configs = all_configs[num_train + num_val:]

    print(f"Split: {len(train_sequences)} training, {len(val_sequences)} validation, {len(test_sequences)} test sequences")

    return train_sequences, train_configs, val_sequences, val_configs, test_sequences, test_configs
