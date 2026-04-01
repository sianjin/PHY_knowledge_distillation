"""PKD Model Runtime Evaluation Script.

Measures inference runtime for generating sequences with the trained PKD model.

Usage:
    python -m pkd.example.evaluate_runtime \
        --channel-model 2 \
        --N-t 3 \
        --N-r 2 \
        --BW 20.0 \
        --N-ss 1 \
        --MCS 7 \
        [--snr 20.0] \
        [--num-sequences 50] \
        [--sequence-length 1000]

Example:
    # Runtime with random SNR
    python -m pkd.example.evaluate_runtime --channel-model 2 --N-t 3 --N-r 2 --BW 20.0 --N-ss 1 --MCS 7

    # Runtime with specific SNR
    python -m pkd.example.evaluate_runtime --channel-model 2 --N-t 3 --N-r 2 --BW 20.0 --N-ss 1 --MCS 7 --snr 20.0

    # Custom sequence count
    python -m pkd.example.evaluate_runtime --channel-model 2 --N-t 3 --N-r 2 --BW 20.0 --N-ss 1 --MCS 7 --num-sequences 100
"""

import argparse
import time
import torch
import numpy as np
import os
import sys

from pkd.model import PKDModel
from pkd.per_lut import AWGNPERLookup
from pkd.infer import PKDInference


# Channel model ID to name mapping (from documentation/YAML files)
CHANNEL_MODEL_NAMES = {
    1: 'A',
    2: 'B',
    3: 'C',
    4: 'D',
    5: 'E',
    6: 'F'
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate PKD model runtime performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument('--channel-model', type=int, required=True,
                        help='Channel model ID (matches data format, 2=Model B is common)')
    parser.add_argument('--N-t', type=int, required=True,
                        help='Number of transmit antennas')
    parser.add_argument('--N-r', type=int, required=True,
                        help='Number of receive antennas')
    parser.add_argument('--BW', type=float, required=True,
                        help='Bandwidth in MHz')
    parser.add_argument('--N-ss', type=int, required=True,
                        help='Number of spatial streams')
    parser.add_argument('--MCS', type=int, required=True,
                        help='Modulation & Coding Scheme (0-9)')

    # Optional arguments
    parser.add_argument('--snr', type=float, default=None,
                        help='Average SNR in dB (default: average over 10 SNRs from 10 to 55 dB)')
    parser.add_argument('--num-sequences', type=int, default=50,
                        help='Number of sequences to generate per SNR (default: 50)')
    parser.add_argument('--sequence-length', type=int, default=1000,
                        help='Length of each sequence (default: 1000)')
    parser.add_argument('--model-path', type=str, default='pkd/pkd_model.pt',
                        help='Path to trained model (default: pkd/pkd_model.pt)')

    args = parser.parse_args()

    # Validate arguments (allow wide range, let model embedding handle validation)
    if args.channel_model < 0 or args.channel_model > 10:
        parser.error('--channel-model must be a valid channel model ID')
    if args.MCS < 0 or args.MCS > 9:
        parser.error('--MCS must be between 0 and 9')
    if args.num_sequences < 1:
        parser.error('--num-sequences must be at least 1')
    if args.sequence_length < 1:
        parser.error('--sequence-length must be at least 1')

    return args


def load_model(model_path, device):
    """Load trained PKD model from checkpoint.

    Args:
        model_path: Path to model checkpoint file
        device: Torch device ('cuda' or 'cpu')

    Returns:
        Loaded PKDModel instance in eval mode
    """
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Extract model config with backward compatibility
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']

        # Add missing keys for backward compatibility
        if 'kappa_max' not in model_config:
            model_config['kappa_max'] = 0.95

        if 'innovation_type' not in model_config:
            model_config['innovation_type'] = 'gaussian'
            model_config['min_sigma'] = 0.1

        if 'num_R' not in model_config:
            model_config['num_R'] = 8
    else:
        # Default config for old checkpoints
        model_config = {
            'num_channel_models': 5,
            'num_mcs': 10,
            'num_nss': 4,
            'num_R': 8,
            'ar_order': 10,
            'hidden_dim': 128,
            'kappa_max': 0.95,
            'innovation_type': 'gaussian',
            'min_sigma': 0.1
        }

    # Instantiate model
    model = PKDModel(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def main():
    """Main runtime evaluation function."""
    args = parse_args()

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Handle SNR: use provided value or average over 10 SNR values
    if args.snr is None:
        # Use 10 SNR values evenly spaced from 10 to 55 dB for fair comparison
        snr_values = np.linspace(10, 55, 10)
        snr_source = 'average over 10 SNRs (10-55 dB)'
        use_multiple_snrs = True
    else:
        snr_values = [args.snr]
        snr_source = 'user-specified'
        use_multiple_snrs = False

    # Print configuration
    print('=' * 60)
    print('PKD Model Runtime Evaluation')
    print('=' * 60)
    print('\nConfiguration:')
    channel_name = CHANNEL_MODEL_NAMES.get(args.channel_model, 'Unknown')
    print(f'  Channel Model: {channel_name} (id={args.channel_model})')
    print(f'  Antennas: N_t={args.N_t}, N_r={args.N_r}')
    print(f'  Bandwidth: {args.BW} MHz')
    print(f'  MCS: {args.MCS}, N_ss: {args.N_ss}')
    if use_multiple_snrs:
        print(f'  SNR values: {snr_values} dB')
        print(f'  SNR mode: {snr_source}')
    else:
        print(f'  SNR: {snr_values[0]:.2f} dB ({snr_source})')
    print(f'\nRuntime Settings:')
    print(f'  Sequences per SNR: {args.num_sequences}')
    print(f'  Length per sequence: {args.sequence_length}')
    print(f'  Number of SNR values: {len(snr_values)}')
    print(f'  Total sequences: {args.num_sequences * len(snr_values)}')
    print(f'  Device: {device}')
    print(f'  Model path: {args.model_path}')

    # Load model
    print(f'\nLoading model from {args.model_path}...')
    model = load_model(args.model_path, device)
    print('Model loaded successfully')

    # Load PER LUT
    per_lut = AWGNPERLookup.load_ldpc_lut()

    # Create inference engine
    inference = PKDInference(
        model=model,
        per_lut=per_lut,
        ar_order=model.ar_order,
        device=device
    )

    # Run runtime evaluation across all SNR values
    print('\n' + '=' * 60)
    print('Running Runtime Evaluation...')
    print('=' * 60)

    all_runtimes = []
    total_start_time = time.time()

    for snr_idx, snr in enumerate(snr_values):
        print(f'\nSNR {snr_idx+1}/{len(snr_values)}: {snr:.2f} dB')

        # Build configuration dict for this SNR
        # Note: channel_model_id is 0-indexed (0-4) matching model embeddings
        # N_ss is 1-indexed (1 to num_nss), encoder converts internally
        config = {
            'channel_model_id': args.channel_model,  # Use as-is (0-indexed)
            'N_t': args.N_t,
            'N_r': args.N_r,
            'BW': float(args.BW),  # Ensure float type
            'SNR_bar': float(snr),  # Convert numpy float64 to Python float
            'MCS': args.MCS,
            'N_ss': args.N_ss,  # Keep 1-indexed (encoder expects 1-indexed)
            'R_t': 0,  # Full-band allocation (default)
            'packet_length': 1000  # Data packet length in bytes
        }

        # Build config trajectory (constant config repeated sequence_length times)
        config_trajectory = [config] * args.sequence_length

        # Measure runtime for this SNR
        snr_start_time = time.time()

        for i in range(args.num_sequences):
            # Generate sequence
            results = inference.run_sequence(config_trajectory)
            # Note: Each call to run_sequence() performs cold_start() internally

            # Print progress every 10 sequences
            if (i + 1) % 10 == 0:
                elapsed = time.time() - snr_start_time
                print(f'  Progress: {i+1}/{args.num_sequences} sequences '
                      f'({elapsed:.2f}s elapsed)')

        snr_elapsed_time = time.time() - snr_start_time
        all_runtimes.append(snr_elapsed_time)

        print(f'  Runtime for SNR={snr:.2f} dB: {snr_elapsed_time:.4f} seconds')

    total_elapsed_time = time.time() - total_start_time

    # Compute statistics
    if use_multiple_snrs:
        avg_runtime = np.mean(all_runtimes)
        std_runtime = np.std(all_runtimes)
        min_runtime = np.min(all_runtimes)
        max_runtime = np.max(all_runtimes)

        total_samples_per_snr = args.num_sequences * args.sequence_length
        avg_time_per_sequence = avg_runtime / args.num_sequences
        avg_time_per_sample = avg_runtime / total_samples_per_snr

        # Print results
        print('\n' + '=' * 60)
        print('Results (Averaged over SNRs):')
        print('=' * 60)
        print(f'  Average total runtime: {avg_runtime:.4f} seconds')
        print(f'  Std dev runtime: {std_runtime:.4f} seconds')
        print(f'  Min runtime: {min_runtime:.4f} seconds')
        print(f'  Max runtime: {max_runtime:.4f} seconds')
        print(f'  Total wall-clock time: {total_elapsed_time:.4f} seconds')
        print(f'\n  Average time per sequence: {avg_time_per_sequence:.4f} seconds')
        print(f'  Average time per sample: {avg_time_per_sample:.6f} seconds')
        print(f'  Average throughput: {total_samples_per_snr/avg_runtime:.2f} samples/second')
        print('=' * 60)
    else:
        # Single SNR case
        elapsed_time = all_runtimes[0]
        total_samples = args.num_sequences * args.sequence_length
        time_per_sequence = elapsed_time / args.num_sequences
        time_per_sample = elapsed_time / total_samples

        print('\n' + '=' * 60)
        print('Results:')
        print('=' * 60)
        print(f'  Total runtime: {elapsed_time:.4f} seconds')
        print(f'  Time per sequence: {time_per_sequence:.4f} seconds')
        print(f'  Time per sample: {time_per_sample:.6f} seconds')
        print(f'  Throughput: {total_samples/elapsed_time:.2f} samples/second')
        print('=' * 60)


if __name__ == '__main__':
    main()
