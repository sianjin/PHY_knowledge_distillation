"""PKD Example Script - Entry Point

This is the main entry point that delegates to the refactored example/ modules.

Usage:
    python example.py train [N] [--exclusion-config PATH]    # Train with real data
    python example.py test [N] [--slice key:value ...]       # Evaluate on test set
    python example.py eval [idx] [N]                         # Qualitative evaluation

Examples:
    # Training
    python example.py train                                  # Train with all data
    python example.py train --exclusion-config pkd/example/training_exclusions.yaml
    python example.py train 10 --exclusion-config my_exclusions.yaml

    # Testing
    python example.py test                                   # Auto-select most common slice
    python example.py test --slice N_t:4 N_r:2               # Specify antenna config
    python example.py test --slice channel_model_id:2 N_t:4 N_r:2 BW:20.0 N_ss:2 MCS:7
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from example.main import (
    example_training,
    example_test_evaluation,
    example_evaluation
)


def parse_slice_spec(args):
    """Parse slice specification from command-line arguments.

    Args:
        args: List of command-line arguments

    Returns:
        tuple: (slice_spec dict or None, remaining args)

    Examples:
        ['--slice', 'N_t:4', 'N_r:2'] -> ({'N_t': 4, 'N_r': 2}, [])
        ['10', '--slice', 'N_t:4'] -> ({'N_t': 4}, ['10'])
        ['--slice', 'N_t:4', 'MCS:7'] -> ({'N_t': 4, 'MCS': 7}, [])
    """
    if '--slice' not in args:
        return None, args

    slice_idx = args.index('--slice')
    before_slice = args[:slice_idx]
    after_slice = args[slice_idx + 1:]

    slice_spec = {}
    remaining = []

    for arg in after_slice:
        if ':' in arg:
            key, value = arg.split(':', 1)
            # Try to convert to appropriate type
            if key in ['channel_model_id', 'N_t', 'N_r', 'N_ss', 'packet_length', 'MCS']:
                slice_spec[key] = int(value)
            elif key in ['BW', 'SNR_bar']:
                slice_spec[key] = float(value)
            else:
                # Unknown key, treat as string
                slice_spec[key] = value
        else:
            remaining.append(arg)

    return slice_spec if slice_spec else None, before_slice + remaining


def parse_exclusion_config(args):
    """Parse --exclusion-config argument from command-line arguments.

    Args:
        args: List of command-line arguments

    Returns:
        tuple: (exclusion_config_path or None, remaining args)

    Examples:
        ['--exclusion-config', 'config.yaml'] -> ('config.yaml', [])
        ['10', '--exclusion-config', 'config.yaml'] -> ('config.yaml', ['10'])
    """
    if '--exclusion-config' not in args:
        return None, args

    config_idx = args.index('--exclusion-config')

    if config_idx + 1 >= len(args):
        raise ValueError("--exclusion-config requires a path argument")

    config_path = args[config_idx + 1]
    remaining = args[:config_idx] + args[config_idx + 2:]

    return config_path, remaining


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1]
    remaining_args = sys.argv[2:]

    # Resolve data directory path relative to project root
    # Script is at pkd/example.py, data is at ../data/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_data_dir = os.path.join(project_root, 'data')

    if mode == 'train':
        # Train with real data
        exclusion_config, remaining_args = parse_exclusion_config(remaining_args)
        max_files = int(remaining_args[0]) if len(remaining_args) > 0 else None
        data_dir = remaining_args[1] if len(remaining_args) > 1 else default_data_dir
        example_training(data_dir=data_dir, max_files=max_files, exclusion_config=exclusion_config)

    elif mode == 'test':
        # Comprehensive test set evaluation with optional slice specification
        slice_spec, remaining = parse_slice_spec(remaining_args)
        max_files = int(remaining[0]) if len(remaining) > 0 else None
        data_dir = remaining[1] if len(remaining) > 1 else default_data_dir
        example_test_evaluation(data_dir=data_dir, max_files=max_files, slice_spec=slice_spec)

    elif mode == 'eval':
        # Qualitative evaluation with real data
        test_idx = int(remaining_args[0]) if len(remaining_args) > 0 else 0
        max_files = int(remaining_args[1]) if len(remaining_args) > 1 else None
        data_dir = remaining_args[2] if len(remaining_args) > 2 else default_data_dir
        example_evaluation(data_dir=data_dir, test_idx=test_idx, max_files=max_files)

    else:
        print(f"Unknown mode: {mode}")
        print(__doc__)
        sys.exit(1)
