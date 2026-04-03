"""PKD Example Script - Entry Point

Usage:
    python -m pkd.example train [N] [--exclusion-config PATH] [--exclusion-mcs PCT] [--exclusion-seed SEED]
    python -m pkd.example test [N] [--slice key:value ...]
    python -m pkd.example eval [idx] [N]
    python -m pkd.example eval --slice key:value ... [--idx N]

Examples:
    # Training
    python -m pkd.example train
    python -m pkd.example train --exclusion-mcs 30%
    python -m pkd.example train --exclusion-config pkd/example/training_exclusions.yaml
    python -m pkd.example train --exclusion-mcs 25% --exclusion-seed 12345

    # Testing
    python -m pkd.example test
    python -m pkd.example test --slice N_t:4 N_r:2 MCS:7
    python -m pkd.example test --slice channel_model_id:2 N_t:3 N_r:2 BW:20.0 N_ss:2 MCS:7

    # Evaluation
    python -m pkd.example eval --slice N_t:3 N_r:2 MCS:7
    python -m pkd.example eval --slice channel_model_id:2 N_t:3 N_r:2 BW:40 N_ss:2 MCS:7
    python -m pkd.example eval --slice N_t:3 N_r:2 MCS:7 --idx 5
    python -m pkd.example eval 50
"""

import sys
import os

from .main import (
    example_training,
    example_test_evaluation,
    example_evaluation
)


def parse_slice_spec(args):
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
            if key in ['channel_model_id', 'N_t', 'N_r', 'N_ss', 'packet_length', 'MCS']:
                slice_spec[key] = int(value)
            elif key in ['BW', 'SNR_bar']:
                slice_spec[key] = float(value)
            else:
                slice_spec[key] = value
        else:
            remaining.append(arg)

    return slice_spec if slice_spec else None, before_slice + remaining


def parse_exclusion_config(args):
    if '--exclusion-config' not in args:
        return None, args

    config_idx = args.index('--exclusion-config')
    if config_idx + 1 >= len(args):
        raise ValueError("--exclusion-config requires a path argument")

    config_path = args[config_idx + 1]
    remaining = args[:config_idx] + args[config_idx + 2:]
    return config_path, remaining


def parse_exclusion_mcs(args):
    if '--exclusion-mcs' not in args:
        return None, args

    mcs_idx = args.index('--exclusion-mcs')
    if mcs_idx + 1 >= len(args):
        raise ValueError("--exclusion-mcs requires a percentage argument")

    percentage_str = args[mcs_idx + 1]
    remaining = args[:mcs_idx] + args[mcs_idx + 2:]

    if percentage_str.endswith('%'):
        percentage = float(percentage_str[:-1])
    else:
        percentage = float(percentage_str)
        if percentage < 1.0:
            percentage = percentage * 100

    if not (0 < percentage < 100):
        raise ValueError(f"Percentage must be between 0 and 100, got {percentage}%.")

    return percentage, remaining


def parse_exclusion_seed(args):
    if '--exclusion-seed' not in args:
        return 42, args

    seed_idx = args.index('--exclusion-seed')
    if seed_idx + 1 >= len(args):
        raise ValueError("--exclusion-seed requires an integer argument")

    seed = int(args[seed_idx + 1])
    remaining = args[:seed_idx] + args[seed_idx + 2:]

    if seed < 0:
        raise ValueError(f"Random seed must be non-negative, got {seed}")

    return seed, remaining


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1]
    remaining_args = sys.argv[2:]

    # __file__ is pkd/example/__main__.py; data is at <project_root>/data/
    script_dir = os.path.dirname(os.path.abspath(__file__))   # .../pkd/example/
    pkd_dir = os.path.dirname(script_dir)                      # .../pkd/
    project_root = os.path.dirname(pkd_dir)                    # .../PHY_knowledge_distillation/
    default_data_dir = os.path.join(project_root, 'data')

    if mode == 'train':
        exclusion_config, remaining_args = parse_exclusion_config(remaining_args)
        exclusion_mcs_pct, remaining_args = parse_exclusion_mcs(remaining_args)
        exclusion_seed, remaining_args = parse_exclusion_seed(remaining_args)
        max_files = int(remaining_args[0]) if len(remaining_args) > 0 else None
        data_dir = remaining_args[1] if len(remaining_args) > 1 else default_data_dir
        example_training(
            data_dir=data_dir,
            max_files=max_files,
            exclusion_config=exclusion_config,
            exclusion_mcs_percentage=exclusion_mcs_pct,
            exclusion_seed=exclusion_seed
        )

    elif mode == 'test':
        slice_spec, remaining = parse_slice_spec(remaining_args)
        max_files = int(remaining[0]) if len(remaining) > 0 else None
        data_dir = remaining[1] if len(remaining) > 1 else default_data_dir
        example_test_evaluation(data_dir=data_dir, max_files=max_files, slice_spec=slice_spec)

    elif mode == 'eval':
        slice_spec, remaining = parse_slice_spec(remaining_args)

        idx = None
        if '--idx' in remaining:
            idx_pos = remaining.index('--idx')
            if idx_pos + 1 >= len(remaining):
                raise ValueError("--idx requires an integer argument")
            idx = int(remaining[idx_pos + 1])
            remaining = remaining[:idx_pos] + remaining[idx_pos + 2:]

        if slice_spec is None and idx is None:
            test_idx = int(remaining[0]) if len(remaining) > 0 else 0
            max_files = int(remaining[1]) if len(remaining) > 1 else None
            data_dir = remaining[2] if len(remaining) > 2 else default_data_dir
            example_evaluation(data_dir=data_dir, test_idx=test_idx, max_files=max_files)
        else:
            max_files = int(remaining[0]) if len(remaining) > 0 else None
            data_dir = remaining[1] if len(remaining) > 1 else default_data_dir
            example_evaluation(data_dir=data_dir, max_files=max_files,
                               slice_spec=slice_spec, slice_idx=idx)

    else:
        print(f"Unknown mode: {mode}")
        print(__doc__)
        sys.exit(1)
