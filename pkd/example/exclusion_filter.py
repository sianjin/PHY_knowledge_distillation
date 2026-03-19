"""Exclusion filtering for training data generalization testing.

This module provides functionality to exclude specific (config_slice, MCS, SNR) combinations
from training/validation data to test model generalization on held-out configurations.
"""

import os
import warnings
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    warnings.warn("pyyaml not installed. Install with: pip install pyyaml")


def load_exclusion_config(config_path: str) -> dict:
    """Load and parse exclusion configuration file.

    Supports YAML format. Auto-detects format by file extension.

    Args:
        config_path: Path to configuration file (.yaml or .yml)

    Returns:
        Parsed configuration dictionary

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If file format not supported or invalid YAML
        ImportError: If pyyaml not installed
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Exclusion config file not found: {config_path}")

    # Check file extension
    ext = os.path.splitext(config_path)[1].lower()

    if ext in ['.yaml', '.yml']:
        if not YAML_AVAILABLE:
            raise ImportError(
                "pyyaml is required for YAML config files. "
                "Install with: pip install pyyaml"
            )

        with open(config_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML syntax in {config_path}: {e}")
    else:
        raise ValueError(
            f"Unsupported config file format: {ext}. "
            f"Supported formats: .yaml, .yml"
        )

    return config


def validate_exclusion_config(config: dict) -> None:
    """Validate exclusion configuration schema.

    Args:
        config: Parsed configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    # Check version
    if 'version' not in config:
        warnings.warn("No version specified in exclusion config, assuming 1.0")

    # Check required fields
    if 'exclusions' not in config:
        raise ValueError("Missing required field 'exclusions' in config")

    if not isinstance(config['exclusions'], list):
        raise ValueError("'exclusions' must be a list")

    # Validate settings if present
    if 'settings' in config:
        settings = config['settings']
        valid_settings = {'apply_to_train', 'apply_to_val', 'apply_to_test'}
        invalid_keys = set(settings.keys()) - valid_settings
        if invalid_keys:
            warnings.warn(f"Unknown settings keys (will be ignored): {invalid_keys}")

    # Validate each exclusion rule
    valid_config_keys = {'channel_model_id', 'N_t', 'N_r', 'BW', 'N_ss', 'packet_length'}

    for idx, rule in enumerate(config['exclusions']):
        # Check required fields
        if 'config' not in rule:
            raise ValueError(f"Rule {idx}: missing 'config' field")
        if 'mcs_list' not in rule:
            raise ValueError(f"Rule {idx}: missing 'mcs_list' field")

        # Validate MCS values
        mcs_list = rule['mcs_list']
        if not isinstance(mcs_list, list):
            raise ValueError(f"Rule {idx}: 'mcs_list' must be a list")

        for mcs in mcs_list:
            if not isinstance(mcs, int) or mcs < 0 or mcs > 9:
                raise ValueError(
                    f"Rule {idx}: Invalid MCS {mcs}. Must be integer in [0, 9]"
                )

        # Validate config keys
        config_spec = rule['config']
        if not isinstance(config_spec, dict):
            raise ValueError(f"Rule {idx}: 'config' must be a dictionary")

        invalid_keys = set(config_spec.keys()) - valid_config_keys
        if invalid_keys:
            raise ValueError(
                f"Rule {idx}: Invalid config keys {invalid_keys}. "
                f"Valid keys: {valid_config_keys}"
            )

        # Validate config values
        if 'channel_model_id' in config_spec:
            ch_id = config_spec['channel_model_id']
            if not isinstance(ch_id, int) or ch_id < 1 or ch_id > 6:
                raise ValueError(
                    f"Rule {idx}: channel_model_id must be integer in [1, 6], got {ch_id}"
                )

        if 'BW' in config_spec:
            bw = config_spec['BW']
            if bw not in [20.0, 40.0, 80.0, 160.0]:
                warnings.warn(
                    f"Rule {idx}: Unusual BW value {bw} MHz. "
                    f"Common values: 20, 40, 80, 160"
                )

        # Validate SNR filter if present
        if 'snr_filter' in rule:
            _validate_snr_filter(rule['snr_filter'], idx)

    # Check for duplicate rules
    _check_duplicate_rules(config['exclusions'])


def _validate_snr_filter(snr_filter: dict, rule_idx: int) -> None:
    """Validate SNR filter specification.

    Args:
        snr_filter: SNR filter dictionary
        rule_idx: Rule index for error messages

    Raises:
        ValueError: If SNR filter is invalid
    """
    if 'type' not in snr_filter:
        raise ValueError(f"Rule {rule_idx}: snr_filter missing 'type' field")

    filter_type = snr_filter['type']

    if filter_type == 'values':
        if 'values' not in snr_filter:
            raise ValueError(f"Rule {rule_idx}: snr_filter type 'values' requires 'values' field")
        if not isinstance(snr_filter['values'], list):
            raise ValueError(f"Rule {rule_idx}: snr_filter 'values' must be a list")
        if not snr_filter['values']:
            raise ValueError(f"Rule {rule_idx}: snr_filter 'values' cannot be empty")

    elif filter_type == 'range':
        if 'min' not in snr_filter or 'max' not in snr_filter:
            raise ValueError(
                f"Rule {rule_idx}: snr_filter type 'range' requires 'min' and 'max' fields"
            )
        if snr_filter['min'] > snr_filter['max']:
            raise ValueError(
                f"Rule {rule_idx}: snr_filter min ({snr_filter['min']}) > max ({snr_filter['max']})"
            )

    elif filter_type == 'threshold':
        if 'operator' not in snr_filter or 'value' not in snr_filter:
            raise ValueError(
                f"Rule {rule_idx}: snr_filter type 'threshold' requires 'operator' and 'value' fields"
            )
        valid_ops = {'gte', 'lte', 'gt', 'lt'}
        if snr_filter['operator'] not in valid_ops:
            raise ValueError(
                f"Rule {rule_idx}: Invalid operator '{snr_filter['operator']}'. "
                f"Valid: {valid_ops}"
            )

    else:
        raise ValueError(
            f"Rule {rule_idx}: Unknown snr_filter type '{filter_type}'. "
            f"Valid types: 'values', 'range', 'threshold'"
        )


def _check_duplicate_rules(exclusions: List[dict]) -> None:
    """Check for duplicate exclusion rules and warn.

    Args:
        exclusions: List of exclusion rules
    """
    signatures = set()

    for rule in exclusions:
        # Create signature from config + mcs_list
        config_sig = tuple(sorted(rule['config'].items()))
        mcs_sig = tuple(sorted(rule['mcs_list']))
        sig = (config_sig, mcs_sig)

        if sig in signatures:
            rule_name = rule.get('name', 'unnamed')
            warnings.warn(f"Duplicate exclusion rule detected: {rule_name}")

        signatures.add(sig)


def matches_snr_filter(snr_value: float, snr_filter: Optional[dict]) -> bool:
    """Check if SNR value matches SNR filter.

    Args:
        snr_value: SNR value to check
        snr_filter: SNR filter specification (None means match all SNR values)

    Returns:
        True if SNR matches filter, False otherwise
    """
    # No filter means match all SNR values
    if snr_filter is None:
        return True

    filter_type = snr_filter['type']

    if filter_type == 'values':
        # Check if SNR is in the list of values
        # Use approximate matching for floating point
        return any(abs(snr_value - v) < 0.5 for v in snr_filter['values'])

    elif filter_type == 'range':
        # Check if SNR is in [min, max]
        return snr_filter['min'] <= snr_value <= snr_filter['max']

    elif filter_type == 'threshold':
        # Check threshold comparison
        op = snr_filter['operator']
        threshold = snr_filter['value']

        if op == 'gte':
            return snr_value >= threshold
        elif op == 'lte':
            return snr_value <= threshold
        elif op == 'gt':
            return snr_value > threshold
        elif op == 'lt':
            return snr_value < threshold

    return False


def matches_exclusion_rule(config_dict: dict, rule: dict) -> bool:
    """Check if a sequence configuration matches an exclusion rule.

    Args:
        config_dict: Sequence configuration dictionary
        rule: Exclusion rule specification

    Returns:
        True if sequence should be excluded, False otherwise
    """
    # Check config slice match
    rule_config = rule['config']
    for key, value in rule_config.items():
        if config_dict.get(key) != value:
            return False

    # Check MCS match
    seq_mcs = config_dict.get('MCS')
    if seq_mcs not in rule['mcs_list']:
        return False

    # Check SNR filter if present
    snr_filter = rule.get('snr_filter', None)
    if snr_filter is not None:
        seq_snr = config_dict.get('SNR_bar')
        if not matches_snr_filter(seq_snr, snr_filter):
            return False

    return True


def apply_exclusion_filter(
    sequences: List,
    configs: List[dict],
    exclusion_config: dict,
    split_name: str = 'train'
) -> Tuple[List, List, dict]:
    """Filter sequences based on exclusion rules.

    Args:
        sequences: List of sequences
        configs: List of configuration dictionaries
        exclusion_config: Parsed exclusion configuration
        split_name: Name of split ('train', 'val', or 'test')

    Returns:
        Tuple of (filtered_sequences, filtered_configs, stats_dict)

        stats_dict contains:
            - total_before: Number of sequences before filtering
            - total_after: Number of sequences after filtering
            - excluded_count: Number of sequences excluded
            - exclusions_applied: List of rules that matched sequences
    """
    # Check if filtering should be applied to this split
    settings = exclusion_config.get('settings', {})
    apply_key = f'apply_to_{split_name}'
    should_apply = settings.get(apply_key, split_name != 'test')

    if not should_apply:
        # No filtering for this split
        return sequences, configs, {
            'total_before': len(sequences),
            'total_after': len(sequences),
            'excluded_count': 0,
            'exclusions_applied': []
        }

    # Apply exclusion rules
    exclusion_rules = exclusion_config.get('exclusions', [])

    kept_sequences = []
    kept_configs = []
    excluded_count = 0
    rule_match_counts = defaultdict(int)

    for seq, cfg in zip(sequences, configs):
        excluded = False

        for rule_idx, rule in enumerate(exclusion_rules):
            if matches_exclusion_rule(cfg, rule):
                excluded = True
                rule_match_counts[rule_idx] += 1
                break

        if not excluded:
            kept_sequences.append(seq)
            kept_configs.append(cfg)
        else:
            excluded_count += 1

    # Build statistics
    exclusions_applied = []
    for rule_idx, count in rule_match_counts.items():
        rule = exclusion_rules[rule_idx]
        exclusions_applied.append({
            'name': rule.get('name', f'Rule {rule_idx}'),
            'config': rule['config'],
            'mcs_list': rule['mcs_list'],
            'count': count
        })

    stats = {
        'total_before': len(sequences),
        'total_after': len(kept_sequences),
        'excluded_count': excluded_count,
        'exclusions_applied': exclusions_applied
    }

    return kept_sequences, kept_configs, stats


def print_exclusion_report(filter_stats: Dict[str, dict]) -> None:
    """Print comprehensive filtering report.

    Args:
        filter_stats: Dictionary mapping split names to stats dictionaries
    """
    print("\n" + "="*70)
    print("EXCLUSION FILTER REPORT")
    print("="*70)

    for split_name, stats in filter_stats.items():
        print(f"\n{split_name.upper()} SET:")
        print(f"  Total sequences before: {stats['total_before']}")
        print(f"  Total sequences after:  {stats['total_after']}")

        if stats['total_before'] > 0:
            pct = 100 * stats['excluded_count'] / stats['total_before']
        else:
            pct = 0.0

        print(f"  Excluded sequences:     {stats['excluded_count']} ({pct:.2f}%)")

        if stats['exclusions_applied']:
            print(f"\n  Exclusion rules applied ({len(stats['exclusions_applied'])}):")
            for rule_info in stats['exclusions_applied']:
                print(f"    [{rule_info['count']:4d} seqs] {rule_info['name']}")

                # Format config for display
                config_str = ', '.join(f"{k}:{v}" for k, v in rule_info['config'].items())
                print(f"              Config: {config_str}")
                print(f"              MCS:    {rule_info['mcs_list']}")

    print("="*70)


def filter_dataset_with_exclusions(
    train_sequences: List,
    train_configs: List[dict],
    val_sequences: List,
    val_configs: List[dict],
    test_sequences: List,
    test_configs: List[dict],
    exclusion_config_path: Optional[str] = None
) -> Tuple:
    """High-level wrapper to filter entire dataset with exclusions.

    Args:
        train_sequences: Training sequences
        train_configs: Training configurations
        val_sequences: Validation sequences
        val_configs: Validation configurations
        test_sequences: Test sequences
        test_configs: Test configurations
        exclusion_config_path: Path to exclusion config file (None = no filtering)

    Returns:
        Tuple of (filtered_train_seq, filtered_train_cfg,
                 filtered_val_seq, filtered_val_cfg,
                 filtered_test_seq, filtered_test_cfg,
                 filter_stats)

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config is invalid or all data filtered out
    """
    # If no config path, return original data
    if exclusion_config_path is None:
        warnings.warn("No exclusion config provided, using all data")
        return (train_sequences, train_configs,
                val_sequences, val_configs,
                test_sequences, test_configs,
                {})

    # Load and validate config
    try:
        exclusion_config = load_exclusion_config(exclusion_config_path)
    except FileNotFoundError:
        warnings.warn(f"Exclusion config file not found: {exclusion_config_path}. "
                     "Continuing without filtering.")
        return (train_sequences, train_configs,
                val_sequences, val_configs,
                test_sequences, test_configs,
                {})

    validate_exclusion_config(exclusion_config)

    # Apply filtering to each split
    filter_stats = {}

    train_seq_f, train_cfg_f, train_stats = apply_exclusion_filter(
        train_sequences, train_configs, exclusion_config, 'train'
    )
    filter_stats['train'] = train_stats

    val_seq_f, val_cfg_f, val_stats = apply_exclusion_filter(
        val_sequences, val_configs, exclusion_config, 'val'
    )
    filter_stats['val'] = val_stats

    test_seq_f, test_cfg_f, test_stats = apply_exclusion_filter(
        test_sequences, test_configs, exclusion_config, 'test'
    )
    filter_stats['test'] = test_stats

    # Safety checks
    if len(train_seq_f) == 0:
        raise ValueError(
            "All training data was excluded! Please adjust your exclusion rules."
        )

    # Warn if too much data excluded
    if train_stats['total_before'] > 0:
        train_pct = train_stats['excluded_count'] / train_stats['total_before']
        if train_pct > 0.5:
            warnings.warn(
                f"More than 50% of training data excluded ({train_pct*100:.1f}%). "
                f"Consider adjusting exclusion rules."
            )

    # Print comprehensive report
    print_exclusion_report(filter_stats)

    return (train_seq_f, train_cfg_f,
            val_seq_f, val_cfg_f,
            test_seq_f, test_cfg_f,
            filter_stats)


def _get_channel_name(channel_id: int) -> str:
    """Get channel model letter name from ID.

    Args:
        channel_id: Channel model ID (1-6)

    Returns:
        Letter name (A-F) or string of ID if unknown
    """
    names = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}
    return names.get(channel_id, str(channel_id))


def generate_random_mcs_exclusions(
    train_configs: List[dict],
    val_configs: List[dict],
    percentage: float,
    random_seed: int = 42
) -> dict:
    """Generate random MCS exclusions per configuration slice.

    This function analyzes the training data distribution and randomly selects
    X% of MCS values to exclude for each unique configuration slice. The random
    selection is reproducible via the random_seed parameter.

    Args:
        train_configs: Training configuration dicts
        val_configs: Validation configuration dicts
        percentage: Percentage of MCS values to exclude per slice (0-100)
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        Exclusion config dict in the same format as YAML configs, containing:
        - version: Config schema version
        - metadata: Generation details (percentage, seed, timestamp)
        - settings: Which splits to apply exclusions to
        - exclusions: List of exclusion rules

    Algorithm:
        1. Combine train+val configs to analyze full training distribution
        2. Group by config slice (channel_model_id, N_t, N_r, BW, N_ss)
        3. For each slice, collect unique MCS values
        4. Randomly select X% of MCS values to exclude
        5. Generate exclusion rules in YAML-compatible format

    Example:
        >>> random_config = generate_random_mcs_exclusions(
        ...     train_configs, val_configs, percentage=30.0, random_seed=42
        ... )
        >>> print(len(random_config['exclusions']))
        15  # 15 unique config slices found
    """
    import numpy as np
    from datetime import datetime

    np.random.seed(random_seed)

    # Define slice keys (exclude MCS and SNR_bar which vary within slices)
    slice_keys = ['channel_model_id', 'N_t', 'N_r', 'BW', 'N_ss']

    # Group configs by slice and collect MCS values
    slice_mcs_map = defaultdict(set)

    for cfg in train_configs + val_configs:
        # Create slice signature (tuple of key-value pairs)
        slice_sig = tuple((k, cfg[k]) for k in slice_keys if k in cfg)
        slice_mcs_map[slice_sig].add(cfg['MCS'])

    # Print analysis header
    print(f"\nGenerating random {percentage}% MCS exclusions (seed={random_seed})")
    print(f"  Analyzing training configurations...")
    print(f"  Found {len(slice_mcs_map)} unique configuration slices")

    # Generate exclusion rules
    exclusion_rules = []

    for slice_idx, (slice_sig, mcs_set) in enumerate(sorted(slice_mcs_map.items()), 1):
        mcs_list = sorted(list(mcs_set))

        # Calculate how many MCS to exclude
        num_to_exclude = max(1, round(len(mcs_list) * percentage / 100))

        # Ensure we don't exclude all MCS values
        if num_to_exclude >= len(mcs_list):
            num_to_exclude = len(mcs_list) - 1 if len(mcs_list) > 1 else 0

        # Skip if no MCS to exclude
        if num_to_exclude == 0:
            continue

        # Randomly select MCS values to exclude
        excluded_mcs = sorted(
            np.random.choice(mcs_list, size=num_to_exclude, replace=False).tolist()
        )

        # Get training MCS (remaining)
        training_mcs = sorted(list(set(mcs_list) - set(excluded_mcs)))

        # Create config dict from slice signature
        config_dict = dict(slice_sig)

        # Print detailed per-slice info
        print(f"\n  Slice {slice_idx}/{len(slice_mcs_map)}: " +
              f"Model-{_get_channel_name(config_dict['channel_model_id'])} " +
              f"{config_dict['N_t']}x{config_dict['N_r']}:{config_dict['N_ss']}, " +
              f"BW={config_dict['BW']}MHz")
        print(f"    Available MCS: {mcs_list} ({len(mcs_list)} total)")
        print(f"    Excluding {len(excluded_mcs)} MCS ({percentage}%): {excluded_mcs}")
        print(f"    Training on: {training_mcs}")

        # Create exclusion rule
        rule = {
            'name': f"Random {percentage}% exclusion for " +
                   f"Model-{_get_channel_name(config_dict['channel_model_id'])} " +
                   f"{config_dict['N_t']}x{config_dict['N_r']}:{config_dict['N_ss']}",
            'description': f"Auto-generated random exclusion (seed={random_seed})",
            'config': config_dict,
            'mcs_list': excluded_mcs
        }
        exclusion_rules.append(rule)

    # Print summary
    print(f"\nGenerated {len(exclusion_rules)} exclusion rules")

    # Build full config structure
    exclusion_config = {
        'version': '1.0',
        'metadata': {
            'generated_by': 'random_mcs_exclusion',
            'percentage': percentage,
            'random_seed': random_seed,
            'timestamp': datetime.now().isoformat(),
            'total_slices': len(slice_mcs_map),
            'total_rules': len(exclusion_rules)
        },
        'settings': {
            'apply_to_train': True,
            'apply_to_val': True,
            'apply_to_test': False
        },
        'exclusions': exclusion_rules
    }

    return exclusion_config


def save_exclusion_config(config: dict, output_path: str) -> None:
    """Save exclusion config to YAML file.

    Args:
        config: Exclusion configuration dictionary
        output_path: Path to save YAML file

    Raises:
        ImportError: If pyyaml not installed
    """
    if not YAML_AVAILABLE:
        raise ImportError(
            "pyyaml is required to save exclusion configs. "
            "Install with: pip install pyyaml"
        )

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Saved random exclusion config to: {output_path}")


def merge_exclusion_configs(config1: dict, config2: dict) -> dict:
    """Merge two exclusion configurations.

    This combines exclusion rules from two configs, typically a random-generated
    config and a manual YAML config.

    Args:
        config1: First exclusion config (e.g., random-generated)
        config2: Second exclusion config (e.g., manual YAML)

    Returns:
        Merged configuration with combined exclusion rules

    Note:
        - Settings from config1 take precedence
        - Exclusion rules are concatenated (both sets applied)
        - Duplicate detection warns but allows duplicates

    Example:
        >>> random_config = generate_random_mcs_exclusions(...)
        >>> manual_config = load_exclusion_config('manual.yaml')
        >>> merged = merge_exclusion_configs(random_config, manual_config)
        >>> len(merged['exclusions']) == len(random_config['exclusions']) + len(manual_config['exclusions'])
        True
    """
    from datetime import datetime

    merged = {
        'version': config1.get('version', '1.0'),
        'metadata': {
            'merged': True,
            'merge_timestamp': datetime.now().isoformat(),
            'config1_metadata': config1.get('metadata', {}),
            'config2_metadata': config2.get('metadata', {})
        },
        'settings': config1.get('settings', {}),
        'exclusions': config1.get('exclusions', []) + config2.get('exclusions', [])
    }

    print(f"Merged {len(config1.get('exclusions', []))} random rules + " +
          f"{len(config2.get('exclusions', []))} manual rules = " +
          f"{len(merged['exclusions'])} total rules")

    return merged


def save_exclusion_manifest(
    config: dict,
    output_path: str,
    train_configs: List[dict],
    val_configs: List[dict]
) -> None:
    """Generate JSON manifest summarizing held-out configurations.

    This creates a machine-readable summary for easy testing and analysis
    of which MCS values are held-out for each configuration slice.

    Args:
        config: Exclusion configuration dictionary
        output_path: Path to save JSON file
        train_configs: Training configurations (for context)
        val_configs: Validation configurations (for context)

    Example JSON output:
        {
          "metadata": {"percentage": 30.0, "random_seed": 42, ...},
          "held_out_slices": [
            {
              "slice": {"channel_model_id": 2, "N_t": 3, ...},
              "available_mcs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
              "excluded_mcs": [2, 5, 7],
              "training_mcs": [0, 1, 3, 4, 6, 8, 9]
            },
            ...
          ]
        }
    """
    import json

    # Define slice keys
    slice_keys = ['channel_model_id', 'N_t', 'N_r', 'BW', 'N_ss']

    # Analyze each exclusion rule
    held_out_slices = []

    for rule in config.get('exclusions', []):
        slice_config = rule['config']
        excluded_mcs = rule['mcs_list']

        # Determine available MCS for this slice from training data
        available_mcs = set()
        for cfg in train_configs + val_configs:
            # Check if config matches this slice
            if all(cfg.get(k) == v for k, v in slice_config.items()):
                available_mcs.add(cfg['MCS'])

        training_mcs = sorted(list(available_mcs - set(excluded_mcs)))

        held_out_slices.append({
            'slice': slice_config,
            'available_mcs': sorted(list(available_mcs)),
            'excluded_mcs': excluded_mcs,
            'training_mcs': training_mcs
        })

    manifest = {
        'metadata': config.get('metadata', {}),
        'held_out_slices': held_out_slices
    }

    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved exclusion manifest to: {output_path}")
