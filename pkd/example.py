"""PKD Example Script - Entry Point

This is the main entry point that delegates to the refactored example/ modules.

Usage:
    python example.py train [N]           # Train with real data
    python example.py test [N]            # Evaluate on test set
    python example.py eval [idx] [N]      # Qualitative evaluation
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


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1]

    # Resolve data directory path relative to project root
    # Script is at pkd/example.py, data is at ../data/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_data_dir = os.path.join(project_root, 'data')

    if mode == 'train':
        # Train with real data
        max_files = int(sys.argv[2]) if len(sys.argv) > 2 else None
        data_dir = sys.argv[3] if len(sys.argv) > 3 else default_data_dir
        example_training(data_dir=data_dir, max_files=max_files)

    elif mode == 'test':
        # Comprehensive test set evaluation
        max_files = int(sys.argv[2]) if len(sys.argv) > 2 else None
        data_dir = sys.argv[3] if len(sys.argv) > 3 else default_data_dir
        example_test_evaluation(data_dir=data_dir, max_files=max_files)

    elif mode == 'eval':
        # Qualitative evaluation with real data
        test_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        max_files = int(sys.argv[3]) if len(sys.argv) > 3 else None
        data_dir = sys.argv[4] if len(sys.argv) > 4 else default_data_dir
        example_evaluation(data_dir=data_dir, test_idx=test_idx, max_files=max_files)

    else:
        print(f"Unknown mode: {mode}")
        print(__doc__)
        sys.exit(1)
