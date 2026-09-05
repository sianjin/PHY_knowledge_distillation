"""Fig. 9/11 analogs for full-tuple exclusion: single-test-sequence
qualitative comparison (CCDF/QQ marginal, ACF/PSD temporal) for the
representative excluded tuple at 30% exclusion. Mirrors the paper's
MCS-only Fig. 9/11 (pkd/example/main.py's example_evaluation), but
against an exclude_config_30 checkpoint and a fully-excluded
(CH, MCS, N_t, N_r, N_ss, BW) tuple rather than a single excluded MCS.

    PYTHONPATH=. python self_review/gen_fig_9_11.py
"""
import os

import numpy as np
import torch

from pkd.model import PKDModel
from pkd.per_lut import AWGNPERLookup
from pkd.infer import PKDInference

from pkd.example.data_loader import load_real_data
from pkd.example.utils import filter_by_slice
from pkd.example.evaluation import evaluate_marginal_distribution, evaluate_temporal_dependence

REPRESENTATIVE_TUPLE = {
    'channel_model_id': 2, 'N_t': 3, 'N_r': 2, 'BW': 40.0, 'N_ss': 2, 'MCS': 7,
}
CHECKPOINT_PATH = 'pkd/trained_models/exclude_config_30/pkd_model.pt'


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model_config = checkpoint['model_config']
    if 'num_R' not in model_config:
        model_config['num_R'] = 8
    model = PKDModel(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    per_lut = AWGNPERLookup.load_ldpc_lut()
    inference = PKDInference(model, per_lut, ar_order=model_config['ar_order'], device=device)

    print("Loading full dataset...")
    _, _, _, _, test_sequences, test_configs = load_real_data('data', random_seed=42)
    print(f"Loaded {len(test_sequences)} test sequences")

    filtered_sequences, filtered_configs, _ = filter_by_slice(
        test_sequences, test_configs, REPRESENTATIVE_TUPLE
    )
    print(f"Matching sequences for {REPRESENTATIVE_TUPLE}: {len(filtered_sequences)}")
    assert len(filtered_sequences) > 0, "Representative tuple has no matching test sequences"

    np.random.seed(42)
    selection_idx = np.random.randint(0, len(filtered_sequences))
    teacher_seq = filtered_sequences[selection_idx]
    config_dict = filtered_configs[selection_idx]
    print(f"Selected sequence index {selection_idx}/{len(filtered_sequences)-1}")

    config = {
        'channel_model_id': torch.tensor(config_dict['channel_model_id']),
        'N_t': torch.tensor(config_dict['N_t']),
        'N_r': torch.tensor(config_dict['N_r']),
        'BW': torch.tensor(config_dict['BW']),
        'SNR_bar': torch.tensor(config_dict['SNR_bar']),
        'MCS': torch.tensor(config_dict['MCS']),
        'N_ss': torch.tensor(config_dict['N_ss']),
        'R_t': torch.tensor(config_dict.get('R_t', 0)),
        'packet_length': torch.tensor(config_dict.get('packet_length', 1000)),
    }
    config_traj = [config] * len(teacher_seq)
    student_results = inference.run_sequence(config_traj)
    student_seq_db = student_results['gamma_eff']
    student_seq = student_seq_db * np.log(10) / 10

    teacher_log = teacher_seq
    student_log = student_seq

    os.makedirs('figures', exist_ok=True)
    print("\n--- Marginal Distribution (Fig. 9 analog) ---")
    marginal_metrics = evaluate_marginal_distribution(
        teacher_log, student_log, save_prefix='figures/tuple_marginal_30pct'
    )

    print("\n--- Temporal Dependence (Fig. 11 analog) ---")
    temporal_metrics = evaluate_temporal_dependence(
        teacher_log, student_log, save_prefix='figures/tuple_temporal_30pct'
    )

    print("\n-> figures/tuple_marginal_30pct_ccdf.png, _qq.png, _quantile_error.png (Fig. 9)")
    print("-> figures/tuple_temporal_30pct_acf.png, _psd.png (Fig. 11)")
    print(f"KS statistic: {marginal_metrics['ks_stat']:.4f}")
    print(f"ACF RMSE: {temporal_metrics['acf_rmse']:.4f}")
    print("DONE fig 9/11")


if __name__ == '__main__':
    main()
