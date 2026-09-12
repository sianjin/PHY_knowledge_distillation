"""Resource-usage runtime evaluation: PKD vs. EESM-log-AR (Python), CPU
utilization + peak memory alongside wall-clock time.

Addresses two reviewer comments on the runtime table (Table III/IV):

1. "Table III compares PKD only with traditional PHY abstraction. Including
   EESM-log-AR inference runtime would more directly isolate the
   scalability gain over stochastic abstraction." -- This script adds a
   free-running EESM-log-AR generator (pkd.baselines.generate, the same
   one used for the sparse-MCS baseline comparisons in Figs. 12-14) as a
   second timed method, calibrated once (offline, untimed) on teacher
   training sequences at the target (slice, MCS, SNR), then timed only on
   its free-running generation -- mirroring exactly what is timed for PKD
   (PKDInference.run_sequence), and mirroring how the MATLAB traditional-
   abstraction runtime table excludes offline beta calibration from the
   timed section (see phy/runtime-benchmark/README.md).

2. "The authors do not measure CPU/GPU utilization and memory footprint...
   CPU-only runtime does not fully characterize the computational
   requirements." -- Both methods are run under a background psutil
   sampling thread that records RSS (resident set size) and per-core CPU
   utilization at a fixed interval throughout the timed region, reporting
   peak/mean alongside the existing wall-clock and throughput numbers. If
   CUDA is available, GPU utilization and peak allocated memory are also
   reported for PKD via torch.cuda.

Both methods are evaluated on identical (slice, MCS, SNR) points and
identical (num_sequences, sequence_length), so the comparison isolates the
inference-time cost of the neural PKD student vs. the classical calibrated
EESM-log-AR process -- both bypass channel/waveform simulation equally;
the difference measured here is the added cost of the neural conditioning
network relative to a directly-stored (mu, phi, sigma) parameter tuple.

Usage:
    python -m pkd.example.evaluate_resource_usage \
        --channel-model 2 --N-t 3 --N-r 2 --BW 40.0 --N-ss 2 --MCS 7 \
        [--snr 20.0] [--num-sequences 50] [--sequence-length 1000] \
        [--sample-interval 0.05] [--data-dir data] [--max-files N]
"""
import argparse
import os
import threading
import time

import numpy as np
import torch

try:
    import psutil
except ImportError as e:
    raise ImportError(
        "evaluate_resource_usage requires psutil (pip install psutil) to "
        "measure CPU utilization and memory footprint."
    ) from e

from pkd.model import PKDModel
from pkd.per_lut import AWGNPERLookup
from pkd.infer import PKDInference
from pkd.baselines.calibration import calibrate_ar_params
from pkd.baselines.generate import generate_ar_sequence
from pkd.example.data_loader import load_real_data
from pkd.example.utils import filter_by_slice

CHANNEL_MODEL_NAMES = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}
AR_ORDER = 10


# ----------------------------------------------------------------------------
# Resource sampling
# ----------------------------------------------------------------------------

class ResourceSampler:
    """Background thread sampling this process's RSS and CPU utilization
    at a fixed interval, plus GPU utilization/memory via torch.cuda if
    available. Start before the timed region, stop immediately after.
    """

    def __init__(self, interval_s=0.05, device='cpu'):
        self.interval_s = interval_s
        self.device = device
        self._proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread = None
        self.rss_samples = []
        self.cpu_samples = []  # percent, 100 = one full core
        self.gpu_util_samples = []
        self.gpu_mem_samples = []

    def _run(self):
        # Prime cpu_percent (first call after interval=None returns 0.0/garbage)
        self._proc.cpu_percent(interval=None)
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        while not self._stop.is_set():
            self.rss_samples.append(self._proc.memory_info().rss)
            self.cpu_samples.append(self._proc.cpu_percent(interval=None))
            if self.device == 'cuda' and torch.cuda.is_available():
                try:
                    self.gpu_util_samples.append(torch.cuda.utilization())
                except Exception:
                    pass
                self.gpu_mem_samples.append(torch.cuda.memory_allocated())
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2 * self.interval_s + 1.0)

    def summary(self):
        out = {
            'peak_rss_mb': (max(self.rss_samples) / 1e6) if self.rss_samples else float('nan'),
            'mean_cpu_pct': (np.mean(self.cpu_samples[1:]) if len(self.cpu_samples) > 1
                             else (self.cpu_samples[0] if self.cpu_samples else float('nan'))),
            'peak_cpu_pct': (max(self.cpu_samples) if self.cpu_samples else float('nan')),
            'n_samples': len(self.rss_samples),
        }
        if self.gpu_mem_samples:
            out['peak_gpu_mem_mb'] = max(self.gpu_mem_samples) / 1e6
        if self.gpu_util_samples:
            out['mean_gpu_util_pct'] = float(np.mean(self.gpu_util_samples))
        return out


# ----------------------------------------------------------------------------
# PKD timing
# ----------------------------------------------------------------------------

def load_pkd_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get('model_config', {
        'num_channel_models': 5, 'num_mcs': 10, 'num_nss': 4, 'num_R': 8,
        'ar_order': AR_ORDER, 'hidden_dim': 128, 'kappa_max': 0.95,
        'innovation_type': 'gaussian', 'min_sigma': 0.1,
    })
    model_config.setdefault('kappa_max', 0.95)
    model_config.setdefault('innovation_type', 'gaussian')
    model_config.setdefault('min_sigma', 0.1)
    model_config.setdefault('num_R', 8)
    model = PKDModel(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, model_config


def time_pkd(config, num_sequences, sequence_length, model_path, device, sample_interval):
    model, model_config = load_pkd_model(model_path, device)
    per_lut = AWGNPERLookup.load_ldpc_lut()
    inference = PKDInference(model, per_lut, ar_order=model_config['ar_order'], device=device)
    config_trajectory = [config] * sequence_length

    with ResourceSampler(interval_s=sample_interval, device=device) as sampler:
        t0 = time.time()
        for _ in range(num_sequences):
            results = inference.run_sequence(config_trajectory)
            gamma_eff_db = results['gamma_eff']
            per_sequence = per_lut.lookup(gamma_eff_db, config['MCS'], packet_length=config.get('packet_length', 1000))
            _ = np.random.rand(len(per_sequence)) <= per_sequence
        elapsed = time.time() - t0

    return elapsed, sampler.summary()


# ----------------------------------------------------------------------------
# EESM-log-AR timing
# ----------------------------------------------------------------------------

def calibrate_eesm_log_ar(data_dir, max_files, slice_spec, mcs, snr, ar_order=AR_ORDER):
    """Offline (untimed) calibration: fit (mu, phi, sigma) from training-
    split teacher sequences at the exact (slice, MCS, SNR) cell, mirroring
    how the MATLAB traditional-abstraction runtime table excludes beta
    calibration from the timed section.
    """
    train_sequences, train_configs, _, _, _, _ = load_real_data(
        data_dir=data_dir, train_ratio=0.7, val_ratio=0.1, max_files=max_files, random_seed=42,
    )
    full_spec = dict(slice_spec)
    full_spec['MCS'] = mcs
    matched_seq, matched_cfg, _ = filter_by_slice(train_sequences, train_configs, full_spec)
    if snr is not None:
        keep = [i for i, c in enumerate(matched_cfg) if abs(c['SNR_bar'] - snr) < 1e-6]
        matched_seq = [matched_seq[i] for i in keep]
        matched_cfg = [matched_cfg[i] for i in keep]
    if len(matched_seq) == 0:
        raise ValueError(
            f"No training sequences found for slice={slice_spec}, MCS={mcs}, SNR={snr}. "
            "Check --channel-model/--N-t/--N-r/--BW/--N-ss/--MCS/--snr match the dataset."
        )
    snr_used = matched_cfg[0]['SNR_bar']
    params = calibrate_ar_params(matched_seq, ar_order=ar_order)
    return params, snr_used, len(matched_seq)


def time_eesm_log_ar(params, mcs, num_sequences, sequence_length, sample_interval, ar_order=AR_ORDER, packet_length=1000):
    per_lut = AWGNPERLookup.load_ldpc_lut()
    rng = np.random.default_rng(0)

    with ResourceSampler(interval_s=sample_interval, device='cpu') as sampler:
        t0 = time.time()
        for _ in range(num_sequences):
            X = generate_ar_sequence(params, length=sequence_length, ar_order=ar_order, burn_in=50, rng=rng)
            gamma_eff_db = X * 10 / np.log(10)
            per_sequence = per_lut.lookup(gamma_eff_db, mcs, packet_length=packet_length)
            _ = np.random.rand(len(per_sequence)) <= per_sequence
        elapsed = time.time() - t0

    return elapsed, sampler.summary()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='PKD vs. EESM-log-AR resource-usage runtime evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    p.add_argument('--channel-model', type=int, required=True)
    p.add_argument('--N-t', type=int, required=True)
    p.add_argument('--N-r', type=int, required=True)
    p.add_argument('--BW', type=float, required=True)
    p.add_argument('--N-ss', type=int, required=True)
    p.add_argument('--MCS', type=int, required=True)
    p.add_argument('--snr', type=float, default=None,
                   help='If omitted, uses whichever SNR the matched training sequences carry (single value expected).')
    p.add_argument('--num-sequences', type=int, default=50)
    p.add_argument('--sequence-length', type=int, default=1000)
    p.add_argument('--sample-interval', type=float, default=0.01,
                   help='Resource-sampling interval in seconds (default 0.01). '
                        'Use a value well below the expected per-run wall-clock '
                        'time so each timed run collects enough samples for a '
                        'meaningful mean/peak CPU utilization.')
    p.add_argument('--model-path', type=str, default='pkd/pkd_model.pt')
    p.add_argument('--data-dir', type=str, default='data')
    p.add_argument('--max-files', type=int, default=None)
    p.add_argument('--method', choices=['both', 'pkd', 'eesm'], default='both',
                   help="Which method to time. 'both' (default) times PKD and "
                        "EESM-log-AR back-to-back in one process for a quick "
                        "wall-clock/CPU-utilization comparison, but their peak-RSS "
                        "numbers then share one process's baseline overhead (see "
                        "the printed note). For a paper-quality, per-method "
                        "isolated peak-RSS number, run this script twice in "
                        "separate processes with --method pkd and --method eesm.")
    return p.parse_args()


def fmt_summary(label, elapsed, n_seq, seq_len, summary):
    total_samples = n_seq * seq_len
    lines = [
        f'{label}:',
        f'  Wall-clock time:      {elapsed:.4f} s  ({elapsed/n_seq:.4f} s/sequence, '
        f'{elapsed/total_samples*1e3:.4f} ms/packet)',
        f'  Throughput:           {total_samples/elapsed:.1f} packets/s',
        f'  Peak RSS (memory):    {summary["peak_rss_mb"]:.1f} MB',
        f'  CPU utilization:      mean {summary["mean_cpu_pct"]:.1f}%, peak {summary["peak_cpu_pct"]:.1f}% '
        f'(100% = 1 core; {os.cpu_count()} logical cores available)',
    ]
    if 'peak_gpu_mem_mb' in summary:
        lines.append(f'  Peak GPU memory:      {summary["peak_gpu_mem_mb"]:.1f} MB')
    if 'mean_gpu_util_pct' in summary:
        lines.append(f'  Mean GPU utilization: {summary["mean_gpu_util_pct"]:.1f}%')
    lines.append(f'  Resource samples:     {summary["n_samples"]} (interval {summary.get("interval", "?")})')
    return '\n'.join(lines)


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    slice_spec = {
        'channel_model_id': args.channel_model,
        'N_t': args.N_t,
        'N_r': args.N_r,
        'BW': float(args.BW),
        'N_ss': args.N_ss,
    }
    channel_name = CHANNEL_MODEL_NAMES.get(args.channel_model, 'Unknown')

    print('=' * 70)
    print('Resource-Usage Runtime Evaluation: PKD vs. EESM-log-AR')
    print('=' * 70)
    print(f'Configuration: Channel {channel_name} (id={args.channel_model}), '
          f'{args.N_t}x{args.N_r}:{args.N_ss}, BW={args.BW} MHz, MCS={args.MCS}')
    print(f'Sequences: {args.num_sequences} x {args.sequence_length} packets')
    print(f'Device (PKD): {device}   |   Logical CPU cores: {os.cpu_count()}')

    # ---- Offline (untimed) EESM-log-AR calibration ----
    # Loads the full training-split dataset (~200k sequences), which by
    # itself measurably raises this process's RSS (confirmed: +200 MB even
    # after gc.collect(), since CPython/glibc do not reliably return freed
    # heap pages to the OS). For an ISOLATED --method pkd peak-RSS number,
    # this step must be skipped entirely -- which requires --snr to be
    # given explicitly, since --snr is otherwise inferred from calibration.
    if args.method == 'pkd':
        if args.snr is None:
            raise SystemExit(
                "--method pkd requires --snr to be given explicitly, so this "
                "run can skip EESM-log-AR calibration (which loads the full "
                "training dataset and would inflate PKD's isolated peak-RSS "
                "measurement with unrelated dataset-loading memory). Pick "
                "any SNR present in the dataset for this slice/MCS, e.g. by "
                "first running --method eesm (or both) once to see the SNR "
                "value calibration reports for this configuration."
            )
        params, snr_used, n_calib_seq = None, args.snr, None
    else:
        print('\nCalibrating EESM-log-AR (offline, untimed, from training-split teacher sequences)...')
        params, snr_used, n_calib_seq = calibrate_eesm_log_ar(
            args.data_dir, args.max_files, slice_spec, args.MCS, args.snr, ar_order=AR_ORDER,
        )
        print(f'  Calibrated from {n_calib_seq} training sequences at SNR={snr_used:.2f} dB')
        print(f'  mu={params.mu:.4f}, sigma={params.sigma:.4f}, sum(phi)={np.sum(params.phi):.4f}')

    eesm_elapsed = eesm_summary = pkd_elapsed = pkd_summary = None

    if args.method in ('both', 'eesm'):
        print('\nTiming EESM-log-AR free-running generation...')
        eesm_elapsed, eesm_summary = time_eesm_log_ar(
            params, args.MCS, args.num_sequences, args.sequence_length, args.sample_interval,
        )
        eesm_summary['interval'] = f'{args.sample_interval}s'

    if args.method in ('both', 'pkd'):
        print('\nTiming PKD inference...')
        pkd_config = {
            'channel_model_id': args.channel_model, 'N_t': args.N_t, 'N_r': args.N_r,
            'BW': float(args.BW), 'SNR_bar': float(snr_used), 'MCS': args.MCS,
            'N_ss': args.N_ss, 'R_t': 0, 'packet_length': 1000,
        }
        pkd_elapsed, pkd_summary = time_pkd(
            pkd_config, args.num_sequences, args.sequence_length, args.model_path, device, args.sample_interval,
        )
        pkd_summary['interval'] = f'{args.sample_interval}s'

    # ---- Report ----
    print('\n' + '=' * 70)
    if args.method == 'both':
        print('Results (same slice/MCS/SNR/sequence count for both methods)')
    else:
        print(f'Results ({args.method})')
    print('=' * 70)
    if eesm_summary is not None:
        print(fmt_summary('EESM-log-AR (Python, calibrated (mu,phi,sigma))', eesm_elapsed,
                           args.num_sequences, args.sequence_length, eesm_summary))
        print()
    if pkd_summary is not None:
        print(fmt_summary(f'PKD ({device})', pkd_elapsed,
                           args.num_sequences, args.sequence_length, pkd_summary))
        print()

    if args.method == 'both':
        ratio = pkd_elapsed / eesm_elapsed if eesm_elapsed > 0 else float('nan')
        print(f'PKD / EESM-log-AR wall-clock time ratio: {ratio:.2f}x '
              f'({"slower" if ratio > 1 else "faster"})')
        mem_ratio = pkd_summary['peak_rss_mb'] / eesm_summary['peak_rss_mb'] if eesm_summary['peak_rss_mb'] > 0 else float('nan')
        print(f'PKD / EESM-log-AR peak-RSS ratio:         {mem_ratio:.2f}x')
        print('=' * 70)
        print('\nNote: with --method both, peak RSS is sampled for the whole '
              'Python process across BOTH timed regions, which includes the '
              'CPython/NumPy/PyTorch/h5py runtime and the loaded dataset '
              'shared by both -- it is NOT an isolated per-method allocation, '
              'and the two methods also run in a fixed order (EESM-log-AR '
              'first) so PKD\'s peak-RSS number includes whatever EESM-log-AR '
              'already allocated. For a paper-quality, isolated peak-RSS '
              'number per method, run this script twice in SEPARATE processes '
              'with --method pkd and --method eesm and compare those two runs\' '
              'peak-RSS values instead of the ratio printed here. The '
              'wall-clock/throughput/CPU-utilization numbers are not affected '
              'by this and are safe to compare directly from a --method both run.')
    else:
        print('=' * 70)


if __name__ == '__main__':
    main()
