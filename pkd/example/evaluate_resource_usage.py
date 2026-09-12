"""Resource-usage runtime evaluation: PKD vs. EESM-log-AR (Python), CPU
utilization + peak memory alongside wall-clock time, AVERAGED OVER ALL SNR
OPERATING POINTS for the given (slice, MCS) -- matching Table III/IV's
convention (Table II: "SNR: 10 operating points along the PER-SNR waterfall
curve"; phy/runtime-benchmark/main.m loops isnr = 1:10 and reports
tAvg = tEnd/numSnr*numCores; pkd/example/evaluate_runtime.py reports the
average total runtime across 10 SNR values).

This script extends that runtime comparison in two ways:

1. It adds a free-running EESM-log-AR generator (pkd.baselines.generate,
   the same one used for the sparse-MCS baseline comparisons in
   Figs. 12-14) as a second timed method alongside PKD, calibrated once
   (offline, untimed) PER SNR POINT on teacher training sequences
   (mirroring how the MATLAB traditional-abstraction runtime table
   excludes offline beta calibration from the timed section), then timed
   only on its free-running generation -- mirroring exactly what is timed
   for PKD (PKDInference.run_sequence).

   Unlike PKD (queried at any SNR via its continuous conditioning network)
   or phy/runtime-benchmark/main.m's arbitrary linspace(10,55,10) grid
   (irrelevant there since PKD needs no real data at those points),
   EESM-log-AR MUST be calibrated from real teacher sequences, so its SNR
   grid is the dataset's actual waterfall operating points for this
   (slice, MCS) -- the same points phy/runtime-benchmark/main.m indexes
   via isnr = 1:10. PKD is timed at that SAME set of SNR values so the two
   methods' averages are computed over identical operating points.

2. Both methods run under a background psutil sampling thread recording
   RSS and CPU utilization THROUGHOUT THE WHOLE SNR LOOP (one continuous
   timed region per method, no parfor/subprocess-per-SNR), reporting
   peak/mean alongside the existing wall-clock/throughput numbers, plus
   per-SNR breakdown and the cross-SNR average matching
   evaluate_runtime.py's "Average total runtime" convention. GPU
   utilization/memory via torch.cuda for PKD when CUDA is available.

Usage:
    python -m pkd.example.evaluate_resource_usage \
        --channel-model 2 --N-t 3 --N-r 2 --BW 40.0 --N-ss 2 --MCS 7 \
        [--num-snr 10] [--num-sequences 50] [--sequence-length 1000] \
        [--sample-interval 0.01] [--data-dir data] [--max-files N] \
        [--method {both,pkd,eesm}]
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

    def __init__(self, interval_s=0.01, device='cpu'):
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
# Dataset SNR grid
# ----------------------------------------------------------------------------

def get_snr_grid(data_dir, max_files, slice_spec, mcs):
    """The dataset's actual SNR operating points for this (slice, MCS),
    e.g. the 10-point PER-SNR waterfall grid (Table II). Loaded once from
    the training split; both EESM-log-AR calibration and PKD timing use
    this same grid so their per-SNR averages are over identical points.
    """
    train_sequences, train_configs, _, _, _, _ = load_real_data(
        data_dir=data_dir, train_ratio=0.7, val_ratio=0.1, max_files=max_files, random_seed=42,
    )
    full_spec = dict(slice_spec)
    full_spec['MCS'] = mcs
    matched_seq, matched_cfg, _ = filter_by_slice(train_sequences, train_configs, full_spec)
    if len(matched_seq) == 0:
        raise ValueError(
            f"No training sequences found for slice={slice_spec}, MCS={mcs}. "
            "Check --channel-model/--N-t/--N-r/--BW/--N-ss/--MCS match the dataset."
        )
    snr_values = sorted(set(c['SNR_bar'] for c in matched_cfg))
    per_snr_seq = {
        snr: [matched_seq[i] for i, c in enumerate(matched_cfg) if c['SNR_bar'] == snr]
        for snr in snr_values
    }
    return snr_values, per_snr_seq


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


def time_pkd_all_snr(base_config, snr_values, num_sequences, sequence_length,
                      model_path, device, sample_interval):
    """Time PKD generation across ALL snr_values in one continuous timed
    region (one process, sequential SNR loop, no parfor/subprocess-per-SNR)
    so peak/mean CPU and RSS are sampled over the whole sweep, matching
    main.m's single-metric-per-configuration convention. Returns per-SNR
    elapsed times plus the whole-sweep resource summary.
    """
    model, model_config = load_pkd_model(model_path, device)
    per_lut = AWGNPERLookup.load_ldpc_lut()
    inference = PKDInference(model, per_lut, ar_order=model_config['ar_order'], device=device)

    per_snr_elapsed = {}
    with ResourceSampler(interval_s=sample_interval, device=device) as sampler:
        for snr in snr_values:
            config = dict(base_config)
            config['SNR_bar'] = float(snr)
            config_trajectory = [config] * sequence_length

            t0 = time.time()
            for _ in range(num_sequences):
                results = inference.run_sequence(config_trajectory)
                gamma_eff_db = results['gamma_eff']
                per_sequence = per_lut.lookup(gamma_eff_db, config['MCS'],
                                               packet_length=config.get('packet_length', 1000))
                _ = np.random.rand(len(per_sequence)) <= per_sequence
            per_snr_elapsed[snr] = time.time() - t0

    return per_snr_elapsed, sampler.summary()


# ----------------------------------------------------------------------------
# EESM-log-AR timing
# ----------------------------------------------------------------------------

def calibrate_eesm_log_ar_all_snr(per_snr_seq, ar_order=AR_ORDER):
    """Offline (untimed) calibration: fit (mu, phi, sigma) separately at
    EACH SNR operating point from training-split teacher sequences,
    mirroring how the MATLAB traditional-abstraction runtime table
    excludes offline beta calibration from the timed section (beta is
    also calibrated once per configuration, not per timed run).
    """
    params_by_snr = {}
    for snr, seqs in per_snr_seq.items():
        params_by_snr[snr] = calibrate_ar_params(seqs, ar_order=ar_order)
    return params_by_snr


def time_eesm_log_ar_all_snr(params_by_snr, mcs, num_sequences, sequence_length,
                              sample_interval, ar_order=AR_ORDER, packet_length=1000):
    """Time EESM-log-AR generation across ALL SNR points in one continuous
    timed region, mirroring time_pkd_all_snr."""
    per_lut = AWGNPERLookup.load_ldpc_lut()
    rng = np.random.default_rng(0)

    per_snr_elapsed = {}
    with ResourceSampler(interval_s=sample_interval, device='cpu') as sampler:
        for snr, params in params_by_snr.items():
            t0 = time.time()
            for _ in range(num_sequences):
                X = generate_ar_sequence(params, length=sequence_length, ar_order=ar_order,
                                          burn_in=50, rng=rng)
                gamma_eff_db = X * 10 / np.log(10)
                per_sequence = per_lut.lookup(gamma_eff_db, mcs, packet_length=packet_length)
                _ = np.random.rand(len(per_sequence)) <= per_sequence
            per_snr_elapsed[snr] = time.time() - t0

    return per_snr_elapsed, sampler.summary()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='PKD vs. EESM-log-AR resource-usage runtime evaluation, averaged over SNR',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    p.add_argument('--channel-model', type=int, required=True)
    p.add_argument('--N-t', type=int, required=True)
    p.add_argument('--N-r', type=int, required=True)
    p.add_argument('--BW', type=float, required=True)
    p.add_argument('--N-ss', type=int, required=True)
    p.add_argument('--MCS', type=int, required=True)
    p.add_argument('--num-snr', type=int, default=None,
                   help='If given, use only the first N SNR points of the dataset '
                        'waterfall grid for this (slice, MCS) (default: all available, '
                        'typically 10, matching Table II).')
    p.add_argument('--snr-list', type=float, nargs='+', default=None,
                   help="Explicit SNR values (dB) to time PKD at, e.g. "
                        "--snr-list 26 29 32. ONLY valid with --method pkd: lets "
                        "PKD be timed WITHOUT loading the training dataset at all "
                        "(unlike the default, which loads it to discover the real "
                        "waterfall SNR grid), for a peak-RSS number isolated from "
                        "dataset-load overhead. Get the real grid once via "
                        "--method eesm (it prints 'N SNR points: [...]'), then "
                        "pass those same values here for an apples-to-apples "
                        "average against that eesm run.")
    p.add_argument('--num-sequences', type=int, default=50)
    p.add_argument('--sequence-length', type=int, default=1000)
    p.add_argument('--sample-interval', type=float, default=0.01,
                   help='Resource-sampling interval in seconds (default 0.01). '
                        'Use a value well below the expected per-SNR wall-clock '
                        'time so each SNR point collects enough samples.')
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


def fmt_summary(label, per_snr_elapsed, n_seq, seq_len, summary):
    snr_values = list(per_snr_elapsed.keys())
    elapsed_values = list(per_snr_elapsed.values())
    total_elapsed = sum(elapsed_values)
    avg_elapsed = np.mean(elapsed_values)
    std_elapsed = np.std(elapsed_values)
    total_samples_per_snr = n_seq * seq_len

    lines = [f'{label}:']
    lines.append(f'  SNR points:           {len(snr_values)}  ({min(snr_values):.1f} to {max(snr_values):.1f} dB)')
    for snr, e in per_snr_elapsed.items():
        lines.append(f'    SNR={snr:6.2f} dB:  {e:.4f} s')
    lines += [
        f'  Average per-SNR time: {avg_elapsed:.4f} s  (std {std_elapsed:.4f} s)  '
        f'[{avg_elapsed/n_seq:.4f} s/sequence, {avg_elapsed/total_samples_per_snr*1e3:.4f} ms/packet]',
        f'  Total wall-clock:     {total_elapsed:.4f} s  (sum over all SNR points)',
        f'  Throughput:           {total_samples_per_snr/avg_elapsed:.1f} packets/s (per-SNR average)',
        f'  Peak RSS (memory):    {summary["peak_rss_mb"]:.1f} MB  (whole SNR sweep)',
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
    print('Resource-Usage Runtime Evaluation: PKD vs. EESM-log-AR (avg over SNR)')
    print('=' * 70)
    print(f'Configuration: Channel {channel_name} (id={args.channel_model}), '
          f'{args.N_t}x{args.N_r}:{args.N_ss}, BW={args.BW} MHz, MCS={args.MCS}')
    print(f'Sequences per SNR: {args.num_sequences} x {args.sequence_length} packets')
    print(f'Device (PKD): {device}   |   Logical CPU cores: {os.cpu_count()}')

    if args.snr_list is not None and args.method != 'pkd':
        raise SystemExit("--snr-list is only valid with --method pkd (EESM-log-AR "
                          "must calibrate from real data, so it always needs the "
                          "dataset load; use --method eesm to get the real grid, "
                          "then pass those same values via --snr-list --method pkd).")

    if args.snr_list is not None:
        # Skip the dataset load entirely: PKD can be queried at any SNR via its
        # continuous conditioning network, so this path gives a peak-RSS number
        # isolated from dataset-load overhead (~+200 MB, confirmed empirically,
        # that persists even after gc.collect() since CPython/glibc do not
        # reliably return freed heap pages to the OS). Pass the SAME values
        # reported by a prior --method eesm run for an apples-to-apples average.
        snr_values = sorted(args.snr_list)
        if args.num_snr is not None:
            snr_values = snr_values[:args.num_snr]
        print(f'\nUsing explicit --snr-list (no dataset load): {len(snr_values)} SNR points: '
              f'{[f"{s:.1f}" for s in snr_values]}')
        per_snr_seq = None
    else:
        # Loads the training split to discover the dataset's real waterfall SNR
        # grid -- unconditionally needed for EESM-log-AR (must calibrate from
        # real data) and, by default, also for PKD (so both methods are timed
        # at identical SNR points without you having to already know the grid).
        # This measurably raises RSS (see note above); use --snr-list --method
        # pkd instead if you need PKD's peak-RSS isolated from this load.
        print('\nLoading dataset SNR grid for this (slice, MCS)...')
        snr_values, per_snr_seq = get_snr_grid(args.data_dir, args.max_files, slice_spec, args.MCS)
        if args.num_snr is not None:
            snr_values = snr_values[:args.num_snr]
            per_snr_seq = {k: v for k, v in per_snr_seq.items() if k in snr_values}
        print(f'  {len(snr_values)} SNR points: {[f"{s:.1f}" for s in snr_values]}')

    eesm_per_snr = eesm_summary = pkd_per_snr = pkd_summary = None

    if args.method in ('both', 'eesm'):
        print('\nCalibrating EESM-log-AR at each SNR point (offline, untimed)...')
        params_by_snr = calibrate_eesm_log_ar_all_snr(per_snr_seq, ar_order=AR_ORDER)
        for snr, p in params_by_snr.items():
            print(f'  SNR={snr:.2f} dB: mu={p.mu:.4f}, sigma={p.sigma:.4f}, '
                  f'sum(phi)={np.sum(p.phi):.4f}  ({len(per_snr_seq[snr])} sequences)')

        print('\nTiming EESM-log-AR free-running generation across all SNR points...')
        eesm_per_snr, eesm_summary = time_eesm_log_ar_all_snr(
            params_by_snr, args.MCS, args.num_sequences, args.sequence_length, args.sample_interval,
        )
        eesm_summary['interval'] = f'{args.sample_interval}s'

    if args.method in ('both', 'pkd'):
        print('\nTiming PKD inference across all SNR points...')
        pkd_base_config = {
            'channel_model_id': args.channel_model, 'N_t': args.N_t, 'N_r': args.N_r,
            'BW': float(args.BW), 'MCS': args.MCS, 'N_ss': args.N_ss,
            'R_t': 0, 'packet_length': 1000,
        }
        pkd_per_snr, pkd_summary = time_pkd_all_snr(
            pkd_base_config, snr_values, args.num_sequences, args.sequence_length,
            args.model_path, device, args.sample_interval,
        )
        pkd_summary['interval'] = f'{args.sample_interval}s'

    # ---- Report ----
    print('\n' + '=' * 70)
    if args.method == 'both':
        print(f'Results (same {len(snr_values)} SNR points, same slice/MCS/sequence count)')
    else:
        print(f'Results ({args.method})')
    print('=' * 70)
    if eesm_summary is not None:
        print(fmt_summary('EESM-log-AR (Python, calibrated (mu,phi,sigma) per SNR)', eesm_per_snr,
                           args.num_sequences, args.sequence_length, eesm_summary))
        print()
    if pkd_summary is not None:
        print(fmt_summary(f'PKD ({device})', pkd_per_snr,
                           args.num_sequences, args.sequence_length, pkd_summary))
        print()

    if args.method == 'both':
        eesm_avg = np.mean(list(eesm_per_snr.values()))
        pkd_avg = np.mean(list(pkd_per_snr.values()))
        ratio = pkd_avg / eesm_avg if eesm_avg > 0 else float('nan')
        print(f'PKD / EESM-log-AR average-per-SNR wall-clock ratio: {ratio:.2f}x '
              f'({"slower" if ratio > 1 else "faster"})')
        mem_ratio = pkd_summary['peak_rss_mb'] / eesm_summary['peak_rss_mb'] if eesm_summary['peak_rss_mb'] > 0 else float('nan')
        print(f'PKD / EESM-log-AR peak-RSS ratio:                    {mem_ratio:.2f}x')
        print('=' * 70)
        print('\nNote: with --method both, peak RSS is sampled for the whole '
              'Python process across BOTH timed regions (each spanning the full '
              'SNR sweep), which includes the CPython/NumPy/PyTorch/h5py runtime '
              'and the loaded dataset shared by both -- it is NOT an isolated '
              'per-method allocation, and the two methods also run in a fixed '
              'order (EESM-log-AR first) so PKD\'s peak-RSS number includes '
              'whatever EESM-log-AR already allocated. For a paper-quality, '
              'isolated peak-RSS number per method, run this script twice in '
              'SEPARATE processes with --method pkd and --method eesm. Note '
              'that even --method pkd alone still loads the training split (to '
              'learn the SNR grid), so its peak-RSS is not isolated from that '
              'load either -- see the note printed above the SNR grid. The '
              'wall-clock/throughput/CPU-utilization numbers are not affected '
              'by any of this and are safe to compare directly.')
    else:
        print('=' * 70)


if __name__ == '__main__':
    main()
