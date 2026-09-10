"""Fig. 15 -- closed-loop rate adaptation under a time-varying configuration.

Addresses Reviewer 1: the framework is motivated by time-varying
configurations, yet every other experiment fixes C_t = C. Here the average
SNR follows a common deterministic trajectory {SNR_t} and the MCS is selected
packet-by-packet by a rate controller, so C_t != C.

The experiment replaces the PHY simulator with PKD in a closed rate-adaptation
loop and asks whether it preserves the *statistical* rate-adaptation
behaviour -- the MCS-selection probability over time and the achieved-goodput
distribution -- NOT whether it reproduces the same stochastic realization.

Teacher and student share:
  * the same time-varying SNR trajectory (phy/rate-control/snr_trajectory.mat)
  * the same rate-control policy (pkd.rate_control.RateController, a
    byte-for-byte twin of phy/rate-control/rateController.m)
The only difference is the source of effective SINR:
  * Teacher  -> MATLAB PHY simulator  (phy/rate-control/, run separately)
  * Student  -> PKD log-AR process    (this script)

Fidelity hierarchy: effective-SINR -> PER -> closed-loop rate adaptation.

PER LUT note: the teacher's controller (in MATLAB) drives its decisions from
the WLAN Toolbox L2SM AWGN PER table; the student's controller here uses
pkd.per_lut.AWGNPERLookup (the embedded LDPC table, packet_length=1000). This
is the same split already used for the Fig. 14 PER-SNR waterfall
(pkd/baselines/evaluate_per_waterfall.py). The two tables are close but not
identical, so a small teacher/student gap is expected even for matched
effective SINR.

Inputs (produced by phy/rate-control/main_rate_control.m):
  phy/rate-control/snr_trajectory.mat
  phy/rate-control/teacher_rate_control.mat

Output (4 separate subfigure PNGs, assembled by LaTeX -- see
self_review/5Experiment.tex):
  figures/rate_control_snr_trajectory.png    -- (a) common SNR trajectory
  figures/rate_control_mcs_prob_teacher.png  -- (b) teacher MCS-selection prob.
  figures/rate_control_mcs_prob_pkd.png      -- (c) PKD MCS-selection prob.
  figures/rate_control_goodput_cdf.png       -- (d) achieved-goodput CDF

Usage:
  python -m pkd.example.evaluate_rate_control \
      [--checkpoint pkd/trained_models/exclude_config_0/pkd_model.pt] \
      [--n-runs 100] [--burn-in 50] [--seed 42] [--out-dir figures]
"""
import argparse
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch

from pkd.model import PKDModel
from pkd.per_lut import AWGNPERLookup
from pkd.infer import PKDInference
from pkd.rate_control import RateController, MCS_MIN, MCS_MAX

# ----------------------------------------------------------------------------
# Fixed static slice for the experiment: Model-B, 3x2:2, CBW40
# (same slice as the Fig. 14 PER-SNR waterfall). Must match the MATLAB teacher.
# ----------------------------------------------------------------------------
SLICE_SPEC = {
    'channel_model_id': 2,   # Model-B
    'N_t': 3,
    'N_r': 2,
    'BW': 40.0,
    'N_ss': 2,
    'R_t': 0,                # full-band allocation
}
PACKET_LENGTH = 1000         # APEPLength = 1000 bytes, matches the teacher
MCS_INIT = 4                 # controller start, matches the teacher

TEACHER_MAT = 'phy/rate-control/teacher_rate_control.mat'
TRAJ_MAT = 'phy/rate-control/snr_trajectory.mat'


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt['model_config']
    cfg.setdefault('num_R', 8)
    cfg.setdefault('kappa_max', 0.95)
    cfg.setdefault('innovation_type', 'gaussian')
    cfg.setdefault('min_sigma', 0.1)
    model = PKDModel(**cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def run_pkd_closed_loop(
    model,
    per_lut,
    snr_traj: np.ndarray,
    n_runs: int,
    ar_order: int,
    device: str,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Run the PKD closed rate-adaptation loop n_runs times over the common
    SNR trajectory.

    Each run:
      cold-start PKDInference on the slice at SNR_traj[0], MCS_INIT, then for
      each packet t:
        gamma_eff_t <- PKD          (config = slice + SNR_traj[t] + MCS_t)
        per_t       <- AWGN LUT(gamma_eff_t, MCS_t, packet_length=1000)
        error_t     <- Bernoulli(per_t)
        MCS_{t+1}   <- RateController.update(per_t, MCS_t)

    Returns dict with mcs (n_runs x T), errors (n_runs x T), gamma_eff, per.
    """
    T = len(snr_traj)
    rng = np.random.default_rng(seed)

    mcs_all = np.zeros((n_runs, T), dtype=np.int64)
    err_all = np.zeros((n_runs, T), dtype=np.int64)
    gamma_all = np.zeros((n_runs, T), dtype=np.float64)
    per_all = np.zeros((n_runs, T), dtype=np.float64)

    inference = PKDInference(model, per_lut, ar_order=ar_order, device=device)

    for n in range(n_runs):
        # PKDInference.step samples the innovation and the error event from the
        # global numpy RNG; reseed per run for reproducibility.
        np.random.seed(int(rng.integers(0, 2**31 - 1)))

        ctrl = RateController()
        mcs_cur = MCS_INIT

        def cfg_at(t, mcs):
            c = dict(SLICE_SPEC)
            c.update({
                'SNR_bar': float(snr_traj[t]),
                'MCS': int(mcs),
                'packet_length': PACKET_LENGTH,
            })
            return c

        inference.cold_start(cfg_at(0, mcs_cur))

        for t in range(T):
            gamma_eff, per, error = inference.step(cfg_at(t, mcs_cur), return_per=True)
            mcs_all[n, t] = mcs_cur
            err_all[n, t] = int(error)
            gamma_all[n, t] = float(gamma_eff)
            per_all[n, t] = float(per)
            mcs_cur = ctrl.update(per, mcs_cur)

    return {'mcs': mcs_all, 'errors': err_all, 'gamma_eff': gamma_all, 'per': per_all}


def mcs_selection_prob(mcs_all: np.ndarray, smooth: int = 21) -> np.ndarray:
    """P(MCS_t = m) over runs, shape (num_mcs, T).

    `smooth` applies a centered moving-average of that many packets along t.
    The SNR trajectory is quasi-static over ~20 packets, so this only removes
    the Monte-Carlo / controller-dither noise in the per-packet estimate from
    N=100 runs; it does not distort the MCS-vs-time structure.
    """
    n_runs, T = mcs_all.shape
    num_mcs = MCS_MAX - MCS_MIN + 1
    prob = np.zeros((num_mcs, T))
    for m in range(num_mcs):
        prob[m] = (mcs_all == m).mean(axis=0)
    if smooth and smooth > 1:
        k = np.ones(smooth) / smooth
        prob = np.array([np.convolve(row, k, mode='same') for row in prob])
        # renormalize columns (convolve 'same' shrinks edges slightly)
        col_sum = prob.sum(axis=0, keepdims=True)
        prob = np.divide(prob, col_sum, out=np.zeros_like(prob), where=col_sum > 0)
    return prob


def windowed_goodput(
    mcs_all: np.ndarray,
    err_all: np.ndarray,
    packet_dur_by_mcs: np.ndarray,
    payload_bits: int,
    seg_len: int,
    burn_in: int,
) -> np.ndarray:
    """Time-resolved achieved goodput: one sample per (run, non-overlapping
    seg_len-packet window) with window start >= burn_in. Mbps.

    Matches phy/rate-control/corrPHYRateControl.m goodputSamplesMbps.
    """
    n_runs, T = mcs_all.shape
    seg_starts = np.arange(burn_in, T - seg_len + 1, seg_len)
    samples = []
    airtime = packet_dur_by_mcs[np.clip(mcs_all, 0, len(packet_dur_by_mcs) - 1)]
    success_bits = (err_all == 0).astype(float) * payload_bits
    for s in seg_starts:
        idx = slice(s, s + seg_len)
        gp = success_bits[:, idx].sum(axis=1) / airtime[:, idx].sum(axis=1) / 1e6
        samples.append(gp)
    return np.concatenate(samples)


def ecdf(x: np.ndarray):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


# Four separate subfigure files, assembled by LaTeX \subfigure (see
# self_review/5Experiment.tex), matching the Fig. 14 style: no in-figure
# titles (LaTeX captions carry them), Title-Case axes.
SUBFIG_NAMES = {
    'snr': 'rate_control_snr_trajectory.png',
    'teacher': 'rate_control_mcs_prob_teacher.png',
    'pkd': 'rate_control_mcs_prob_pkd.png',
    'cdf': 'rate_control_goodput_cdf.png',
}
TEACHER_COLOR = '#1f77b4'
PKD_COLOR = '#d62728'

# LaTeX layout (self_review/5Experiment.tex), 2x2 at 0.49\columnwidth each:
#     row 1:  (a) SNR trajectory   |  (d) goodput CDF      -- wide, short
#     row 2:  (b) teacher heatmap  |  (c) PKD heatmap       -- rectangular
# All four share the SAME WIDTH in inches so \includegraphics[width=\linewidth]
# renders them aligned; the line plots are shorter than the heatmaps.
# bbox_inches='tight' is deliberately NOT used (it re-crops each figure and
# breaks the shared width); constrained_layout keeps labels/colorbar inside.
FIG_W = 6.4                       # inches -- identical for every subfigure
FIG_H_LINE = 3.1                  # (a) trajectory, (d) CDF
FIG_H_HEAT = 4.4                  # (b), (c) heatmaps
SAVE_KW = dict(dpi=300)

RC = {
    'font.family': 'DejaVu Sans',
    'font.size': 13,
    'axes.titlesize': 13,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
}


def _new_ax(height):
    fig, ax = plt.subplots(figsize=(FIG_W, height), constrained_layout=True)
    return fig, ax


def _mcs_heatmap(prob, vmax, T, save_path):
    num_mcs = prob.shape[0]
    fig, ax = _new_ax(FIG_H_HEAT)
    im = ax.imshow(
        prob, aspect='auto', origin='lower',
        extent=[0, T, MCS_MIN - 0.5, MCS_MAX + 0.5],
        cmap='turbo', vmin=0, vmax=vmax, interpolation='bilinear',
    )
    ax.set_xlabel('Packet Number')
    ax.set_ylabel('MCS Index')
    ax.set_yticks(range(num_mcs))
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label('Selection Probability')
    fig.savefig(save_path, **SAVE_KW)
    plt.close(fig)
    print(f'Saved {save_path}')


def make_figure(
    snr_traj: np.ndarray,
    teacher: Dict[str, np.ndarray],
    student: Dict[str, np.ndarray],
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    T = len(snr_traj)
    pkt = np.arange(1, T + 1)

    p_teacher = mcs_selection_prob(teacher['mcs'])
    p_student = mcs_selection_prob(student['mcs'])
    # shared color scale, capped near the actual peak so bands stay readable
    vmax = float(np.ceil(max(p_teacher.max(), p_student.max()) * 10) / 10)
    vmax = min(max(vmax, 0.3), 0.8)

    gp_t = teacher['goodput_samples']
    gp_s = student['goodput_samples']

    with plt.rc_context(RC):
        # --- (a) common SNR trajectory ---
        fig, ax = _new_ax(FIG_H_LINE)
        ax.plot(pkt, snr_traj, color=TEACHER_COLOR, lw=1.3)
        ax.set_xlabel('Packet Number')
        ax.set_ylabel('SNR (dB)')
        ax.set_xlim(0, T)
        fig.savefig(os.path.join(out_dir, SUBFIG_NAMES['snr']), **SAVE_KW)
        plt.close(fig)
        print(f"Saved {os.path.join(out_dir, SUBFIG_NAMES['snr'])}")

        # --- (b) teacher / (c) PKD MCS-selection probability heatmaps ---
        _mcs_heatmap(p_teacher, vmax, T, os.path.join(out_dir, SUBFIG_NAMES['teacher']))
        _mcs_heatmap(p_student, vmax, T, os.path.join(out_dir, SUBFIG_NAMES['pkd']))

        # --- (d) achieved-goodput CDF ---
        xt, yt = ecdf(gp_t)
        xs, ys = ecdf(gp_s)
        fig, ax = _new_ax(FIG_H_LINE)
        ax.plot(xt, yt, color=TEACHER_COLOR, lw=2, label='Teacher (PHY Simulator)')
        ax.plot(xs, ys, color=PKD_COLOR, lw=2, ls='--', label='PKD (Student)')
        ax.set_xlabel('Achieved Goodput (Mbps)')
        ax.set_ylabel('CDF')
        ax.set_ylim(0, 1)
        ax.legend(framealpha=0.9, loc='upper left', fontsize=11)

        mt, st = gp_t.mean(), gp_t.std()
        ms, ss = gp_s.mean(), gp_s.std()
        rel = 100 * abs(mt - ms) / mt
        txt = (f'Mean goodput (Mbps)\n'
               f'Teacher: {mt:.1f} ± {st:.1f}\n'
               f'PKD:     {ms:.1f} ± {ss:.1f}\n'
               f'Rel. diff.: {rel:.1f}%')
        ax.text(0.97, 0.05, txt, transform=ax.transAxes, fontsize=10,
                va='bottom', ha='right', family='monospace',
                bbox=dict(boxstyle='round', fc='white', ec='0.7'))
        fig.savefig(os.path.join(out_dir, SUBFIG_NAMES['cdf']), **SAVE_KW)
        plt.close(fig)
        print(f"Saved {os.path.join(out_dir, SUBFIG_NAMES['cdf'])}")


def print_diagnostics(name: str, res: Dict[str, np.ndarray], snr_traj: np.ndarray):
    mcs = res['mcs']; err = res['errors']
    mt = mcs.mean(axis=0)
    print(f'  [{name}] mean MCS {mcs.mean():.2f}  overall PER {err.mean():.4f}  '
          f'corr(meanMCS_t, SNR_t) {np.corrcoef(mt, snr_traj)[0, 1]:.3f}  '
          f'median across-run MCS std {np.median(mcs.std(axis=0)):.2f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', default='pkd/trained_models/exclude_config_0/pkd_model.pt')
    ap.add_argument('--n-runs', type=int, default=None,
                    help='PKD realizations (default: match the teacher N)')
    ap.add_argument('--burn-in', type=int, default=None,
                    help='leading packets excluded from panels b/c (default: from teacher meta)')
    ap.add_argument('--seg-len', type=int, default=None,
                    help='goodput window (default: from teacher meta)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ar-order', type=int, default=10)
    ap.add_argument('--out-dir', default='figures',
                    help='directory for the 4 subfigure PNGs')
    ap.add_argument('--device', default=None)
    ap.add_argument('--student-cache', default='figures/rate_control_pkd_student.npz',
                    help='reuse a saved PKD closed-loop result if present (delete to force a re-run)')
    args = ap.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- teacher ----
    td = sio.loadmat(TEACHER_MAT, simplify_cells=True)
    meta = td['meta']
    snr_traj = np.asarray(td['snrTraj'], dtype=float).ravel()
    teacher_mcs = np.asarray(td['mcsAll'], dtype=np.int64)
    teacher_err = np.asarray(td['errorAll'], dtype=np.int64)
    packet_dur_by_mcs = np.asarray(meta['packetDurationByMCS'], dtype=float).ravel()
    payload_bits = int(td['payloadBits'])
    seg_len = args.seg_len or int(meta['segLen'])
    burn_in = args.burn_in if args.burn_in is not None else int(meta['burnIn'])
    n_runs = args.n_runs or teacher_mcs.shape[0]

    print(f'Teacher: {teacher_mcs.shape[0]} runs x {teacher_mcs.shape[1]} packets, '
          f'SNR {snr_traj.min():.0f}-{snr_traj.max():.0f} dB, seg_len {seg_len}, burn_in {burn_in}')

    # sanity: trajectory file matches the teacher output
    trj = sio.loadmat(TRAJ_MAT, simplify_cells=True)
    if not np.allclose(np.asarray(trj['snrTraj'], dtype=float).ravel(), snr_traj):
        raise RuntimeError('snr_trajectory.mat does not match teacher_rate_control.mat snrTraj')

    teacher = {
        'mcs': teacher_mcs,
        'errors': teacher_err,
        'goodput_samples': np.asarray(td['goodputSamplesMbps'], dtype=float).ravel(),
    }

    # ---- student ----
    print(f'PKD: loading {args.checkpoint} on {device}')
    model = load_model(args.checkpoint, device)
    per_lut = AWGNPERLookup.load_ldpc_lut()

    cache = args.student_cache
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if z['mcs'].shape == (n_runs, len(snr_traj)) and int(z['seed']) == args.seed:
            print(f'PKD: reusing cached closed-loop result {cache}')
            student = {k: z[k] for k in ('mcs', 'errors', 'gamma_eff', 'per')}
        else:
            student = None
    else:
        student = None

    if student is None:
        print(f'PKD: running {n_runs} closed-loop realizations...')
        student = run_pkd_closed_loop(
            model, per_lut, snr_traj, n_runs=n_runs, ar_order=args.ar_order,
            device=device, seed=args.seed,
        )
        if cache:
            os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
            np.savez(cache, seed=args.seed, **student)
            print(f'PKD: cached closed-loop result -> {cache}')
    student['goodput_samples'] = windowed_goodput(
        student['mcs'], student['errors'], packet_dur_by_mcs, payload_bits, seg_len, burn_in,
    )

    print('\nDiagnostics:')
    print_diagnostics('Teacher', {'mcs': teacher_mcs, 'errors': teacher_err}, snr_traj)
    print_diagnostics('PKD    ', student, snr_traj)

    # trim burn-in from panels (b)/(c) for both
    def trim(res):
        return {k: (v[:, burn_in:] if k in ('mcs', 'errors') else v) for k, v in res.items()}

    make_figure(snr_traj[burn_in:], trim(teacher), trim(student), args.out_dir)


if __name__ == '__main__':
    main()
