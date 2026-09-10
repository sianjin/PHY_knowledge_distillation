"""Parity test for the shared rate-control policy (Fig. 15).

pkd/rate_control.py (RateController) must be a byte-for-byte twin of
phy/rate-control/rateController.m: given the SAME per-packet PER sequence and
the SAME starting MCS, the two implementations must produce an identical MCS
trace. The teacher and student closed loops in Fig. 15 differ ONLY in the
source of the effective SINR; this test guards that the controller half is
truly identical.

The Python side is checked two ways:
  1. Against a hand-computed reference for a short crafted PER sequence
     (self-contained -- always runs).
  2. Against a fixture produced by the MATLAB controller, if present:
     phy/rate-control/rate_controller_fixture.mat
     Generate it once with phy/rate-control/genRateControllerFixture.m.

Run:  python -m pytest pkd/tests/test_rate_control_parity.py
  or: python pkd/tests/test_rate_control_parity.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pkd.rate_control import (  # noqa: E402
    RateController, ALPHA, PER_LOW, PER_HIGH, UP_COUNT, DOWN_COUNT,
    MCS_MIN, MCS_MAX,
)

FIXTURE = os.path.join(
    os.path.dirname(__file__), '..', '..', 'phy', 'rate-control',
    'rate_controller_fixture.mat',
)


def run_python_controller(per_seq, mcs_init):
    """Drive RateController over a PER sequence, returning the per-packet MCS
    trace (MCS used for packet t, before the update that picks packet t+1)."""
    ctrl = RateController()
    mcs = mcs_init
    trace = []
    for per in per_seq:
        trace.append(mcs)
        mcs = ctrl.update(float(per), mcs)
    return np.array(trace, dtype=int)


def reference_controller(per_seq, mcs_init):
    """Independent re-implementation of the documented policy, for the
    self-contained check (deliberately not importing rate_control internals
    beyond the constants)."""
    per_ewma = None
    good = bad = 0
    mcs = mcs_init
    trace = []
    for per in per_seq:
        trace.append(mcs)
        per_ewma = float(per) if per_ewma is None else (1 - ALPHA) * per_ewma + ALPHA * float(per)
        if per_ewma > PER_HIGH:
            good = 0
            bad += 1
            if bad >= DOWN_COUNT:
                mcs = max(mcs - 1, MCS_MIN)
                bad = 0
        elif per_ewma < PER_LOW:
            bad = 0
            good += 1
            if good >= UP_COUNT:
                mcs = min(mcs + 1, MCS_MAX)
                good = 0
        else:
            good = bad = 0
    return np.array(trace, dtype=int)


def _crafted_per_sequence():
    """A sequence that exercises every branch: sustained-good climb, hold in
    the dead-band, single bad packet (must NOT step down with DOWN_COUNT=2),
    sustained-bad drop, and the MCS ceiling/floor clamps."""
    rng = np.random.default_rng(0)
    seq = []
    seq += [0.0] * 40            # climb to the ceiling
    seq += [0.06] * 20           # dead-band: hold
    seq += [0.30, 0.0] * 10      # alternating bad/good: EWMA stays mid -> mostly hold
    seq += [0.5] * 40            # sustained bad: drop toward the floor
    seq += [0.0] * 10            # recover a little
    seq += list(rng.uniform(0, 0.2, size=50))  # noisy
    return np.array(seq)


def test_python_matches_reference():
    per_seq = _crafted_per_sequence()
    for mcs_init in (0, 4, 9):
        a = run_python_controller(per_seq, mcs_init)
        b = reference_controller(per_seq, mcs_init)
        assert np.array_equal(a, b), (
            f"RateController disagrees with the reference policy (mcs_init={mcs_init}) "
            f"at packet {np.argmax(a != b)}"
        )


def test_single_bad_packet_does_not_step_down():
    """DOWN_COUNT = 2: one isolated bad EWMA sample must not drop the MCS."""
    # start settled at MCS 5, one spike, then good again
    per_seq = np.array([0.06] * 5 + [0.5] + [0.06] * 5)
    trace = run_python_controller(per_seq, mcs_init=5)
    assert trace.max() == 5 and trace.min() == 5, f"unexpected step: {trace}"


def test_clamps():
    up = run_python_controller(np.zeros(100), mcs_init=7)
    assert up.max() == MCS_MAX
    down = run_python_controller(np.full(100, 0.9), mcs_init=3)
    assert down.min() == MCS_MIN


def test_matches_matlab_fixture():
    """Compare against the MATLAB controller's own output, if the fixture
    has been generated (phy/rate-control/genRateControllerFixture.m)."""
    if not os.path.exists(FIXTURE):
        print(f"SKIP: fixture not found ({FIXTURE}); run genRateControllerFixture.m")
        return
    import scipy.io as sio
    d = sio.loadmat(FIXTURE, simplify_cells=True)

    # sanity: the fixture's constants must match this Python module
    c = d['constants']
    assert abs(c['ALPHA'] - ALPHA) < 1e-12
    assert abs(c['PER_LOW'] - PER_LOW) < 1e-12
    assert abs(c['PER_HIGH'] - PER_HIGH) < 1e-12
    assert int(c['UP_COUNT']) == UP_COUNT
    assert int(c['DOWN_COUNT']) == DOWN_COUNT

    cases = d['cases'] if isinstance(d['cases'], list) else [d['cases']]
    for k, case in enumerate(cases):
        per_seq = np.asarray(case['per'], dtype=float).ravel()
        mcs_init = int(case['mcsInit'])
        matlab_trace = np.asarray(case['mcsTrace'], dtype=int).ravel()
        py_trace = run_python_controller(per_seq, mcs_init)
        assert np.array_equal(py_trace, matlab_trace), (
            f"Python vs MATLAB controller mismatch in fixture case {k} "
            f"(mcs_init={mcs_init}) at packet {np.argmax(py_trace != matlab_trace)}"
        )
    print(f"OK: matched MATLAB fixture ({len(cases)} cases)")


if __name__ == '__main__':
    test_python_matches_reference()
    print("OK test_python_matches_reference")
    test_single_bad_packet_does_not_step_down()
    print("OK test_single_bad_packet_does_not_step_down")
    test_clamps()
    print("OK test_clamps")
    test_matches_matlab_fixture()
