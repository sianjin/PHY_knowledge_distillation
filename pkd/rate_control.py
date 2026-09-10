"""Shared EWMA-PER dual-threshold rate-control policy (Fig. 15).

Byte-for-byte twin of phy/rate-control/rateController.m. Used identically by
the MATLAB PHY teacher and the PKD student closed loops; the ONLY difference
between the two systems is the source of the effective SINR that produces the
per-packet PER fed to ``RateController.update``.

Policy
------
    per_ewma_t = (1 - ALPHA) * per_ewma_{t-1} + ALPHA * per_inst_t
    if   per_ewma_t > PER_HIGH  -> bad_count++,  good_count = 0
                                   if bad_count  >= DOWN_COUNT: MCS -= 1, reset
    elif per_ewma_t < PER_LOW   -> good_count++, bad_count = 0
                                   if good_count >= UP_COUNT:   MCS += 1, reset
    else                        -> hold, reset both counters

Keep the constants below in sync with rateController.m.
"""
from dataclasses import dataclass

ALPHA = 0.1
PER_LOW = 0.03
PER_HIGH = 0.10
UP_COUNT = 1
DOWN_COUNT = 2
MCS_MIN = 0
MCS_MAX = 9


@dataclass
class RateController:
    """Stateful EWMA-PER dual-threshold rate controller.

    One instance per closed-loop realization. Call ``update`` once per packet
    with the instantaneous estimated PER for the packet just sent and the MCS
    it used; it returns the MCS to use for the next packet.
    """

    per_ewma: float = None
    good_count: int = 0
    bad_count: int = 0

    def update(self, per_inst: float, mcs_cur: int) -> int:
        if self.per_ewma is None:
            self.per_ewma = float(per_inst)
        else:
            self.per_ewma = (1.0 - ALPHA) * self.per_ewma + ALPHA * float(per_inst)

        mcs_next = mcs_cur

        if self.per_ewma > PER_HIGH:
            self.good_count = 0
            self.bad_count += 1
            if self.bad_count >= DOWN_COUNT:
                mcs_next = max(mcs_cur - 1, MCS_MIN)
                self.bad_count = 0
        elif self.per_ewma < PER_LOW:
            self.bad_count = 0
            self.good_count += 1
            if self.good_count >= UP_COUNT:
                mcs_next = min(mcs_cur + 1, MCS_MAX)
                self.good_count = 0
        else:
            self.good_count = 0
            self.bad_count = 0

        return mcs_next
