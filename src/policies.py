"""Budget policies compared at the same budget B (plan v2.1, Section 7). Every policy exposes
allow(remaining, cap, state) -> bool, called at the two points of the exact gate: before a model call (cap is the
exact prompt plus m_out; state["h"] is the holds open now) and before the proposed calls are executed (cap is w per
call; state["h"] is the holds they would leave open). state also carries the milestones, the spend so far and B.
Each policy also has a chooser(kinds, h, q, remaining, table_t) that picks the closure kind run at the cut."""
from __future__ import annotations

import closure as cl
import reserve as rs


class Policy:
    name = "base"

    def bind(self, rec: dict) -> None:
        self.rec = rec

    def allow(self, remaining, cap, state) -> bool:
        raise NotImplementedError

    @staticmethod
    def attempt_useful(kinds, h, q, remaining, table_t) -> str:
        """Baselines without a reserve notion run the useful closure when there is anything to do."""
        return "verified" if (h > 0 or q > 0) else "stop"

    chooser = attempt_useful


class FixedCap(Policy):
    """B0: continue while any budget is left; the model call that overspends is the cut and its proposed calls are
    not executed (DoW fixed-cap semantics: subsequent calls stop, the triggering turn is not capped)."""
    name = "B0_fixed_cap"
    hard_cap = True

    def allow(self, remaining, cap, state):
        return remaining > 0


class FixedFraction(Policy):
    """B1: reserve a fixed share of B (Agent Contracts' B_reserve, practitioner guides)."""

    def __init__(self, rho: float):
        self.rho = rho
        self.name = f"B1_fixed_{int(round(100 * rho))}pct"

    def allow(self, remaining, cap, state):
        return remaining - cap >= self.rho * state["B"]


class SelfEstimate(Policy):
    """B2: the model's own estimate (BAGEN probe) of the budget still needed; estimates[(task, t)] = (lo, hi, impossible)."""
    name = "B2_self_estimate"

    def __init__(self, estimates: dict):
        self.estimates = estimates

    def allow(self, remaining, cap, state):
        lo, hi, impossible = self.estimates.get((self.rec["task_id"], self.rec.get("condition", self.rec["attack"]), state["t"]),
                                                (0.0, 0.0, False))
        if impossible:
            return False
        return remaining - cap >= hi


class ProgressAuthorized(Policy):
    """B3: Zhang et al. progress-authorized cap re-implemented (Algorithm 3): base ceiling B0, raised to 1.3 B0 at
    70% of the host-verified milestones and to 1.6 B0 at all of them; stop when spend reaches the ceiling."""

    def __init__(self, variant: str = "same_ceiling", b0_multiplier: float | None = None):
        self.variant = variant
        self.mult = b0_multiplier
        self.name = f"B3_progress_{variant}"

    def allow(self, remaining, cap, state):
        B = state["B"]
        b0 = B / 1.6 if self.variant == "same_ceiling" else B * self.mult
        frac = state["milestones"] / max(1, state["gt_len"])
        ceiling = b0 * (1.6 if frac >= 1.0 else 1.3 if frac >= 0.7 else 1.0)
        return state["spent"] < ceiling


# B4, the oracle, is not a gate: it is computed by replay.oracle_best as the best reachable outcome of any cut.


class WorstCase(Policy):
    """B5: the deterministic worst-case reserve (equation 4); delta = 0 without calibration."""
    name = "B5_worst_case"

    def __init__(self, bounds: dict):
        self.bounds = bounds    # reserves-shaped dict with the bound in "R"

    def _R(self, kind, h, q):
        return rs.lookup(self.bounds, 0.0, kind, h, q)

    def allow(self, remaining, cap, state):
        # keep enough for the most useful closure of the state this step leads to: the verified closure when a
        # hold is open (its reserve is the certified quantile, or the bound for B5), nothing when nothing is open
        return remaining - cap >= self._R("verified", state["h"], 0)

    def chooser(self, kinds, h, q, remaining, table_t):
        for k in ("verified", "settle", "draft", "stop"):
            if k in table_t and self._R(k, h, q) <= remaining:
                if k in ("draft", "stop") and h > 0:
                    return "settle" if self._R("settle", h, 0) <= remaining else "stop"
                return k
        return "stop"


class ReserveGuard(WorstCase):
    """Ours: per-cell conformal reserves at level delta (equations 5 to 8)."""

    def __init__(self, reserves: dict, delta: float):
        self.reserves, self.delta = reserves, delta
        self.name = f"ReserveGuard_d{int(round(100 * delta))}"

    def _R(self, kind, h, q):
        return rs.lookup(self.reserves, self.delta, kind, h, q)
