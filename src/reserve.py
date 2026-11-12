"""Certified closure reserves (plan v2.1, Section 6, revised). Split-conformal quantiles per Mondrian cell
(kind, h, c) over task-level scores, pooling over c, then over h >= 1, before a worst-case fallback; a
deterministic worst-case bound, bootstrap intervals over calibration tasks, and the coverage checks.

Revision 3 (after the second calibration): the exchangeable unit is the task, not the closure. The closure cost of
a task barely changes between cuts and conditions (the snapshot at a given h is the same and decoding is
deterministic), so the 132 closures of the h = 1 cell were 12 tasks repeated, and the certificate failed on
held-out tasks (coverage 63 to 79 percent at delta 0.05). The score of a task in a cell is now the largest closure
cost over every cut of its trajectories under every condition; the reserve is the conformal quantile of the task
scores. The guarantee then reads: for a new task, with probability at least 1 - delta over tasks, every verified
closure at any cut of any recorded condition costs at most R. A cell with fewer than ceil(1/delta) - 1 tasks is
pooled over h >= 1 (the pooled score is the task's largest closure cost over all h >= 1, which bounds every member
cell), and a pooled cell that is still too small takes the worst-case bound."""
from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd

import config
from closure import CLOSURE_PRESETS, CLOSURE_SYSTEM, KINDS, SETTLED_MAX

CELLS_H = (0, 1, 2)
CELLS_Q = (0, 1)          # the second cell coordinate: confirmed bookings so far, bucketed 0 / 1+


def q_bucket(c: int) -> int:
    return 0 if c == 0 else 1


def n_required(delta: float) -> int:
    return math.ceil(1.0 / delta) - 1


def conformal_quantile(costs, delta: float) -> float | None:
    """C_(k) with k = ceil((n+1)(1-delta)); None when n is too small for the level."""
    if delta <= 0:
        return None
    c = sorted(float(x) for x in costs)
    n = len(c)
    if n < n_required(delta):
        return None
    k = math.ceil((n + 1) * (1 - delta))
    return c[min(k, n) - 1]


def worst_case_bound(kind: str, h: int, q: int, presets: dict, counter, sigma_max_tokens: int) -> int:
    """Deterministic bound on closure cost from the schema and the presets (equation 4).
    verified: M_c model calls whose prompts grow by one return and one output per step, plus one tool call per
    step, plus the host fallback settle. settle: w per open hold. draft, stop: 0."""
    p = dict(presets); p.update(CLOSURE_PRESETS)
    if kind == "settle":
        return h * p["w"]
    if kind in ("draft", "stop") or (kind == "verified" and h == 0):
        return 0
    sys_tokens = counter.count(CLOSURE_SYSTEM.format(M_c=p["M_c"]))
    total = 0
    for j in range(1, p["M_c"] + 1):
        prompt = sys_tokens + sigma_max_tokens + (j - 1) * (p["m_out_c"] + p["L_c"])
        total += prompt + p["m_out_c"] + p["w"]
    return total + h * p["w"]


def sigma_max_tokens(counter, presets: dict, suite) -> int:
    """Token size of the largest snapshot the schema allows: the longest task prompt, H_max + 1 pending holds,
    SETTLED_MAX settled holds, twelve done calls and twelve references."""
    import closure as cl
    longest = max(t.PROMPT for t in suite.user_tasks.values())
    hold = {"hold_id": "H99", "item_type": "hotel", "item_id": "HOT-XXX-9", "name": "Valencia Riverside Inn Garden Lodge",
            "unit_price": 999.99, "start": "2027-12-31", "end": "2027-12-31", "amount": 99999.99, "status": "HELD"}
    settled = dict(hold); settled["status"] = "CONFIRMED"
    done = [{"function": "confirm_booking", "args": {"hold_id": f"H{i}", "key": f"H{i}-confirm-" + "x" * 16}, "verified": True}
            for i in range(12)]
    sigma = {"task": longest, "pending": [hold] * (presets["H_max"] + 1), "settled": [settled] * SETTLED_MAX,
             "done": done, "trusted_refs": [f"HOT-XXX-{i}" for i in range(12)]}
    return counter.count(cl.snapshot_text(sigma))


# ----------------------------------------------------------------------------- task-level scores
def _verified(table: pd.DataFrame) -> pd.DataFrame:
    v = table[(table.kind == "verified") & (table.h > 0)].copy()
    if "cut_off" in v.columns:
        v = v[~v.cut_off.astype(bool)]
    v["cb"] = v["confirmed_before"].map(q_bucket) if "confirmed_before" in v.columns else 0
    return v


def task_scores(table: pd.DataFrame, h: int | None, cb: int | None = None) -> pd.Series:
    """One score per task: its largest verified-closure cost in cell (h, cb). cb = None pools over c (the h cell);
    h = None pools every h >= 1."""
    v = _verified(table)
    if h is not None:
        v = v[v.h == h]
    if cb is not None:
        v = v[v.cb == cb]
    if not len(v):
        return pd.Series(dtype=float)
    return v.groupby("task_id")["cost"].max().astype(float)


def cell_reserve(table: pd.DataFrame, h: int, cb: int, delta: float) -> tuple[float | None, int, str]:
    """The conformal quantile for cell (h, cb) with the pooling chain: own cell, then the h cell, then h >= 1.
    Returns (R or None, number of tasks used, pooling level)."""
    for level, scores in (("cell", task_scores(table, h, cb)), ("h", task_scores(table, h, None)), ("h>=1", task_scores(table, None))):
        R = conformal_quantile(scores, delta)
        if R is not None:
            return R, int(len(scores)), level
    return None, int(len(task_scores(table, h, cb))), "none"


def reserves_from_table(table: pd.DataFrame, deltas, presets: dict, counter, sigma_max_tokens: int) -> dict:
    """table: closure rows of the calibration tasks (kind, h, q, cost, task_id, cut_off). Returns
    {delta: {kind: {"h,qb": {R, n (tasks), unit, pooled_h, worst_case, bound, exact}}}}."""
    out = {}
    for delta in deltas:
        d = {}
        for kind in KINDS:
            d[kind] = {}
            for h in CELLS_H:
                for cb in CELLS_Q:
                    bound = worst_case_bound(kind, h, cb, presets, counter, sigma_max_tokens)
                    entry = {"R": bound, "n": 0, "unit": "task", "pooled": "none", "worst_case": True, "bound": bound,
                             "n_cell": int(len(task_scores(table, h, cb))) if kind == "verified" and h > 0 else 0}
                    if kind in ("settle", "draft", "stop") or (kind == "verified" and h == 0):
                        entry.update({"worst_case": False, "exact": True})     # deterministic classes
                    else:
                        R, n, level = cell_reserve(table, h, cb, delta)
                        if R is not None:
                            entry.update({"R": min(R, bound), "n": n, "pooled": level, "worst_case": False})
                    d[kind][f"{h},{cb}"] = entry
        out[str(delta)] = d
    return out


def lookup(reserves: dict, delta: float, kind: str, h: int, c: int) -> float:
    h = min(h, max(CELLS_H))
    return float(reserves[str(delta)][kind][f"{h},{q_bucket(c)}"]["R"])


def bootstrap_reserves(table: pd.DataFrame, delta: float, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """Resample calibration tasks with replacement; interval on the verified-closure reserve per cell, with the
    same pooling rule as reserves_from_table (the bound is not applied here)."""
    rng = random.Random(seed)
    v = _verified(table)
    tasks = sorted(v.task_id.unique())
    per_task = {t: v[v.task_id == t] for t in tasks}
    rows = []
    for h in [x for x in CELLS_H if x > 0]:
        for cb in CELLS_Q:
            if not len(task_scores(v, h, cb)):
                continue
            vals, pooled_used = [], 0
            for _ in range(n_boot):
                sample = [rng.choice(tasks) for _ in tasks]
                sub = pd.concat([per_task[t].assign(task_id=f"{t}#{i}") for i, t in enumerate(sample)])
                R, _, level = cell_reserve(sub, h, cb, delta)
                if R is not None:
                    vals.append(R)
                    pooled_used += level != "cell"
            if vals:
                rows.append({"h": h, "c": cb, "n_boot": len(vals), "pooled_share": round(pooled_used / len(vals), 3),
                             "R_median": float(np.median(vals)), "R_p2.5": float(np.percentile(vals, 2.5)),
                             "R_p97.5": float(np.percentile(vals, 97.5))})
    return pd.DataFrame(rows)


def coverage(table: pd.DataFrame, reserves: dict, delta: float) -> pd.DataFrame:
    """Closure-level check: share of held-out closures whose cost is at or below the reserve of their cell, per
    condition and kind. Closures of one task are not independent, so this is descriptive; the certified statement
    is checked by task_coverage."""
    from metrics import wilson
    rows = []
    table = table[~((table.kind == "verified") & (table.h == 0))]      # the h = 0 verified closure is exact, not certified
    for (cond, kind), g in table.groupby(["condition", "kind"]):
        covered = sum(1 for _, r in g.iterrows()
                      if r["cost"] <= lookup(reserves, delta, kind, int(r["h"]), int(r.get("confirmed_before", 0))))
        p, lo, hi = wilson(covered, len(g))
        rows.append({"condition": cond, "kind": kind, "delta": delta, "n": len(g), "covered": covered,
                     "coverage": round(p, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                     "target": 1 - delta, "holds": bool(hi >= 1 - delta)})
    return pd.DataFrame(rows)


def task_coverage(table: pd.DataFrame, reserves: dict, delta: float) -> pd.DataFrame:
    """The certified statement on held-out tasks: per cell (h, c), the share of test tasks whose largest
    verified-closure cost in that cell (over every cut and condition) is at or below the reserve."""
    from metrics import wilson
    rows = []
    for h in [x for x in CELLS_H if x > 0]:
        for cb in CELLS_Q:
            scores = task_scores(table, h, cb)
            if not len(scores):
                continue
            R = lookup(reserves, delta, "verified", h, cb)
            covered = int((scores <= R).sum())
            p, lo, hi = wilson(covered, len(scores))
            rows.append({"h": h, "c": cb, "delta": delta, "R": R, "n_tasks": int(len(scores)), "covered": covered,
                         "coverage": round(p, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "target": 1 - delta,
                         "max_score": float(scores.max()), "holds": bool(hi >= 1 - delta)})
    return pd.DataFrame(rows)
