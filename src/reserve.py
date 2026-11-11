"""Certified closure reserves (plan v2.1, Section 6). Split-conformal quantiles per Mondrian cell (kind, h, q)
with pooling over q before a worst-case fallback, a deterministic worst-case bound, bootstrap intervals over
calibration tasks, and the coverage check."""
from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd

import config
from closure import CLOSURE_PRESETS, CLOSURE_SYSTEM, KINDS

CELLS_H = (0, 1, 2)
CELLS_Q = (0, 1)          # q is pooled as {0, 1+}


def q_bucket(q: int) -> int:
    return 0 if q == 0 else 1


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
    if kind in ("draft", "stop"):
        return 0
    sys_tokens = counter.count(CLOSURE_SYSTEM.format(M_c=p["M_c"]))
    total = 0
    for j in range(1, p["M_c"] + 1):
        prompt = sys_tokens + sigma_max_tokens + (j - 1) * (p["m_out_c"] + p["L_c"])
        total += prompt + p["m_out_c"] + p["w"]
    return total + h * p["w"]


def sigma_max_tokens(counter, presets: dict, suite) -> int:
    """Token size of the largest snapshot the schema allows: the longest task prompt, H_max pending holds,
    Q_max queued calls with long keys, twelve done calls and twelve references."""
    import closure as cl
    longest = max(t.PROMPT for t in suite.user_tasks.values())
    hold = {"hold_id": "H99", "item_type": "hotel", "item_id": "HOT-XXX-9", "start": "2027-12-31", "end": "2027-12-31",
            "amount": 99999.99}
    queued = [{"function": "confirm_booking", "args": {"hold_id": "H99", "key": "H99-confirm-" + "x" * 32}},
              {"function": "send_money", "args": {"payee_id": "P99", "amount": 99999.99, "key": "INV-2027-999-" + "x" * 32,
                                                  "payee_name": "City Council Tax Department"}},
              {"function": "reply_email", "args": {"msg_id": "MSG-TRIP-99", "text": "x" * 200}}]
    done = [{"function": "confirm_booking", "args": {"hold_id": f"H{i}", "key": f"H{i}-confirm"}, "verified": True}
            for i in range(12)]
    sigma = {"task": longest, "pending": [hold] * (presets["H_max"] + 1), "queued": queued[: presets["Q_max"]],
             "done": done, "trusted_refs": [f"HOT-XXX-{i}" for i in range(12)]}
    return counter.count(cl.snapshot_text(sigma))


def reserves_from_table(table: pd.DataFrame, deltas, presets: dict, counter, sigma_max_tokens: int) -> dict:
    """table columns: kind, h, q, cost (calibration closures). Returns {delta: {kind: {(h,qb): {...}}}} with the
    reserve R, the sample size, whether q was pooled, whether the worst case was used, and the bound."""
    out = {}
    tab = table.copy()
    tab["qb"] = tab["q"].map(q_bucket)
    for delta in deltas:
        d = {}
        for kind in KINDS:
            d[kind] = {}
            for h in CELLS_H:
                for qb in CELLS_Q:
                    bound = worst_case_bound(kind, h, qb, presets, counter, sigma_max_tokens)
                    entry = {"R": bound, "n": 0, "pooled_q": False, "worst_case": True, "bound": bound}
                    if kind in ("settle", "draft", "stop"):
                        entry.update({"worst_case": False, "exact": True})     # deterministic classes
                    else:
                        cell = tab[(tab.kind == kind) & (tab.h == h) & (tab.qb == qb)]["cost"]
                        R = conformal_quantile(cell, delta)
                        if R is None:
                            pooled = tab[(tab.kind == kind) & (tab.h == h)]["cost"]
                            R = conformal_quantile(pooled, delta)
                            if R is not None:
                                entry.update({"R": min(R, bound), "n": int(len(pooled)), "pooled_q": True, "worst_case": False})
                            else:
                                entry.update({"n": int(len(cell))})
                        else:
                            entry.update({"R": min(R, bound), "n": int(len(cell)), "worst_case": False})
                    d[kind][f"{h},{qb}"] = entry
        out[str(delta)] = d
    return out


def lookup(reserves: dict, delta: float, kind: str, h: int, q: int) -> float:
    h = min(h, max(CELLS_H))
    return float(reserves[str(delta)][kind][f"{h},{q_bucket(q)}"]["R"])


def bootstrap_reserves(table: pd.DataFrame, delta: float, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """Resample calibration tasks with replacement; interval on the verified-closure reserve per cell."""
    rng = random.Random(seed)
    tasks = sorted(table.task_id.unique())
    rows = []
    tab = table[table.kind == "verified"].copy()
    tab["qb"] = tab["q"].map(q_bucket)
    for h in CELLS_H:
        for qb in CELLS_Q:
            vals = []
            for _ in range(n_boot):
                sample = [rng.choice(tasks) for _ in tasks]
                sub = pd.concat([tab[(tab.task_id == t) & (tab.h == h) & (tab.qb == qb)] for t in sample])
                R = conformal_quantile(sub["cost"], delta) if len(sub) else None
                if R is None:
                    pooled = pd.concat([tab[(tab.task_id == t) & (tab.h == h)] for t in sample])
                    R = conformal_quantile(pooled["cost"], delta) if len(pooled) else None
                if R is not None:
                    vals.append(R)
            if vals:
                rows.append({"h": h, "qb": qb, "n_boot": len(vals), "R_median": float(np.median(vals)),
                             "R_p2.5": float(np.percentile(vals, 2.5)), "R_p97.5": float(np.percentile(vals, 97.5))})
    return pd.DataFrame(rows)


def coverage(table: pd.DataFrame, reserves: dict, delta: float) -> pd.DataFrame:
    """Share of held-out closures whose cost is at or below the reserve of their cell, per condition and kind."""
    from metrics import wilson
    rows = []
    for (cond, kind), g in table.groupby(["condition", "kind"]):
        covered = sum(1 for _, r in g.iterrows() if r["cost"] <= lookup(reserves, delta, kind, int(r["h"]), int(r["q"])))
        p, lo, hi = wilson(covered, len(g))
        rows.append({"condition": cond, "kind": kind, "delta": delta, "n": len(g), "covered": covered,
                     "coverage": round(p, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                     "target": 1 - delta, "holds": bool(hi >= 1 - delta)})
    return pd.DataFrame(rows)
