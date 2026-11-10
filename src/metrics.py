"""Shared statistics: Wilson intervals for shares and paired McNemar counts."""
from __future__ import annotations

import math


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    """Point estimate and 95% Wilson interval of a share, as (p, lo, hi) in [0, 1]."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def fmt_share(k: int, n: int) -> str:
    p, lo, hi = wilson(k, n)
    return f"{100 * p:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] ({k}/{n})"


def mcnemar_counts(a: list[bool], b: list[bool]) -> dict:
    """Discordant pair counts between two policies on the same tasks (b wins where b is True and a is False)."""
    assert len(a) == len(b)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    n = a_only + b_only
    # exact two-sided binomial p-value on the discordant pairs
    p = 1.0
    if n > 0:
        k = min(a_only, b_only)
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
        p = min(1.0, 2 * tail)
    return {"b_only": b_only, "a_only": a_only, "discordant": n, "p_exact": p}
