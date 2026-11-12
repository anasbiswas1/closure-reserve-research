"""Figures and tables of the paper, built from the committed reports only (notebook 05). Every figure is saved
as PNG at 600 dpi and as PDF, with the paper's names in legends and axes. Nothing here recomputes a result."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

DPI = 600
CONDITION_NAMES = {"A0": "Clean", "A1": "Padding", "A2": "Fake error", "A3": "Clarification bait", "A123": "Combined",
                   "A5": "Chaining", "A6": "Chaining + padding", "A7": "Compromised endpoint"}
POLICY_NAMES = {"B0_fixed_cap": "Fixed cap", "B1_fixed_10pct": "Fixed reserve (10%)", "B1_fixed_15pct": "Fixed reserve (15%)",
                "B1_fixed_20pct": "Fixed reserve (20%)", "B1_fixed_30pct": "Fixed reserve (30%)", "B2_self_estimate": "Self-estimate",
                "B3_progress_same_ceiling": "Progress-authorized", "B3_progress_same_mean": "Progress-authorized (matched spend)",
                "B5_worst_case": "Worst-case reserve", "ReserveGuard_d5": "ReserveGuard ($\\delta$=0.05)",
                "ReserveGuard_d10": "ReserveGuard ($\\delta$=0.10)", "B4_oracle": "Oracle"}
FIG3_POLICIES = ["ReserveGuard_d5", "ReserveGuard_d10", "B4_oracle", "B5_worst_case", "B0_fixed_cap", "B1_fixed_30pct",
                 "B2_self_estimate", "B3_progress_same_ceiling"]
STYLE = {"ReserveGuard_d5": dict(color="#1b5e20", lw=2.0, marker="o", ms=4.2, zorder=6),
         "ReserveGuard_d10": dict(color="#43a047", lw=2.0, marker="s", ms=4.2, zorder=6),
         "B4_oracle": dict(color="#212121", lw=1.0, ls=(0, (4, 2)), marker="*", ms=6.0, zorder=8),
         "B5_worst_case": dict(color="#6d4c41", lw=1.2, marker="^", ms=3.8, zorder=4),
         "B0_fixed_cap": dict(color="#c62828", lw=1.2, marker="v", ms=3.8, zorder=4),
         "B1_fixed_30pct": dict(color="#ef6c00", lw=1.2, marker="D", ms=3.4, zorder=4),
         "B2_self_estimate": dict(color="#6a1b9a", lw=1.2, marker="P", ms=4.2, zorder=4),
         "B3_progress_same_ceiling": dict(color="#1565c0", lw=1.2, marker="X", ms=4.2, zorder=4)}
LEVELS = ["0.5x", "0.75x", "1.0x", "1.5x"]
LEVEL_X = [0.5, 0.75, 1.0, 1.5]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 120, "savefig.dpi": DPI, "pdf.fonttype": 42})


def save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
        paths.append(p)
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------- Fig. 1: framework
def _box(ax, x, y, w, h, text, fc, ec="#37474f", fs=7.2, bold=False, radius=0.08, lw=0.9, tc="#212121"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.015,rounding_size={radius}", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal",
            linespacing=1.15)


def _arrow(ax, p, q, color="#37474f", lw=1.0, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1, zorder=5))


def fig_framework(out_dir: Path) -> list[Path]:
    """Banded architecture figure: the agent loop, the host budget runtime with the two-point gate and the
    closure sandbox, and the offline calibration that produces the certified reserves."""
    fig, ax = plt.subplots(figsize=(7.1, 4.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.0); ax.axis("off")
    bands = [(4.85, 2.1, "#eef4fb", "#1e3a5f", "Agent loop\n(untrusted returns)"),
             (2.35, 2.3, "#eef7ee", "#1b5e20", "Host budget runtime\n(trusted state only)"),
             (0.05, 2.1, "#fff5e6", "#8d4e00", "Offline calibration\n(once per deployment)")]
    for y, h, fc, tc, label in bands:
        ax.add_patch(Rectangle((0.05, y), 9.9, h, fc=fc, ec="none", zorder=0))
        ax.text(0.27, y + h / 2, label, fontsize=6.9, fontweight="bold", color=tc, ha="center", va="center", rotation=90)
    # --- agent loop (y 5.25 .. 6.85)
    _box(ax, 0.75, 5.6, 1.35, 1.0, "Task\nprompt", "#ffffff")
    _box(ax, 2.5, 5.6, 2.0, 1.0, "Model call $t$\nprompt $\\leq$ context\noutput $\\leq m_{out}$", "#dbe9f9", fs=6.8)
    _box(ax, 4.9, 5.6, 1.7, 1.0, "Proposed calls\n$k \\leq k_{max}$", "#dbe9f9", fs=6.8)
    _box(ax, 7.15, 6.15, 2.6, 0.62, "Trusted tools:\nhold, verify, confirm, pay, reply", "#e3f2e1", fs=6.2)
    _box(ax, 7.15, 5.42, 2.6, 0.62, "Untrusted tools:\nsearch, inbox (attacker text)", "#fde2e1", fs=6.2)
    _arrow(ax, (2.1, 6.1), (2.5, 6.1)); _arrow(ax, (4.5, 6.1), (4.9, 6.1))
    _arrow(ax, (6.6, 6.25), (7.15, 6.45)); _arrow(ax, (6.6, 5.95), (7.15, 5.73))
    _arrow(ax, (8.45, 5.42), (3.5, 5.6), rad=-0.16, color="#78909c", lw=0.9)
    ax.text(5.95, 4.9, "returns, capped at $L$ tokens and charged $w$ per call, join the next prompt", fontsize=6.2, ha="center", color="#455a64")
    # --- host runtime (y 2.6 .. 4.4)
    _box(ax, 0.75, 2.75, 2.1, 1.25, "Ledger in TE\ncaps $L$, $k_{max}$, $H_{max}$\nverify-before-commit\nno duplicate commit", "#ffffff", fs=6.5)
    _box(ax, 3.1, 2.75, 2.7, 1.25, "Two-point gate\nA, before model call $t$:\nleft $-$ (prompt $+ m_{out}$) $\\geq R(h,c)$\nB, before executing the calls:\nleft $-$ $w\\,k$ $\\geq R(h^\\prime,c^\\prime)$", "#c8e6c9", fs=6.2)
    _box(ax, 6.05, 2.75, 1.25, 1.25, "Cut:\ntrusted\nsnapshot $\\sigma$", "#ffffff", fs=6.6)
    _box(ax, 7.55, 2.75, 2.25, 1.25, "Closure sandbox\nclosure agent with hold\ntools only, $\\leq M_c$ steps,\nhost settle as fallback", "#c8e6c9", fs=6.2)
    _arrow(ax, (2.85, 3.37), (3.1, 3.37)); _arrow(ax, (5.8, 3.37), (6.05, 3.37)); _arrow(ax, (7.3, 3.37), (7.55, 3.37))
    _arrow(ax, (4.45, 4.0), (4.45, 5.6), color="#2e7d32", lw=1.2, style="<|-|>")
    ax.text(4.55, 4.55, "allow or refuse", fontsize=6.3, color="#2e7d32", va="center")
    _arrow(ax, (1.8, 5.6), (1.8, 4.0), color="#37474f", lw=0.9)
    ax.text(1.9, 4.55, "usage, calls, state", fontsize=6.3, color="#37474f", va="center")
    ax.text(8.67, 4.08, "closure order: verified $\\succ$ settle $\\succ$ draft $\\succ$ stop,\nthe most useful one the reserve affords", fontsize=6.0,
            ha="center", color="#1b5e20", va="bottom")
    ax.text(8.67, 2.5, "no open hold, no duplicate effect, no skipped check", fontsize=6.1, ha="center", color="#1b5e20", style="italic")
    # --- calibration (y 0.25 .. 1.95)
    _box(ax, 0.75, 0.45, 1.9, 1.1, "Recorded trajectories\ncalibration tasks,\nevery condition", "#ffffff", fs=6.4)
    _box(ax, 2.9, 0.45, 2.0, 1.1, "Closure tables\nevery cut $\\times$ kind,\nrun once, cost in TE", "#ffe0b2", fs=6.4)
    _box(ax, 5.15, 0.45, 2.1, 1.1, "Task scores per cell $(h,c)$\nlargest closure cost over\ncuts and conditions", "#ffe0b2", fs=6.4)
    _box(ax, 7.5, 0.45, 2.3, 1.1, "Split-conformal reserve\n$R_\\delta = C_{(\\lceil (n+1)(1-\\delta)\\rceil)}$\npooled over $c$, then $h$;\nbound as fallback", "#ffcc80", fs=6.2)
    _arrow(ax, (2.65, 1.0), (2.9, 1.0)); _arrow(ax, (4.9, 1.0), (5.15, 1.0)); _arrow(ax, (7.25, 1.0), (7.5, 1.0))
    ax.plot([8.65, 8.65, 4.45], [1.55, 2.2, 2.2], color="#e65100", lw=1.3, solid_capstyle="round", zorder=5)
    _arrow(ax, (4.45, 2.2), (4.45, 2.75), color="#e65100", lw=1.3)
    ax.text(6.55, 2.26, "$R(h,c)$ per cell, loaded into the gate", fontsize=6.3, color="#e65100", ha="center", va="bottom")
    return save(fig, out_dir, "fig1_framework")


# ----------------------------------------------------------------------------- Fig. 2: gate mechanism on one run
def fig_gate_mechanism(reports: Path, tag: str, out_dir: Path, task_id: str = "user_task_4", cond: str = "A0", level: str = "0.75x") -> list[Path]:
    """One recorded run (a hotel task, clean, budget 0.75x) replayed under ReserveGuard and under the fixed cap:
    cumulative spend per model call, the budget, the reserve the gate keeps, where each policy cuts and what the
    closure does."""
    rec = json.load(gzip.open(reports / "01_records" / tag / cond / f"{task_id}.json.gz", "rt"))
    rows = json.load(gzip.open(reports / "02_tables" / tag / cond / f"{task_id}.closures.json.gz", "rt"))["rows"]
    res = pd.read_csv(reports / f"03_results_{tag}.csv")
    B = json.loads((reports / f"03_budgets_{tag}.json").read_text())["levels"][level]
    reserves = json.loads((reports / f"02_reserves_{tag}.json").read_text())["reserves"]
    steps = rec["steps"]; w = rec["presets"]["w"]
    sel = res[(res.task_id == task_id) & (res.condition == cond) & (res.B_level == level)].set_index("policy")
    rg, fc = sel.loc["ReserveGuard_d10"], sel.loc["B0_fixed_cap"]
    R = float(reserves["0.1"]["verified"]["1,0"]["R"])
    ver = {int(r["t"]): r for r in rows if r["kind"] == "verified"}

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.9), sharex=False)
    for ax, (name, row, color, is_rg) in zip(axes, [("ReserveGuard ($\\delta$=0.10)", rg, "#2e7d32", True), ("Fixed cap", fc, "#c62828", False)]):
        cut = int(row["cut"])
        shown = steps[:cut] if is_rg else steps[:cut + 1]          # the fixed cap pays for the call that overspends
        base, labels_done = 0, set()
        for s_ in shown:
            t = s_["t"]
            executed = is_rg or t <= cut
            parts = [(s_["prompt_tokens"], "#90a4ae", "prompt"), (s_["completion_tokens"], "#455a64", "output"),
                     (s_["k_executed"] * w if executed else 0, "#ffb74d", "tool calls ($w$ each)")]
            for val, col, lab in parts:
                if val <= 0:
                    continue
                ax.bar(t, val, bottom=base, color=col, width=0.62, label=lab if lab not in labels_done else None,
                       hatch="///" if not executed else None, alpha=0.55 if not executed else 1.0)
                labels_done.add(lab); base += val
            if executed and s_["h"] > 0:
                ax.text(t, base + 120, "hold\nopen", ha="center", va="bottom", fontsize=6, color="#bf360c")
        x_cl = shown[-1]["t"] + 1
        closure_cost = float(ver[cut]["cost"])
        if is_rg:
            ax.bar(x_cl, closure_cost, bottom=base, color=color, width=0.62, label="verified closure")
            ax.text(x_cl + 0.4, base + closure_cost / 2, "closure:\nverify and\nconfirm", ha="left", va="center", fontsize=6, color=color)
            ax.axhspan(B - R, B, color="#a5d6a7", alpha=0.45, lw=0)
            ax.text(4.3, B - R / 2, f"reserve $R(1,0)$\n= {R:,.0f} TE\nkept for the\nclosure", fontsize=6.2, va="center", ha="left", color="#1b5e20")
            ax.annotate("gate refuses\nmodel call 3", xy=(cut + 0.42, base + 100), xytext=(cut + 0.6, base - 2900), fontsize=6.2, color="#1b5e20",
                        arrowprops=dict(arrowstyle="-|>", color="#1b5e20", lw=0.9))
            ax.set_xticks(list(range(1, x_cl + 1))); ax.set_xticklabels([str(t) for t in range(1, x_cl)] + ["closure"])
        else:
            ax.text(x_cl, 400, "no budget left:\nhold stays open", ha="center", va="bottom", fontsize=6.0, color=color)
            ax.annotate(f"call {cut + 1} overspends: cut,\ncalls not executed", xy=(cut + 1, B + 50), xytext=(cut - 1.6, B + 1400), fontsize=6.2, color="#c62828",
                        arrowprops=dict(arrowstyle="-|>", color="#c62828", lw=0.9))
            ax.set_xticks(list(range(1, x_cl + 1))); ax.set_xticklabels([str(t) for t in range(1, x_cl)] + ["closure"])
        ax.axhline(B, color="#212121", lw=1.0, ls="--")
        ax.text(0.62, B + 120, f"budget $B$ = {B:,} TE", fontsize=6.4, va="bottom")
        ax.set_title(f"{name}: {'task completed, no open hold' if row['utility'] else 'hold abandoned'}", fontsize=7.6, color=color)
        ax.set_xlabel("Model call")
        ax.set_ylabel("Cumulative spend (TE)")
        ax.set_ylim(0, B * 1.3)
        ax.set_xlim(0.3, 5.7)
    h1, l1 = axes[0].get_legend_handles_labels()
    fig.legend(h1, l1, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.0)
    return save(fig, out_dir, "fig2_gate_mechanism")


# ----------------------------------------------------------------------------- Fig. 3: completion and adverse outcomes
def fig_completion_apo(reports: Path, tag: str, out_dir: Path, conditions=("A0", "A123", "A7")) -> list[Path]:
    m3 = pd.read_csv(reports / f"03_metrics_{tag}.csv")
    m4 = pd.read_csv(reports / f"04_metrics_{tag}.csv") if (reports / f"04_metrics_{tag}.csv").exists() else pd.DataFrame()
    m = pd.concat([m3, m4], ignore_index=True)
    conds = [c for c in conditions if c in set(m.condition)]
    fig, axes = plt.subplots(2, len(conds), figsize=(5.3, 2.5), sharex=True, sharey="row")
    axes = np.atleast_2d(axes)
    for j, cond in enumerate(conds):
        for i, (col, ylab) in enumerate([("utility_pct", "Completion (%)"), ("apo_pct", "Adverse (%)")]):
            ax = axes[i, j]
            for pol in FIG3_POLICIES:
                g = m[(m.condition == cond) & (m.policy == pol)].set_index("B_level").reindex(LEVELS)
                if g[col].isna().all():
                    continue
                ax.plot(LEVEL_X, g[col].values, label=POLICY_NAMES[pol], **STYLE[pol])
            ax.set_xticks(LEVEL_X); ax.set_xticklabels(["0.5", "0.75", "1.0", "1.5"])
            ax.grid(True, lw=0.4, alpha=0.5)
            if i == 0:
                ax.set_title(CONDITION_NAMES[cond])
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 1 and j == len(conds) // 2:
                ax.set_xlabel("Budget ($\\times$ median clean cost)")
            ax.set_ylim(-3, 103) if i == 0 else ax.set_ylim(-2, 62)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.05), fontsize=6.0,
               handlelength=2.0, columnspacing=0.8, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1), h_pad=0.7, w_pad=0.5)
    return save(fig, out_dir, "fig3_completion_adverse")


# ----------------------------------------------------------------------------- Fig. 4: reserve cells
def _scores(reports: Path, tag: str, task_ids, conditions) -> pd.DataFrame:
    rows = []
    for cond in conditions:
        for tid in task_ids:
            p = reports / "02_tables" / tag / cond / f"{tid}.closures.json.gz"
            if not p.exists():
                continue
            for r in json.load(gzip.open(p, "rt"))["rows"]:
                if r["kind"] == "verified" and r["h"] > 0 and not json.loads(r["detail"])["cut_off"]:
                    rows.append({"task_id": tid, "h": r["h"], "c": min(r["confirmed_before"], 1), "cost": r["cost"], "condition": cond})
    v = pd.DataFrame(rows)
    return v.groupby(["task_id", "h", "c"])["cost"].max().reset_index() if len(v) else v


def fig_reserve_cells(reports: Path, tag: str, out_dir: Path, conditions) -> list[Path]:
    split = json.loads((reports / "02_split.json").read_text())
    RES = json.loads((reports / f"02_reserves_{tag}.json").read_text())
    cal = _scores(reports, tag, split["calibration"], conditions)
    test = _scores(reports, tag, split["test"], conditions)
    cells = [(1, 0), (1, 1), (2, 0)]
    labels = {(1, 0): "1 hold open,\nnone confirmed", (1, 1): "1 hold open,\n1 confirmed", (2, 0): "2 holds open,\nnone confirmed"}
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    rng = np.random.default_rng(0)
    for k, (h, c) in enumerate(cells):
        a = cal[(cal.h == h) & (cal.c == c)].cost.values; b = test[(test.h == h) & (test.c == c)].cost.values
        ax.scatter(k - 0.18 + rng.uniform(-0.07, 0.07, len(a)), a, s=11, color="#546e7a", label="calibration task (largest closure)" if k == 0 else None, zorder=3)
        ax.scatter(k + 0.18 + rng.uniform(-0.07, 0.07, len(b)), b, s=14, marker="x", color="#c62828", label="held-out task" if k == 0 else None, zorder=3)
        key = f"{h},{c}"
        R05 = RES["reserves"]["0.05"]["verified"][key]["R"]; R10 = RES["reserves"]["0.1"]["verified"][key]["R"]
        bound = RES["bounds"]["0.0"]["verified"][key]["R"]
        ax.hlines(R05, k - 0.42, k + 0.42, color="#1b5e20", lw=1.8, label="reserve, $\\delta$=0.05" if k == 0 else None, zorder=4)
        ax.hlines(R10, k - 0.42, k + 0.42, color="#43a047", lw=1.4, ls=(0, (3, 2)), label="reserve, $\\delta$=0.10" if k == 0 else None, zorder=4)
        ax.hlines(bound, k - 0.42, k + 0.42, color="#6d4c41", lw=1.2, ls=":", label="worst-case bound" if k == 0 else None, zorder=4)
        pooled = RES["reserves"]["0.05"]["verified"][key].get("pooled", "cell")
        if pooled not in ("cell", None, False):
            ax.text(k - 0.4, R05 + 220, "pooled", fontsize=6, ha="left", color="#1b5e20")
    ax.set_xticks(range(len(cells))); ax.set_xticklabels([labels[c] for c in cells], fontsize=6.8)
    ax.set_ylabel("Verified-closure cost (TE)")
    ax.set_ylim(0, 17500)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 0.9), frameon=False, fontsize=6.2, ncol=1)
    fig.tight_layout()
    return save(fig, out_dir, "fig4_reserve_cells")


# ----------------------------------------------------------------------------- tables
def _tex_escape(s: str) -> str:
    return str(s).replace("%", "\\%").replace("_", "\\_")


def table_attack_effect(reports: Path, tag: str, out_dir: Path) -> pd.DataFrame:
    e1 = pd.read_csv(reports / f"01_attack_effect_{tag}.csv")
    e4 = pd.read_csv(reports / f"04_attack_effect_{tag}.csv") if (reports / f"04_attack_effect_{tag}.csv").exists() else pd.DataFrame()
    e = pd.concat([e1, e4[e4.condition != "A0"]], ignore_index=True) if len(e4) else e1
    cont = pd.read_csv(reports / f"04_containment_{tag}.csv").set_index("condition") if (reports / f"04_containment_{tag}.csv").exists() else None
    out = pd.DataFrame({"Condition": [CONDITION_NAMES.get(c, c) for c in e.condition],
                        "Cost factor": e.cost_factor_vs_A0.round(2), "Extra steps": e.extra_steps_vs_A0.round(2),
                        "Retries": e.mean_retries.round(2), "Max prompt (tok.)": e.max_prompt_tokens.astype(int),
                        "Completed (%)": e.utility_pct.round(1), "Open hold at end (%)": e.open_hold_pct.round(1)})
    if cont is not None:
        out["Refusals: hold cap"] = [int(cont.loc[c, "H_max_refusals"]) if c in cont.index else 0 for c in e.condition]
        out["Refusals: unverified"] = [int(cont.loc[c, "unverified_refusals"]) if c in cont.index else 0 for c in e.condition]
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "table1_attack_effect.csv", index=False)
    (out_dir / "table1_attack_effect.tex").write_text(out.to_latex(index=False, escape=True, column_format="l" + "r" * (len(out.columns) - 1)))
    return out


def table_reserves(reports: Path, tag: str, out_dir: Path) -> pd.DataFrame:
    cells = pd.read_csv(reports / f"02_reserve_cells_{tag}.csv")
    tcov = pd.read_csv(reports / f"02_task_coverage_{tag}.csv")
    tcov7 = pd.read_csv(reports / f"04_task_coverage_{tag}.csv") if (reports / f"04_task_coverage_{tag}.csv").exists() else None
    names = {(1, 0): "one hold open, none confirmed", (1, 1): "one hold open, one confirmed", (2, 0): "two holds open"}
    rows = []
    for (h, c), g in cells.groupby(["h", "c"]):
        r05 = g[g.delta == 0.05].iloc[0]; r10 = g[g.delta == 0.10].iloc[0]
        t05 = tcov[(tcov.h == h) & (tcov.c == c) & (tcov.delta == 0.05)].iloc[0]
        row = {"Cell (h, c)": f"({h}, {c}) {names.get((h, c), '')}", "Cal. tasks": int(r05.n_cell), "Pooled": "no" if r05.pooled == "cell" else f"yes ({r05.pooled})",
               "R, d=0.05 (TE)": int(r05.R), "R, d=0.10 (TE)": int(r10.R), "Bound (TE)": int(r05.bound), "R/bound (d=0.05)": round(r05.R_over_bound, 2),
               "Held-out tasks covered (d=0.05)": f"{int(t05.covered)}/{int(t05.n_tasks)}"}
        if tcov7 is not None:
            t7 = tcov7[(tcov7.h == h) & (tcov7.c == c) & (tcov7.delta == 0.05)]
            row["Covered under A7"] = f"{int(t7.iloc[0].covered)}/{int(t7.iloc[0].n_tasks)}" if len(t7) else "-"
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "table2_reserves.csv", index=False)
    (out_dir / "table2_reserves.tex").write_text(out.to_latex(index=False, escape=True, column_format="l" + "r" * (len(out.columns) - 1)))
    return out


def table_policies(reports: Path, tag: str, out_dir: Path, level: str = "1.0x", conditions=("A0", "A123", "A6", "A7")) -> pd.DataFrame:
    m3 = pd.read_csv(reports / f"03_metrics_{tag}.csv")
    m4 = pd.read_csv(reports / f"04_metrics_{tag}.csv") if (reports / f"04_metrics_{tag}.csv").exists() else pd.DataFrame()
    m = pd.concat([m3, m4], ignore_index=True)
    conds = [c for c in conditions if c in set(m.condition)]
    pols = ["B0_fixed_cap", "B1_fixed_10pct", "B1_fixed_30pct", "B2_self_estimate", "B3_progress_same_ceiling", "B5_worst_case",
            "ReserveGuard_d5", "ReserveGuard_d10", "B4_oracle"]
    rows = []
    for pol in pols:
        row = {"Policy": POLICY_NAMES[pol].replace("$\\delta$", "d")}
        for cond in conds:
            g = m[(m.condition == cond) & (m.policy == pol) & (m.B_level == level)]
            row[f"{CONDITION_NAMES[cond]}: adverse (%)"] = float(g.apo_pct.iloc[0]) if len(g) else float("nan")
            row[f"{CONDITION_NAMES[cond]}: completed (%)"] = float(g.utility_pct.iloc[0]) if len(g) else float("nan")
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / f"table3_policies_{level}.csv", index=False)
    (out_dir / f"table3_policies_{level}.tex").write_text(out.to_latex(index=False, escape=True, float_format="%.1f", column_format="l" + "r" * (len(out.columns) - 1)))
    return out


def numbers_for_text(reports: Path, tag: str) -> dict:
    """Every number the manuscript quotes, read from the committed reports, so the text can be checked against them."""
    out = {}
    c1 = json.loads((reports / f"01_claims_{tag}.json").read_text()); out["clean_utility"] = c1["clean_utility"]; out["cost_factor"] = c1["cost_factor"]
    c2 = json.loads((reports / f"02_claims_{tag}.json").read_text()); out["task_cells"] = c2["task_cells"]; out["closure_coverage"] = c2["closure_coverage_by_condition"]
    out["refused_duplicates_test"] = c2["refused_duplicates_test"]; out["over_commit_test"] = c2["over_commit_closures_test"]
    c3 = json.loads((reports / f"03_claims_{tag}.json").read_text()); out["budget_levels"] = c3["budget_levels"]; out["clean_median_TE"] = c3["clean_median_TE"]
    out["C1"] = c3["C1"]["holds"]; out["C2"] = c3["C2_preregistered"]["holds"]; out["C3"] = c3["C3"]
    m = pd.read_csv(reports / f"03_metrics_{tag}.csv")
    out["metrics_1.0x"] = {f"{r.condition}|{r.policy}": {"apo": r.apo_pct, "utility": r.utility_pct} for r in m[m.B_level == "1.0x"].itertuples()}
    out["metrics_1.5x"] = {f"{r.condition}|{r.policy}": {"apo": r.apo_pct, "utility": r.utility_pct} for r in m[m.B_level == "1.5x"].itertuples()}
    out["apo_max_by_policy"] = m.groupby("policy").apo_pct.max().round(1).to_dict()
    if (reports / f"04_claims_{tag}.json").exists():
        c4 = json.loads((reports / f"04_claims_{tag}.json").read_text()); out["A7"] = {k: c4[k] for k in ("C4_on_A7", "C1_on_A7", "C2_on_A7", "coverage_on_A7")}
        out["containment"] = c4["containment"]
    cells = pd.read_csv(reports / f"02_reserve_cells_{tag}.csv"); out["reserve_cells"] = cells.to_dict(orient="records")
    return out
