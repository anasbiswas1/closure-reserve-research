"""Record-once runner (plan v2.1, Section 8.2). Runs one (task, condition) pair through the budget runtime with
no budget policy, records every step's usage, calls, refusals and state, and writes the record so that every
policy and budget level can later be evaluated by replaying the recorded prefix to its cut point."""
from __future__ import annotations

import gzip
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from agentdojo.attacks.attack_registry import load_attack
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.task_suite import functions_stack_trace_from_messages, model_output_from_messages

import config
import budget_attacks  # noqa: F401  (registers the attacks)
import bookings_suite
from budget_runtime import CloseNow, Ledger, TokenCounter, build_pipeline


def serialize_messages(messages) -> list[dict]:
    out = []
    for m in messages:
        d = {"role": m["role"]}
        content = m.get("content")
        d["content"] = "" if content is None else "\n".join(c.get("content") or "" for c in content)
        if m["role"] == "assistant":
            d["tool_calls"] = [{"function": c.function, "args": json.loads(json.dumps(dict(c.args), default=str)),
                                "id": c.id} for c in (m.get("tool_calls") or [])]
        if m["role"] == "tool":
            d["tool_call"] = {"function": m["tool_call"].function,
                              "args": json.loads(json.dumps(dict(m["tool_call"].args), default=str))}
            d["error"] = m.get("error")
        out.append(d)
    return out


TOOL_LEVEL_ATTACKS = {"endpoint"}       # conditions that compromise the runtime, not the data


def injections_for(suite, user_task, attack_name: str | None, pipeline) -> dict[str, str]:
    if attack_name is None or attack_name in TOOL_LEVEL_ATTACKS:
        return {}
    attack = load_attack(attack_name, suite, pipeline)
    injection_task = next(iter(suite.injection_tasks.values()))
    return attack.attack(user_task, injection_task)


def make_runtime(suite, attack_name: str | None, presets: dict, counter: TokenCounter):
    """The runtime the recorded run executes its calls on: AgentDojo's for every data-level condition, the
    compromised one for the tool-level condition."""
    if attack_name == "endpoint":
        return budget_attacks.CompromisedRuntime(suite.tools, counter, presets["L"], config.UNTRUSTED_TOOLS, presets.get("E", 1))
    return FunctionsRuntime(suite.tools)


def run_task(suite, user_task, attack_name: str | None, make_llm, presets: dict, counter: TokenCounter,
             policy=None, B: int | None = None, host_verify: bool = True, context_limit: int | None = None,
             model_name: str = "unknown") -> dict:
    """make_llm(user_task) returns the model element (so a scripted agent can be bound to the task)."""
    llm = make_llm(user_task)
    probe = build_pipeline(llm, presets, counter, None, None, host_verify, context_limit)
    injections = injections_for(suite, user_task, attack_name, probe)
    env = suite.load_and_inject_default_environment(injections)
    task_env = user_task.init_environment(env)
    pre = task_env.model_copy(deep=True)
    gt_names = [c.function for c in user_task.ground_truth(pre)]
    runtime = make_runtime(suite, attack_name, presets, counter)
    ledger = Ledger(B=B, w=presets["w"])
    extra = {"ledger": ledger, "executed": []}
    pipeline = build_pipeline(llm, presets, counter, policy, gt_names, host_verify, context_limit)
    queued: list[dict] = []
    cut_reason = None
    try:
        _, _, task_env, messages, extra = pipeline.query(user_task.PROMPT, runtime, task_env, [], extra)
    except CloseNow as e:
        messages, task_env = e.messages, e.task_environment
        queued = [{"function": c.function, "args": dict(c.args)} for c in e.queued]
        cut_reason = e.reason
    if len(ledger.steps) >= presets["M"] and cut_reason is None and ledger.end_reason == "finished":
        ledger.end_reason = "max_iters"
    model_output = model_output_from_messages(messages) or []
    trace = functions_stack_trace_from_messages(messages)
    utility = bool(suite._check_user_task_utility(user_task, model_output, pre, task_env, trace))
    executed = extra["executed"]
    outcomes = bookings_suite.outcomes(pre, task_env, executed, getattr(user_task, "FAMILY", ""))
    retries = sum(1 for a, b in zip(executed, executed[1:]) if a["function"] == b["function"] and a["args"] == b["args"])
    return {
        "runtime_version": config.RUNTIME_VERSION,
        "task_id": user_task.ID, "family": getattr(user_task, "FAMILY", ""), "attack": attack_name or "A0",
        "model": model_name, "policy": getattr(policy, "name", "record"), "B": B, "host_verify": host_verify,
        "presets": presets, "token_counter": counter.kind, "injected_slots": sorted(injections),
        "ledger": ledger.summary(), "steps": [asdict(s) for s in ledger.steps], "executed": executed,
        "queued_at_cut": queued, "cut_reason": cut_reason, "utility": utility, **outcomes,
        "retries": retries, "ask_user_calls": sum(1 for c in executed if c["function"] == "ask_user"),
        "tool_errors": sum(1 for c in executed if c.get("error") is not None),
        "refused_total": sum(len(s.refused) for s in ledger.steps),
        "milestones": _final_milestones(executed, gt_names), "gt_len": len(gt_names),
        "messages": serialize_messages(messages),
        "final_text": model_output[0]["content"] if model_output else "",
    }


def _final_milestones(executed, gt_names) -> int:
    j = 0
    for c in executed:
        if c.get("error") is None and j < len(gt_names) and c["function"] == gt_names[j]:
            j += 1
    return j


def record_path(root: Path, model_tag: str, condition: str, task_id: str) -> Path:
    return root / model_tag / condition / f"{task_id}.json.gz"


def save_record(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(rec, f, default=str)


def load_record(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


SUMMARY_COLS = ["task_id", "family", "attack", "model", "utility", "spent", "n_steps", "tool_calls_total",
                "prompt_tokens_total", "max_prompt_tokens", "end_reason", "cut_reason", "g2_open_hold", "open_holds",
                "dup_payment", "n_consequential", "retries", "ask_user_calls", "tool_errors", "refused_total", "milestones", "gt_len"]


def summary_row(rec: dict) -> dict:
    row = {k: rec.get(k) for k in SUMMARY_COLS if k in rec}
    row.update({k: rec["ledger"][k] for k in ("spent", "n_steps", "tool_calls_total", "prompt_tokens_total",
                                               "max_prompt_tokens", "end_reason")})
    return row


def records_version(root: Path, model_tag: str) -> str | None:
    """The runtime version shared by every record of this model on disk; None if any record is missing it or
    the records disagree (then everything is treated as stale)."""
    paths = list((root / model_tag).glob("*/*.json.gz")) if (root / model_tag).exists() else []
    if not paths:
        return None
    versions = set()
    for p in paths:
        try:
            versions.add(load_record(p).get("runtime_version"))
        except Exception:
            versions.add(None)
    return versions.pop() if len(versions) == 1 else None


def record_grid(suite, task_ids, conditions, make_llm, presets, counter, root: Path, model_tag: str,
                host_verify: bool = True, context_limit: int | None = None, force_rerun: bool = False,
                log=print, workers: int = 1) -> pd.DataFrame:
    """A record on disk is reused only if it carries the current runtime version; otherwise it is re-recorded.
    With workers > 1 the (condition, task) runs go through a thread pool: each run has its own environment,
    runtime, ledger and pipeline, and the model server batches the concurrent requests."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def one(job):
        cond, tid = job
        attack_name = config.ATTACK_BY_CONDITION[cond]
        path = record_path(root, model_tag, cond, tid)
        if path.exists() and not force_rerun:
            old = load_record(path)
            if old.get("runtime_version") == config.RUNTIME_VERSION:
                return "reused", summary_row(old), None
        task = suite.user_tasks[tid]
        rec = run_task(suite, task, attack_name, make_llm, presets, counter, None, None, host_verify, context_limit, model_tag)
        rec["condition"] = cond
        save_record(path, rec)
        r = summary_row(rec)
        line = (f"{cond} {tid} {rec['family']} utility={r['utility']} spent={r['spent']} steps={r['n_steps']} "
                f"end={r['end_reason']} open_holds={r['open_holds']} retries={r['retries']} asks={r['ask_user_calls']}")
        return "new", r, line

    jobs = [(cond, tid) for cond in conditions for tid in task_ids]
    by_job, n_new, n_reused = {}, 0, 0
    def take(job, result):
        nonlocal n_new, n_reused
        status, r, line = result
        by_job[job] = r
        if status == "new":
            n_new += 1; log(f"[{n_new + n_reused}/{len(jobs)}] {line}")
        else:
            n_reused += 1
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(one, j): j for j in jobs}
            for fut in as_completed(futures):
                take(futures[fut], fut.result())
    else:
        for j in jobs:
            take(j, one(j))
    log(f"records: {n_new} recorded now, {n_reused} reused (runtime version {config.RUNTIME_VERSION})")
    df = pd.DataFrame([by_job[j] for j in jobs])
    df.insert(2, "condition", [c for c in conditions for _ in task_ids])
    return df


def attack_effect_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per condition: mean cost, cost factor against A0 (paired by task), steps, retries, clarification turns,
    utility and how runs ended. Table 3 of the paper."""
    base = df[df.condition == "A0"].set_index("task_id")["spent"]
    rows = []
    for cond, g in df.groupby("condition", sort=False):
        paired = g.set_index("task_id")["spent"] / base.reindex(g["task_id"]).values
        rows.append({"condition": cond, "n": len(g), "mean_spent_TE": round(g["spent"].mean(), 0),
                     "cost_factor_vs_A0": round(float(paired.mean()), 2),
                     "mean_steps": round(g["n_steps"].mean(), 2),
                     "extra_steps_vs_A0": round(float(g["n_steps"].mean() - df[df.condition == "A0"]["n_steps"].mean()), 2),
                     "mean_retries": round(g["retries"].mean(), 2),
                     "mean_ask_user": round(g["ask_user_calls"].mean(), 2),
                     "utility_pct": round(100 * g["utility"].mean(), 1),
                     "open_hold_pct": round(100 * g["g2_open_hold"].mean(), 1),
                     "dup_payment_pct": round(100 * g["dup_payment"].mean(), 1),
                     "ended_by_ceiling_pct": round(100 * g["end_reason"].isin(["context_ceiling", "record_ceiling"]).mean(), 1),
                     "max_prompt_tokens": int(g["max_prompt_tokens"].max())})
    return pd.DataFrame(rows)
