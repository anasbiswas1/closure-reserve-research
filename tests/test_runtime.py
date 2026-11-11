"""CPU tests of the suite and the budget runtime with the scripted agent. Run from the repository root:
python -m pytest -q tests"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CLOSURE_REPO_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import bookings_data_gen as gen  # noqa: E402

gen.write(config.SUITE_DATA)

import bookings_suite as bs  # noqa: E402
import budget_runtime as br  # noqa: E402
import record  # noqa: E402
import scripted_agent as sa  # noqa: E402

SUITE = bs.suite
P = dict(config.PRESETS)
COUNTER = br.TokenCounter(None)
ONE_PER_FAMILY = [next(t.ID for t in SUITE.user_tasks.values() if t.FAMILY == f) for f in ("F1", "F2", "F3", "F4")]


def _mk(gullible=False, skip_verify=False, presets=None):
    presets = presets or P
    return lambda task: sa.ScriptedAgent(task, COUNTER, presets, gullible=gullible, skip_verify=skip_verify)


def test_suite_ground_truth_and_injectability():
    ok, (ut, _) = SUITE.check(check_injectable=False)
    assert ok, {k: v for k, v in ut.items() if not v[0]}
    assert len(SUITE.user_tasks) == 48


def test_clean_runs_complete_every_task():
    for tid, task in SUITE.user_tasks.items():
        rec = record.run_task(SUITE, task, None, _mk(), P, COUNTER)
        assert rec["utility"], tid
        assert rec["open_holds"] == 0 and rec["ledger"]["end_reason"] == "finished", tid


@pytest.mark.parametrize("attack", ["padding", "transient_error", "clarification_bait", "combined"])
def test_attacks_raise_cost_for_a_gullible_agent(attack):
    for tid in ONE_PER_FAMILY:
        task = SUITE.user_tasks[tid]
        clean = record.run_task(SUITE, task, None, _mk(True), P, COUNTER)
        hit = record.run_task(SUITE, task, attack, _mk(True), P, COUNTER)
        assert hit["ledger"]["spent"] >= clean["ledger"]["spent"], (attack, tid)


def test_pending_hold_cap_refuses_third_hold():
    task = SUITE.user_tasks[ONE_PER_FAMILY[1]]
    rec = record.run_task(SUITE, task, "pending_flood", _mk(True), P, COUNTER)
    assert any(r["why"] == "H_max" for s in rec["steps"] for r in s["refused"])


def test_host_verify_rule_refuses_unverified_consequential_calls():
    for tid in (ONE_PER_FAMILY[0], ONE_PER_FAMILY[2]):
        task = SUITE.user_tasks[tid]
        rec = record.run_task(SUITE, task, None, _mk(skip_verify=True), P, COUNTER)
        assert any(r["why"] == "unverified" for s in rec["steps"] for r in s["refused"])
        assert rec["utility"]   # the scripted agent verifies after the refusal and finishes
        rec_off = record.run_task(SUITE, task, None, _mk(skip_verify=True), P, COUNTER, host_verify=False)
        assert rec_off["refused_total"] == 0


class _FixedCap:
    name = "B0"

    def allow(self, remaining, cap, state):
        return remaining - cap >= 0


def test_gate_cuts_under_a_fixed_cap_and_records_queued_calls():
    task = SUITE.user_tasks[ONE_PER_FAMILY[0]]
    rec = record.run_task(SUITE, task, None, _mk(), P, COUNTER, policy=_FixedCap(), B=3000)
    assert rec["ledger"]["end_reason"] == "policy:B0" and rec["ledger"]["spent"] <= 3000
    rec = record.run_task(SUITE, task, None, _mk(), P, COUNTER, policy=_FixedCap(), B=10 ** 6)
    assert rec["utility"]


def test_ceilings():
    task = SUITE.user_tasks[ONE_PER_FAMILY[0]]
    p2 = dict(P); p2["record_ceiling"] = 2500
    assert record.run_task(SUITE, task, None, _mk(presets=p2), p2, COUNTER)["ledger"]["end_reason"] == "record_ceiling"
    assert record.run_task(SUITE, task, None, _mk(), P, COUNTER, context_limit=1500)["ledger"]["end_reason"] == "context_ceiling"


def test_record_round_trip(tmp_path):
    task = SUITE.user_tasks[ONE_PER_FAMILY[3]]
    rec = record.run_task(SUITE, task, "combined", _mk(True), P, COUNTER)
    path = tmp_path / "r.json.gz"
    record.save_record(path, rec)
    back = record.load_record(path)
    assert back["task_id"] == rec["task_id"] and len(back["messages"]) == len(rec["messages"])


def test_parse_calls_accepts_qwen_closers_and_stray_braces():
    good = br.parse_calls('Let us search.\n\n<function=search_hotels>{"city": "Prague", "start": "2027-06-10", "end": "2027-06-12"}</function>', 3)
    qwen = br.parse_calls('Let us search.\n\n<function=search_hotels>{"city": "Prague", "start": "2027-06-10", "end": "2027-06-12"}<function>', 3)
    stray = br.parse_calls('<function=search_inbox>{"query": "Lena Fischer Prague"}}</function>', 3)
    bare = br.parse_calls('<function=search_inbox>{"query": "Sam Reilly"}', 3)
    two = br.parse_calls('<function=a>{"p": 1}<function> then <function=b>{}</function>', 3)
    assert [c.function for c in good] == ["search_hotels"] and good[0].args["city"] == "Prague"
    assert [c.function for c in qwen] == ["search_hotels"] and qwen[0].args == good[0].args
    assert [c.function for c in stray] == ["search_inbox"] and stray[0].args == {"query": "Lena Fischer Prague"}
    assert [c.function for c in bare] == ["search_inbox"]
    assert [c.function for c in two] == ["a", "b"]
    assert br.parse_calls("no call here", 3) == []


def test_search_inbox_tolerates_extra_words():
    env = SUITE.load_and_inject_default_environment({})
    from agentdojo.functions_runtime import FunctionsRuntime
    rt = FunctionsRuntime(SUITE.tools)
    for q, sender in (("Alice Landlord", "Alice Landlord"), ("City Council Tax invoice", "City Council Tax"),
                      ("Lena Fischer Prague trip", "Lena Fischer"), ("email from Sam Reilly", "Sam Reilly")):
        res, err = rt.run_function(env, "search_inbox", {"query": q})
        assert err is None and len(res) == 1 and res[0].sender_name == sender, (q, err, [m.sender_name for m in res])
