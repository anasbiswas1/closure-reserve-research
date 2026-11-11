"""CPU tests of closure, reserves and replay on scripted records (one task per family)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CLOSURE_REPO_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import bookings_data_gen as gen  # noqa: E402

gen.write(config.SUITE_DATA)

import bookings_suite as bs  # noqa: E402
import budget_runtime as br  # noqa: E402
import closure as cl  # noqa: E402
import policies as pol  # noqa: E402
import record  # noqa: E402
import replay as rp  # noqa: E402
import reserve as rs  # noqa: E402
import scripted_agent as sa  # noqa: E402
import scripted_closure as sc  # noqa: E402

SUITE = bs.suite
P = dict(config.PRESETS)
COUNTER = br.TokenCounter(None)
IDS = [t.ID for t in SUITE.user_tasks.values() if int(t.ID.split("_")[-1]) % 12 < 2]   # two per family


def _records(tmp_path):
    mk = lambda task: sa.ScriptedAgent(task, COUNTER, P, gullible=True)
    df = record.record_grid(SUITE, IDS, ["A0", "A123"], mk, P, COUNTER, tmp_path / "records", "scripted", log=lambda s: None)
    probe = br.build_pipeline(mk(SUITE.user_tasks[IDS[0]]), P, COUNTER)
    runner = cl.ClosureRunner(SUITE, P, COUNTER, lambda sigma: sc.ScriptedClosureAgent(COUNTER, P))
    return df, probe, runner


def test_closure_tables_reserves_coverage_and_replay(tmp_path):
    df, probe, runner = _records(tmp_path)
    tab = rp.build_closure_tables(SUITE, runner, probe, P, tmp_path / "records", tmp_path / "tables", "scripted",
                                  ["A0", "A123"], IDS, log=lambda s: None)
    v = tab[tab.kind == "verified"]
    assert len(v) > 0 and not any(__import__("json").loads(d)["cut_off"] for d in v.detail)   # unconstrained closures finish
    assert (tab[tab.kind == "settle"].cost == tab[tab.kind == "settle"].h * P["w"]).all()       # settle is exact
    smax = rs.sigma_max_tokens(COUNTER, P, SUITE)
    reserves = rs.reserves_from_table(v, [0.5], P, COUNTER, smax)      # delta 0.5 so tiny cells calibrate; all conditions
    for cell in reserves["0.5"]["verified"].values():
        assert cell["R"] <= cell["bound"]
    bounds = rs.reserves_from_table(v, [0.0], P, COUNTER, smax)
    assert all(c["worst_case"] for k, c in bounds["0.0"]["verified"].items() if not k.startswith("0,"))
    assert all(c["R"] == 0 for k, c in bounds["0.0"]["verified"].items() if k.startswith("0,"))
    cov = rs.coverage(v, reserves, 0.5)
    assert len(cov) and (cov.coverage >= 0.5).all()
    med = df[df.condition == "A0"].spent.median()
    for cond in ("A0", "A123"):
        for tid in IDS[:2]:
            rec = record.load_record(record.record_path(tmp_path / "records", "scripted", cond, tid)); rec["condition"] = cond
            table = rp.table_by_cut(tab[(tab.task_id == tid) & (tab.condition == cond)].to_dict("records"))
            for B in (int(0.5 * med), int(1.5 * med)):
                o = rp.oracle_best(rec, B, P, table)
                assert not o["cut_off"] and o["spent_total"] <= B
                for p_ in (pol.FixedCap(), pol.FixedFraction(0.2), pol.ProgressAuthorized(), pol.WorstCase(bounds),
                           pol.ReserveGuard(reserves, 0.5)):
                    r = rp.replay(rec, p_, B, P, table, p_.chooser)
                    assert r["spent_total"] >= 0 and r["kind"] in cl.KINDS
                    if p_.name.startswith("ReserveGuard") or p_.name == "B5_worst_case":
                        assert not r["cut_off"] or r["kind"] == "verified"     # the safety classes are exact
                        assert int(r["utility"]) <= int(o["utility"]) or (r["utility"] and not o["utility"]) is False
