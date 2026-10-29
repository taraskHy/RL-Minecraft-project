"""Skill gate (parkour/gate.py): fixed slots play like their expert, decisions at starts and landings, the semi-MDP
returns, the block-kind features, swaps, a CPU smoke run, and the watchdog. CPU only."""
import csv
import dataclasses
import json
import math

import numpy as np
import pytest
import torch

from parkour import courses as C
from parkour import gate as G
from parkour import train as T
from parkour.env import CoursePool, EnvConfig, ParkourEnv
from parkour.model import ActorCritic
from parkour.paths import ROOT

FINAL = ROOT / "results" / "final_policy.pt"
NEO = ROOT / "results" / "neo_policy.pt"
S = C.JumpSpec


def tiny_expert(path, seed: int):
    """A small random policy in the checkpoint format parkour.train.load_frozen reads."""
    torch.manual_seed(seed)
    torch.save({"model": ActorCritic(hidden=(8,)).state_dict(), "config": {"hidden": [8], "hide": ""}}, path)
    return path


def experts_file(path, entries):
    path.write_text(json.dumps({"slots": [{"slot": n, "skill": s, "checkpoint": str(c)} for n, s, c in entries]}))
    return path


@pytest.fixture(scope="module")
def small_sets():
    """The measurement's validation sets with one course per level and part (36 courses, every group)."""
    return T.skill_sets("validation", 1, 10, ("basic", "advanced", "extra"), ice=True)


@pytest.fixture()
def three_slots(tmp_path):
    cfg = experts_file(tmp_path / "experts.json", [("final", "basic", FINAL), ("neo", "neo", NEO),
                                                   ("tiny", "all", tiny_expert(tmp_path / "tiny.pt", 0))])
    return G.read_slots(cfg)


def test_fixed_slot_plays_exactly_like_its_expert_alone(three_slots, small_sets):
    experts = G.Experts(three_slots, "cpu")
    completed = {}
    for k, s in enumerate(three_slots):
        ev = G.validate(None, experts, small_sets, "cpu", batch_size=600, stall_ticks=200, fixed_slot=k)
        alone = T.load_frozen(str(G.resolve(s["checkpoint"])), "cpu")
        completed[s["slot"]] = 0
        for g in T.GROUP_ORDER:
            for part, courses in small_sets[g].items():
                if not courses:
                    continue
                ref = T.evaluate(alone, CoursePool(courses, "cpu", C.GRID, compact_ice=True), "cpu", 200,
                                 per_course=True)["per_course"]
                got = ev["per_course"][g][part]
                assert got["success"] == ref["success"], (s["slot"], g, part)
                assert got["ticks"] == ref["ticks"], (s["slot"], g, part)
                completed[s["slot"]] += sum(ref["success"])
            assert ev["usage"][g]["decisions"][s["slot"]] == 1.0 and ev["usage"][g]["ticks"][s["slot"]] == 1.0
    assert completed["final"] > 10 and completed["neo"] > 10  # not only failures are compared


def check_decisions(trace, start_all=True):
    """Decisions only at an episode's start or right after a landing (cp increased), and a slot only changes there."""
    if start_all:
        assert trace[0]["decide"].all()
    landings = 0
    for prev, cur in zip(trace, trace[1:]):
        landed = ~prev["done"] & (cur["cp"] > prev["cp"])
        assert torch.equal(cur["decide"], prev["done"] | landed)
        keep = ~cur["decide"]
        assert torch.equal(cur["slot"][keep], prev["slot"][keep])
        landings += int(landed.sum())
    return landings


def test_decisions_only_at_the_start_and_at_landings(three_slots, small_sets):
    experts = G.Experts(three_slots, "cpu")
    torch.manual_seed(3)
    gate = G.Gate(len(three_slots), (16,))
    courses = small_sets["basic"]["standard"][:3] + small_sets["extra"]["standard"][:3]
    trace = []
    G.play(gate, experts, CoursePool(courses, "cpu", C.GRID, compact_ice=True), "cpu", trace=trace)
    assert check_decisions(trace) > 0
    # the training rollouts too, over two of them (options carried from the first end in the second)
    pool = CoursePool(courses, "cpu", C.GRID, compact_ice=True)
    env = ParkourEnv(pool, EnvConfig(num_envs=8, stall_ticks=200), sampler=None)
    col = G.Collector(env, gate, experts, G.GateFeatures(pool), 30, torch.Generator().manual_seed(0))
    trace = []
    col.collect(trace)
    col.collect(trace)
    assert check_decisions(trace) > 0
    assert len({int(s) for tr in trace for s in tr["slot"]}) > 1  # the sampled gate switches experts


def test_semi_mdp_transitions_and_advantages(three_slots, small_sets):
    """Each transition's reward is the plain sum of its ticks' rewards, its successor is discounted by
    gamma ** ticks, and GAE runs over decisions; checked against a per-environment recomputation from the trace,
    across two rollouts (a decision still running at a rollout's end completes in the next one)."""
    experts = G.Experts(three_slots, "cpu")
    torch.manual_seed(5)
    gate = G.Gate(len(three_slots), (16,))
    courses = small_sets["basic"]["standard"][:4] + small_sets["extra"]["obstacles"][:4]
    pool = CoursePool(courses, "cpu", C.GRID, compact_ice=True)
    n, steps, gamma, lam = 8, 40, 0.995, 0.95
    env = ParkourEnv(pool, EnvConfig(num_envs=n, stall_ticks=30), sampler=None)
    col = G.Collector(env, gate, experts, G.GateFeatures(pool), steps, torch.Generator().manual_seed(1))
    trace, values, batches, lasts = [], [], [], []
    for _ in range(2):
        col.collect(trace)
        values += [col.D_val[t + 1].clone() for t in range(steps)]  # the gate's value of every row at every tick
        batches.append(col.batch(gamma, lam))
        lasts.append(col.last_val.clone())
    dec = torch.stack([tr["decide"] for tr in trace])
    rew = torch.stack([tr["reward"] for tr in trace]).double()
    done = torch.stack([tr["done"] for tr in trace])
    trunc = torch.stack([tr["trunc"] for tr in trace])
    boot = torch.stack([tr["boot"] for tr in trace]).double()
    seen = {"done": 0, "time limit": 0, "landing": 0, "carried": 0}
    for r, b in enumerate(batches):
        lo, hi = r * steps, (r + 1) * steps - 1
        got = {(int(j), int(e)): (float(a), int(k), bool(d)) for (j, e), a, k, d in
               zip(b["index"].tolist(), b["adv"].tolist(), b["ticks"].tolist(), b["done"].tolist())}
        want = {}
        for e in range(n):
            starts = [t for t in range(len(trace)) if dec[t, e]]
            items = []  # the transitions ending in rollout r: (start, end, R, ticks, episode ended, bootstrap)
            for i, s in enumerate(starts):
                end = starts[i + 1] - 1 if i + 1 < len(starts) else (len(trace) - 1 if col.decide[e] else None)
                if end is not None and lo <= end <= hi:
                    items.append((s, end, float(rew[s:end + 1, e].sum()), end - s + 1, bool(done[end, e]),
                                  float(boot[end, e]) if trunc[end, e] else 0.0))
            adv_at = {}  # start tick -> advantage
            for s, end, R, k, d, bt in reversed(items):
                if d:  # the episode ended: no successor (a time limit bootstraps the final state's value)
                    v_next, a_next = bt, 0.0
                elif end + 1 <= hi:  # a landing: the next decision, one tick later
                    v_next, a_next = float(values[end + 1][e]), adv_at.get(end + 1, 0.0)
                else:  # a landing on the rollout's last tick: its value there, the chain cut
                    v_next, a_next = float(lasts[r][e]), 0.0
                adv_at[s] = R + gamma ** k * v_next - float(values[s][e]) + gamma ** k * lam * a_next
                want[(0 if s < lo else s - lo + 1, e)] = (adv_at[s], k, d)
                seen["done"] += d
                seen["time limit"] += bool(trunc[end, e])
                seen["landing"] += not d
                seen["carried"] += s < lo
        assert set(got) == set(want), r
        for key in want:
            assert got[key][1:] == want[key][1:], key
            assert got[key][0] == pytest.approx(want[key][0], rel=1e-5, abs=1e-4), key
    assert all(seen.values()), seen  # every kind of transition end was exercised


def hand_course(ceiling: int = 1) -> C.Course:
    """Platforms: 0 start, 1 stone (made ice below), 2 ice run-up, 3 its target, 4 bottom slab, 5 its exit,
    6 head-hitter runway (a roof over it; ``ceiling`` 2 also over the gap), 7 its target, 8 slime pad, 9 its exit,
    10 the top of a ladder climb."""
    c = C.from_specs([(S(1, 0, 0, (2, 2)), 0), (S(1, 0, 0, (1, 1), runway=3, ice=1), 0),
                      (S(1, -1, 0, (1, 1), block="bottom_slab"), 0), (S(1, 0, 0, (1, 1), runway=3, ceiling=ceiling), 0),
                      (S(1, -2, 0, (1, 1), bounce=2), 0), (S(1, 2, 0, (1, 1), ladder=1), 0)])
    c.ice = sorted(set(c.ice) | {1})
    return c


def test_block_kind_features_on_hand_made_courses():
    for ceiling in (1, 2):
        c = hand_course(ceiling)
        assert c.n == 11 and c.ice == [1, 2] and list(c.shapes) == [4] and c.slime == [8] and len(c.ladders)
        kind, roof = G.platform_kinds(c)
        names = [G.KINDS[k] for k in kind]
        assert names == ["stone", "ice", "ice", "stone", "shaped", "stone", "stone", "stone", "slime", "stone", "ladder"]
        assert roof.tolist() == [k == 6 for k in range(11)]
        pool = CoursePool([c, hand_course(ceiling)], "cpu", C.GRID, compact_ice=True)
        f = G.GateFeatures(pool)
        for cp in range(c.n):
            x = f(torch.tensor([1]), torch.tensor([cp]))[0]
            assert x[:5].tolist() == [float(i == kind[cp]) for i in range(5)]
            assert x[5:10].tolist() == ([float(i == kind[cp + 1]) for i in range(5)] if cp + 1 < c.n else [0.0] * 5)
            assert x[10].item() == float(cp + 1 < c.n and roof[cp + 1])
    # The current ice/slime/shape/ladder cells in the pool's world agree (the grids the physics reads).
    w = pool.world
    for k, name in enumerate(names):
        x0, x1, z0, z1, y = (int(v) for v in c.platforms[k])
        mat = int(w.slip.view(pool.size, *C.GRID)[0, x0, y, z0])
        assert mat == {"ice": 1, "slime": 2}.get(name, 0), (k, name)
        assert (int(w.shape_ids.view(pool.size, *C.GRID)[0, x0, y, z0]) != 0) == (name == "shaped"), (k, name)
    # Generated courses: beams, pillars and neo walls never count as a ceiling, and no platform has two kinds.
    rng = np.random.default_rng([2026, 99, 0])
    adv = C.load_levels(advanced=True)
    walls = 0
    for _ in range(40):
        oc = C.generate(rng, adv, int(rng.integers(len(adv))), 10, obstacle_prob=1.0)
        assert not G.platform_kinds(oc)[1].any()
        walls += len(oc.extra) > 0
    assert walls > 30
    cfg = T.TrainConfig(seed=0, skills="basic,advanced,extra", ice_prob=0.3, obstacle_prob=0.3)
    made = T.CourseFactory(cfg, C.load_levels(), set()).make(150)
    for mc in made:
        tags = G.platform_tags(mc)
        assert all(len(tg) <= 1 for tg in tags)
        if any(s.ceiling for s in mc.specs):
            assert G.platform_kinds(mc)[1].any()
        if len(mc.ladders):
            assert "ladder" in set().union(*tags)
    assert sum(G.platform_kinds(mc)[1].any() for mc in made) > 0


def test_swap_changes_only_its_slot_and_another_list_needs_swap(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "RUNS", tmp_path / "runs")
    a, b, c, c2 = (tiny_expert(tmp_path / f"{k}.pt", i) for i, k in enumerate(("a", "b", "c", "c2")))
    old = G.read_slots(experts_file(tmp_path / "old.json", [("x", "basic", a), ("y", "ice", b), ("z", "all", c)]))
    new = G.read_slots(experts_file(tmp_path / "new.json", [("x", "basic", a), ("y", "ice", b), ("z", "all", c2)]))
    assert G.compare_slots(old, [dict(s) for s in old], swap=False) == []
    with pytest.raises(ValueError, match="--swap"):
        G.compare_slots(old, new, swap=False)
    assert G.compare_slots(old, new, swap=True) == [{"slot": "z", "old_checkpoint": old[2]["checkpoint"],
                                                     "old_sha1": old[2]["sha1"], "new_checkpoint": new[2]["checkpoint"],
                                                     "new_sha1": new[2]["sha1"]}]
    for bad in ([("x", "basic", a), ("z", "all", c2), ("y", "ice", b)], [("x", "basic", a), ("y", "ice", b)],
                [("x", "basic", a), ("y", "extra", b), ("z", "all", c2)]):
        with pytest.raises(ValueError, match="slot names"):
            G.compare_slots(old, G.read_slots(experts_file(tmp_path / "bad.json", bad)), swap=True)
    # A gate checkpoint with the old slots; a fine-tune with the new expert in slot z.
    torch.manual_seed(0)
    gate = G.Gate(3, (8,))
    ck_path = tmp_path / "gate.pt"
    torch.save(G.checkpoint(gate, old, 9, {}), ck_path)
    cfg = G.GateConfig(run="ft", device="cpu", experts=str(tmp_path / "new.json"), init_from=str(ck_path),
                       num_envs=4, rollout=4, iterations=1, eval_every=1, val_per_level=1, eval_per_level=1,
                       pool_size=8, pool_refresh_every=0, stall_ticks=40, hidden=(8,), save_at="0")
    with pytest.raises(ValueError, match="--swap"):
        G.train(cfg)
    assert not (tmp_path / "runs" / "ft").exists()
    out = G.train(dataclasses.replace(cfg, swap=True))
    config = json.loads((out / "config.json").read_text())
    assert config["swaps"] == G.compare_slots(old, new, swap=True)
    assert config["slots"] == new and config["init_slots"] == old
    start = torch.load(out / "checkpoints" / "updates_00000.pt", weights_only=False)
    assert start["slots"] == new
    for k, v in gate.state_dict().items():
        assert torch.equal(start["gate"][k], v), k  # the gate itself is unchanged by the swap
    before, after = G.Experts(old, "cpu"), G.Experts(new, "cpu")
    for i in (0, 1):  # the other slots keep their networks
        assert all(torch.equal(p, q) for p, q in zip(before.nets[before.index[i]].parameters(),
                                                     after.nets[after.index[i]].parameters()))
    assert not all(torch.equal(p, q) for p, q in zip(before.nets[before.index[2]].parameters(),
                                                     after.nets[after.index[2]].parameters()))
    # one network per distinct checkpoint
    twice = G.read_slots(experts_file(tmp_path / "twice.json", [("x", "basic", a), ("w", "all", a)]))
    assert len(G.Experts(twice, "cpu").nets) == 1


def test_cpu_smoke_run_writes_every_file_and_reloads(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "RUNS", tmp_path / "runs")
    cfg_file = experts_file(tmp_path / "experts.json", [("final", "basic", FINAL),
                                                        ("tiny", "all", tiny_expert(tmp_path / "tiny.pt", 1))])
    cfg = G.GateConfig(run="smoke", device="cpu", experts=str(cfg_file), num_envs=16, rollout=24, iterations=2,
                       eval_every=1, val_per_level=1, eval_per_level=1, pool_size=16, pool_refresh_every=1,
                       stall_ticks=60, hidden=(16,), save_at="0,2")
    out = G.train(cfg)
    for f in ("config.json", "train_log.csv", "eval_val_log.csv", "summary.json", "checkpoints/best.pt",
              "checkpoints/latest.pt", "checkpoints/updates_00000.pt", "checkpoints/updates_00002.pt"):
        assert (out / f).is_file(), f
    rows = list(csv.DictReader((out / "train_log.csv").open()))
    assert [int(r["iteration"]) for r in rows] == [0, 1]
    for r in rows:
        assert int(r["transitions"]) > 0 and int(r["minibatch_updates"]) > 0
        for key in ("pi_loss", "v_loss", "entropy", "kl"):
            assert math.isfinite(float(r[key])), key
    vals = list(csv.DictReader((out / "eval_val_log.csv").open()))
    assert len(vals) == 2 and all(0.0 <= float(v["success"]) <= 1.0 for v in vals)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["completed"] and summary["iteration"] == 1 and "test_eval" not in summary
    best_it = max(range(2), key=lambda i: float(vals[i]["success"]))  # ties to the earlier
    assert summary["best"]["iteration"] == best_it
    assert set(summary["best"]["validation_groups"]) == set(T.GROUP_ORDER)
    assert set(summary["last_eval"]["slot_usage"]["basic"]["decisions"]) == {"final", "tiny"}
    gate, ck = G.load_gate(out / "checkpoints" / "best.pt", "cpu")
    assert ck["slots"] == G.read_slots(cfg_file) and ck["iteration"] == best_it
    sets = T.skill_sets("validation", 1, 10, ("basic", "advanced", "extra"), ice=True)
    ev = G.validate(gate, G.Experts(ck["slots"], "cpu"), sets, "cpu", stall_ticks=60)
    assert ev["success"] == pytest.approx(summary["best"]["validation_success"])
    with pytest.raises(ValueError, match="exists"):
        G.train(cfg)


def test_the_watchdog_pauses_the_gate():
    from scripts import gpu_watchdog as W
    assert W.pausable("python -m parkour.gate train --run gate-r1")
    assert W.pausable("python -m parkour.gate eval --checkpoint runs/gate-r1/checkpoints/best.pt")
    assert not W.pausable("python scripts/keep_awake.py --pattern parkour.gate")


def test_defaults_are_combined_distills_training_courses():
    """The training courses of skills mode with runs/combined-distill/config.json's settings (on the training PC)."""
    path = ROOT / "runs" / "combined-distill" / "config.json"
    if not path.exists():
        pytest.skip("runs/combined-distill is on the training PC")
    ref = json.loads(path.read_text())
    tc = G.course_config(G.GateConfig())
    for key in ("skills", "ice_prob", "obstacle_prob", "pool_size", "pool_refresh_every", "pool_refresh_frac",
                "jumps_min", "jumps_max", "turn_prob", "curriculum", "wide_neo", "seed", "compact_ice", "stall_ticks",
                "eval_per_level", "val_per_level", "eval_jumps"):
        assert getattr(tc, key) == ref[key], key
    assert dataclasses.asdict(tc.reward) == ref["reward"]
    d = G.GateConfig()
    assert (d.gamma, d.lam, d.num_envs, d.rollout, d.iterations, d.eval_every) == (0.995, 0.95, 8192, 64, 150, 10)
    slots = json.loads((ROOT / "configs" / "gate_experts.json").read_text())["slots"]
    assert [s["slot"] for s in slots] == ["final", "g995-ice", "neo", "g995-extra", "combined-distill", "skills-hh-ft",
                                          "neo-combined-r1"]
