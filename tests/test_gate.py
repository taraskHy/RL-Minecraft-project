"""Skill gate (parkour/gate.py): fixed slots play like their expert, decisions at starts and landings, the semi-MDP
returns, the PPO update's KL, the block-kind features, swaps, selection and the test rule, the eval command, a CPU
smoke run, the thermal guard and the watchdog. CPU only."""
import copy
import csv
import dataclasses
import json
import math
from fractions import Fraction

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


def course_hashes(sets):
    return [T.course_hash(c) for g in T.GROUP_ORDER if g in sets for p in ("standard", "obstacles")
            for c in sets[g].get(p) or []]


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
            won = 0
            for part, courses in small_sets[g].items():
                if not courses:
                    continue
                ref = T.evaluate(alone, CoursePool(courses, "cpu", C.GRID, compact_ice=True), "cpu", 200,
                                 per_course=True)["per_course"]
                got = ev["per_course"][g][part]
                assert got["success"] == ref["success"], (s["slot"], g, part)
                assert got["ticks"] == ref["ticks"], (s["slot"], g, part)
                won += sum(ref["success"])
            assert ev["counts"][g] == {"completed": won, "courses": sum(len(p) for p in small_sets[g].values())}
            completed[s["slot"]] += won
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


def test_slot_usage_counts_each_courses_first_episode(three_slots, small_sets):
    """validate()'s per-course decisions and per-group slot usage against an independent count over play()'s trace,
    for a gate whose argmax switches experts: a course's decisions are its first episode's start and the landings
    before its end, its slot ticks the first episode's ticks (nothing of the episodes after it)."""
    experts = G.Experts(three_slots, "cpu")
    torch.manual_seed(0)
    gate = G.Gate(len(three_slots), (16,))
    with torch.no_grad():
        for p in gate.actor.parameters():  # large weights: the argmax follows the state (all three slots play)
            p.normal_(0.0, 1.0)
    sets = {"basic": {"standard": small_sets["basic"]["standard"], "obstacles": small_sets["basic"]["obstacles"]},
            "extra": {"standard": small_sets["extra"]["standard"], "obstacles": []}}
    ev = G.validate(gate, experts, sets, "cpu", stall_ticks=200)
    k = len(three_slots)
    for g in sets:
        dec, tic = [0] * k, [0] * k
        for part, courses in sets[g].items():
            if not courses:
                continue
            trace = []
            G.play(gate, experts, CoursePool(courses, "cpu", C.GRID, compact_ice=True), "cpu", 200, trace=trace)
            per_course = []
            for e in range(len(courses)):
                end = next(t for t, tr in enumerate(trace) if tr["done"][e])  # the first episode's last tick
                landings = sum(int(trace[t + 1]["cp"][e] > trace[t]["cp"][e]) for t in range(end))
                made = [t for t in range(end + 1) if trace[t]["decide"][e]]
                assert made[0] == 0 and len(made) == 1 + landings
                per_course.append(len(made))
                assert ev["per_course"][g][part]["ticks"][e] == end + 1
                for t in range(end + 1):
                    s = int(trace[t]["slot"][e])
                    tic[s] += 1
                    dec[s] += t in made
            assert ev["per_course"][g][part]["decisions"] == per_course
        u = ev["usage"][g]
        assert (u["n_decisions"], u["n_ticks"]) == (sum(dec), sum(tic))
        assert u["decisions"] == {s["slot"]: dec[i] / sum(dec) for i, s in enumerate(three_slots)}
        assert u["ticks"] == {s["slot"]: tic[i] / sum(tic) for i, s in enumerate(three_slots)}
        assert sum(x > 0 for x in dec) > 1 and sum(dec) > sum(len(p) for p in sets[g].values())  # it switches


@torch.no_grad()
def test_semi_mdp_transitions_and_advantages(three_slots, small_sets):
    """Each transition's reward is the plain sum of its ticks' rewards, its successor is discounted by
    gamma ** ticks, and GAE runs over decisions; checked against a per-environment recomputation from the trace,
    across two rollouts with the gate and its normaliser changed in between (as an update changes them). A decision
    still running at a rollout's end completes in the next one: it keeps its slot and its behaviour
    log-probability, and its input is normalised and valued by the gate of the rollout it completes in, like every
    other value of that rollout. A time limit bootstraps the gate's value of the final state on the episode's own
    course and cp, captured inside the environment just before it resets them."""
    experts = G.Experts(three_slots, "cpu")
    torch.manual_seed(5)
    gate = G.Gate(len(three_slots), (16,))
    courses = small_sets["basic"]["standard"][:4] + small_sets["extra"]["obstacles"][:4]
    pool = CoursePool(courses, "cpu", C.GRID, compact_ice=True)
    n, steps, gamma, lam = 8, 40, 0.995, 0.95
    env = ParkourEnv(pool, EnvConfig(num_envs=n, stall_ticks=30), sampler=None)
    feats = G.GateFeatures(pool)
    col = G.Collector(env, gate, experts, feats, steps, torch.Generator().manual_seed(1))
    ended, finals = [], []  # per tick: (course, cp) just before env.step resets the ended rows; the final observation
    real_reset, real_step = env._apply_reset, env.step

    def reset(mask, chosen):
        ended.append((env.course.clone(), env.cp.clone()))
        return real_reset(mask, chosen)

    def step(action):
        out = real_step(action)
        finals.append(out[4].clone())
        return out

    env._apply_reset, env.step = reset, step
    valued, real_value = [], col._value  # the collector's value calls: each tick's final states, then the last state

    def value(obs, course, cp):
        valued.append((obs.clone(), course.clone(), cp.clone()))
        return real_value(obs, course, cp)

    col._value = value
    trace, values, batches, lasts, carried_values, gates = [], [], [], [], [], []
    stale_gap = 0.0
    for r in range(2):
        lo = r * steps
        col.collect(trace)
        gates.append(copy.deepcopy(gate))
        values += [col.D_val[t + 1].clone() for t in range(steps)]  # the gate's value of every row at every tick
        assert len(valued) == (r + 1) * (steps + 1)
        for t in range(lo, lo + steps):  # time limits: the final state's value on the episode's own course and cp
            course0, cp0 = ended[t]
            obs_v, course_v, cp_v = valued[r * (steps + 1) + t - lo]
            assert torch.equal(obs_v, finals[t]) and torch.equal(course_v, course0) and torch.equal(cp_v, cp0)
            want = gate.value(gate.norm(G.gate_input(finals[t], feats, course0, cp0)))
            tl = trace[t]["trunc"]
            assert torch.allclose(trace[t]["boot"][tl], want[tl])
        cv = {}
        for e in col.D_valid[0].nonzero().flatten().tolist():  # decisions carried from the previous rollout
            s = max(t for t in range(lo) if trace[t]["decide"][e])
            x = trace[s]["x"][e:e + 1]
            xn = gate.norm(x)
            assert torch.allclose(col.D_obs[0, e], xn[0]) and torch.allclose(col.D_val[0, e], gate.value(xn)[0])
            old = gates[r - 1]  # the behaviour policy: the gate of the rollout that decided
            slot = int(trace[s]["slot"][e])
            assert int(col.D_slot[0, e]) == slot
            assert torch.allclose(col.D_logp[0, e], torch.log_softmax(old.logits(old.norm(x)), -1)[0, slot])
            cv[e] = float(gate.value(xn)[0])
            stale_gap = max(stale_gap, abs(cv[e] - float(old.value(old.norm(x))[0])))
        carried_values.append(cv)
        batches.append(col.batch(gamma, lam))
        lasts.append(col.last_val.clone())
        for p in gate.parameters():  # an update and the normaliser's refresh change the gate before the next rollout
            p.add_(0.3 * torch.randn_like(p))
        G.masked_update(gate.norm, 2.0 * torch.randn(64, G.GATE_OBS_DIM) + 0.5, torch.ones(64, dtype=torch.bool))
    assert carried_values[1] and not carried_values[0]
    assert stale_gap > 1e-3  # the stored (older) values would have differed
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
                v_start = float(values[s][e]) if s >= lo else carried_values[r][e]  # carried: the current critic
                adv_at[s] = R + gamma ** k * v_next - v_start + gamma ** k * lam * a_next
                want[(0 if s < lo else s - lo + 1, e)] = (adv_at[s], k, d)
                seen["done"] += d
                seen["time limit"] += bool(trunc[end, e])
                seen["landing"] += not d
                seen["carried"] += s < lo
        assert set(got) == set(want), r
        for key in want:
            assert got[key][1:] == want[key][1:], key
            assert got[key][0] == pytest.approx(want[key][0], rel=1e-5, abs=1e-4), key
        assert b["carried"] == sum(1 for (j, _) in want if j == 0)
    assert all(seen.values()), seen  # every kind of transition end was exercised


def test_ppo_kl_is_the_change_since_the_update_began():
    """A quarter of the decisions carry an older policy's log-probability (decisions carried from an earlier
    rollout). Against those the first minibatch would already show KL 0.0125, inside parkour.train's dead band
    (0.005-0.02) where a learning rate at its minimum never rises; measured from the update's start it is 0, and
    the rate rises."""
    torch.manual_seed(0)
    gate = G.Gate(4, (16,))
    n = 4000
    obs = torch.randn(n, G.GATE_OBS_DIM)
    with torch.no_grad():
        logits = gate.logits(obs)
        act = torch.distributions.Categorical(logits=logits).sample()
        logp = torch.log_softmax(logits, -1).gather(1, act[:, None]).squeeze(1)
    carried = torch.arange(n) % 4 == 0
    b = {"obs": obs, "act": act, "logp": torch.where(carried, logp - 0.3, logp), "val": torch.zeros(n),
         "adv": torch.randn(n), "ret": torch.zeros(n)}
    floor = 0.25 * (math.exp(0.3) - 1 - 0.3)
    cfg = G.GateConfig(device="cpu")
    assert 0.5 * cfg.target_kl < floor < 2 * cfg.target_kl
    opt = torch.optim.Adam(gate.parameters(), lr=cfg.lr_min, eps=1e-5)
    stats, lr = G.ppo_update(gate, opt, b, cfg, cfg.lr_min)
    assert abs(stats["kl_first"]) < 1e-6
    assert stats["kl_behaviour"] > 0.5 * cfg.target_kl  # the surrogate's ratio stays the behaviour policy's
    assert lr > 10 * cfg.lr_min and stats["minibatch_updates"] > 0


def hand_course(ceiling: int = 1) -> C.Course:
    """Platforms: 0 start, 1 stone (made ice below), 2 ice run-up, 3 its target, 4 bottom slab, 5 its exit,
    6 head-hitter runway (a roof over it; ``ceiling`` 2 also over the gap), 7 its target, 8 slime pad, 9 its exit,
    10 the top of a ladder climb."""
    c = C.from_specs([(S(1, 0, 0, (2, 2)), 0), (S(1, 0, 0, (1, 1), runway=3, ice=1), 0),
                      (S(1, -1, 0, (1, 1), block="bottom_slab"), 0), (S(1, 0, 0, (1, 1), runway=3, ceiling=ceiling), 0),
                      (S(1, -2, 0, (1, 1), bounce=2), 0), (S(1, 2, 0, (1, 1), ladder=1), 0)])
    c.ice = sorted(set(c.ice) | {1})
    return c


def spec_kinds(c: C.Course):
    """Independent of platform_tags: on a generated course c.specs[k - 1] is the jump that lands on platform k (the
    runway hop, the jump itself, the exit after a shape or a pad), so platform k is slime, shaped or the top of a
    ladder by that spec and ice when k is in c.ice; a head hitter's runway k has the roof (c.specs[k].ceiling)."""
    kinds, roofs = [], []
    for k in range(c.n):
        tags = {"ice"} if k in {int(i) for i in c.ice} else set()
        if k:
            sp = c.specs[k - 1]
            tags |= {t for t, on in (("slime", sp.bounce), ("shaped", sp.block), ("ladder", sp.ladder)) if on}
        assert len(tags) <= 1, (k, tags)
        kinds.append(tags.pop() if tags else "stone")
        roofs.append(bool(k + 1 < c.n and c.specs[k].ceiling > 0))
    return kinds, roofs


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
    # A ladder climb in each of the four headings (ladder facings 3, 0, 1, 2): the top of the climb is the ladder
    # platform, never the takeoff below it.
    steps = [(S(1, 0, 0, (2, 2)), 0)]
    for turn in (0, 1, 1, 1):
        steps += [(S(1, 2, 0, (1, 1), ladder=1), turn), (S(1, 0, 0, (2, 2)), 0)]
    lc = C.from_specs(steps, start=(20, 4, 20), start_size=2)
    assert sorted({int(v) for v in np.asarray(lc.ladders).reshape(-1, 4)[:, 3]}) == [0, 1, 2, 3]
    assert [G.KINDS[k] for k in G.platform_kinds(lc)[0]] == ["stone", "stone"] + ["ladder", "stone"] * 4
    # Generated courses: beams, pillars and neo walls never count as a ceiling.
    rng = np.random.default_rng([2026, 99, 0])
    adv = C.load_levels(advanced=True)
    walls = 0
    for _ in range(40):
        oc = C.generate(rng, adv, int(rng.integers(len(adv))), 10, obstacle_prob=1.0)
        assert not G.platform_kinds(oc)[1].any()
        walls += len(oc.extra) > 0
    assert walls > 30
    # Skills-mode training courses: every platform's kind and roof as the courses' own jump specs say.
    cfg = T.TrainConfig(seed=0, skills="basic,advanced,extra", ice_prob=0.3, obstacle_prob=0.3)
    made = T.CourseFactory(cfg, C.load_levels(), set()).make(150)
    seen = set()
    for mc in made:
        assert len(mc.specs) == mc.n - 1
        kind, roof = G.platform_kinds(mc)
        want_kind, want_roof = spec_kinds(mc)
        assert [G.KINDS[k] for k in kind] == want_kind
        assert roof.tolist() == want_roof
        seen |= set(want_kind) | ({"roof"} if any(want_roof) else set())
    assert seen == set(G.KINDS) | {"roof"}


def test_features_follow_the_pool_when_it_replaces_a_course():
    """After CoursePool.replace (the training pool's refresh) the replaced course's features are the new course's,
    without GateFeatures being told; the other courses keep theirs."""
    pool = CoursePool([hand_course(1), hand_course(2)], "cpu", C.GRID, compact_ice=True)
    f = G.GateFeatures(pool)
    plain = C.from_specs([(S(1, 0, 0, (2, 2)), 0), (S(1, 0, 0, (1, 1), runway=3, ceiling=1), 0)])
    assert f(torch.tensor([0]), torch.tensor([0]))[0, 5:10].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]  # next: ice
    pool.replace(0, plain)
    kind, roof = G.platform_kinds(plain)
    other, other_roof = G.platform_kinds(pool.courses[1])
    for cp in range(plain.n):
        x = f(torch.tensor([0, 1]), torch.tensor([cp, cp]))
        nxt = cp + 1 < plain.n
        assert x[0].tolist() == ([float(i == kind[cp]) for i in range(5)]
                                 + ([float(i == kind[cp + 1]) for i in range(5)] if nxt else [0.0] * 5)
                                 + [float(nxt and roof[cp + 1])])
        assert x[1, :5].tolist() == [float(i == other[cp]) for i in range(5)]
    assert roof.tolist() == [False, False, True, False]


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
    # A gate checkpoint with the old slots and its stored validation; a fine-tune with the new expert in slot z.
    cfg = G.GateConfig(run="ft", device="cpu", experts=str(tmp_path / "new.json"), num_envs=4, rollout=4, iterations=1,
                       eval_every=1, val_per_level=1, val_stall_ticks=40, eval_per_level=1, pool_size=8,
                       pool_refresh_every=0, stall_ticks=40, hidden=(8,), save_at="0")
    torch.manual_seed(0)
    gate = G.Gate(3, (8,))
    ck_path = tmp_path / "gate.pt"
    start_eval = {"success": 0.0, "groups": {g: 0.0 for g in T.GROUP_ORDER},
                  "counts": {g: {"completed": 0, "courses": 9} for g in T.GROUP_ORDER}}
    torch.save(G.checkpoint(gate, old, 9, {"validation": G.validation_spec(cfg)}, ev=start_eval), ck_path)
    cfg = dataclasses.replace(cfg, init_from=str(ck_path))
    with pytest.raises(ValueError, match="--swap"):
        G.train(cfg)
    assert not (tmp_path / "runs" / "ft").exists()
    out = G.train(dataclasses.replace(cfg, swap=True))
    config = json.loads((out / "config.json").read_text())
    assert config["swaps"] == G.compare_slots(old, new, swap=True)
    assert config["slots"] == new and config["init_slots"] == old
    assert (config["init_validation"]["iteration"], config["init_validation"]["success"]) == (9, 0.0)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["init_validation"] == config["init_validation"]
    assert summary["keep"]["keep"] is True  # the 20:20 rule: at least the start gate's mean (0) on the same sets
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
    # A checkpoint overwritten in place (the same path, another network) is another expert: a stored slot list no
    # longer matches, the same file list needs --swap and is logged by sha1, and a pinned sha1 is refused.
    tiny_expert(c, 99)
    with pytest.raises(ValueError, match="changed"):
        G.current_slots(old)
    same = G.read_slots(experts_file(tmp_path / "same.json", [("x", "basic", a), ("y", "ice", b), ("z", "all", c)]))
    with pytest.raises(ValueError, match="--swap"):
        G.compare_slots(old, same, swap=False)
    moved = G.compare_slots(old, same, swap=True)
    assert len(moved) == 1 and moved[0]["slot"] == "z" and moved[0]["old_checkpoint"] == moved[0]["new_checkpoint"]
    assert moved[0]["old_sha1"] == old[2]["sha1"] != moved[0]["new_sha1"]
    pinned = tmp_path / "pinned.json"
    pinned.write_text(json.dumps({"slots": [{"slot": "z", "skill": "all", "checkpoint": str(c),
                                             "sha1": old[2]["sha1"]}]}))
    with pytest.raises(ValueError, match="sha1"):
        G.read_slots(pinned)
    with pytest.raises(ValueError, match="changed"):
        G.evaluate_cmd(["--checkpoint", str(ck_path), "--device", "cpu", "--skills", "basic", "--per-level", "1"])


def test_keep_rule_of_a_fine_tune_after_a_swap():
    spec = G.validation_spec(G.GateConfig())
    def counts(b, i):
        return {"basic": {"completed": b, "courses": 12}, "ice": {"completed": i, "courses": 6}}

    x = T.macro_score({"basic": 10 / 12, "ice": 5 / 6})
    init = {"iteration": 9, "success": x, "counts": counts(10, 5), "validation": spec}
    tie = {"validation_success": math.nextafter(x, 0.0), "validation_counts": counts(8, 6)}  # the same mean exactly
    assert G.keep_rule(tie, init, spec)["keep"] is True
    lower = {"validation_success": x - 0.01, "validation_counts": counts(8, 5)}
    assert G.keep_rule(lower, init, spec)["keep"] is False
    assert G.keep_rule(tie, init, dict(spec, stall_ticks=100))["keep"] is None  # validated on other rules
    assert G.keep_rule(tie, dict(init, success=None), spec)["keep"] is None
    assert G.keep_rule(tie, None, spec) is None  # not a fine-tune


def test_cpu_smoke_run_writes_every_file_and_reloads(tmp_path, monkeypatch):
    """Two updates with half the pool refreshed after each (options cut, courses replaced), every gate input's 11
    features checked against the course its row really plays, a validation after each update (the measurement's
    groups whatever --skills trains on, at its own stall), and best.pt reloaded."""
    monkeypatch.setattr(G, "RUNS", tmp_path / "runs")
    real_input, known, first, refreshed = G.gate_input, {}, [], [0]

    def checked_input(obs, features, course, cp):
        out = real_input(obs, features, course, cp)
        pool = features.pool
        if pool.size == 16 and not first:  # the training pool (the validation batches here hold 3 or 6 courses)
            first.extend(pool.courses)  # its courses before any refresh
        for i, (ci, k) in enumerate(zip(course.tolist(), cp.tolist())):
            c = pool.courses[ci]
            if id(c) not in known:  # (keeps every course alive, so ids stay unique)
                known[id(c)] = (c, *G.platform_kinds(c))
            _, kind, roof = known[id(c)]
            nxt = k + 1 < c.n
            want = ([float(j == kind[k]) for j in range(5)]
                    + ([float(j == kind[k + 1]) for j in range(5)] if nxt else [0.0] * 5)
                    + [float(nxt and roof[k + 1])])
            assert out[i, G.OBS_DIM:].tolist() == want, (ci, k)
            refreshed[0] += pool.size == 16 and all(c is not f for f in first)
        return out

    monkeypatch.setattr(G, "gate_input", checked_input)
    cfg_file = experts_file(tmp_path / "experts.json", [("final", "basic", FINAL),
                                                        ("tiny", "all", tiny_expert(tmp_path / "tiny.pt", 1))])
    cfg = G.GateConfig(run="smoke", device="cpu", experts=str(cfg_file), num_envs=16, rollout=24, iterations=2,
                       eval_every=1, val_per_level=1, val_stall_ticks=50, eval_per_level=1, pool_size=16,
                       pool_refresh_every=1, pool_refresh_frac=0.5, stall_ticks=60, skills="basic,extra",
                       hidden=(16,), save_at="0,2")
    out = G.train(cfg)
    assert refreshed[0] > 0  # rows on courses that replaced others in the pool were checked
    for f in ("config.json", "train_log.csv", "eval_val_log.csv", "summary.json", "checkpoints/best.pt",
              "checkpoints/latest.pt", "checkpoints/updates_00000.pt", "checkpoints/updates_00002.pt"):
        assert (out / f).is_file(), f
    rows = list(csv.DictReader((out / "train_log.csv").open()))
    assert [int(r["iteration"]) for r in rows] == [0, 1]
    for r in rows:
        assert int(r["transitions"]) > 0 and int(r["minibatch_updates"]) > 0
        for key in ("pi_loss", "v_loss", "entropy", "kl", "kl_behaviour"):
            assert math.isfinite(float(r[key])), key
        assert abs(float(r["kl_first"])) < 1e-6  # the first minibatch, before any step: no change yet
    vals = list(csv.DictReader((out / "eval_val_log.csv").open()))
    assert len(vals) == 2 and all(0.0 <= float(v["success"]) <= 1.0 for v in vals)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["completed"] and summary["iteration"] == 1 and "test_eval" not in summary
    spec = summary["validation"]
    assert (spec["per_level"], spec["stall_ticks"], spec["groups"], spec["courses"], spec["as_measurement"]) == \
        (1, 50, ["basic", "advanced", "extra"], 36, False)
    exact = [sum(Fraction(int(v[f"completed_{g}"]), int(v[f"courses_{g}"])) for g in T.GROUP_ORDER) / 4 for v in vals]
    best_it = exact.index(max(exact))  # ties to the earlier
    assert summary["best"]["iteration"] == best_it
    assert set(summary["best"]["validation_groups"]) == set(T.GROUP_ORDER)  # advanced too, untrained here
    assert set(summary["last_eval"]["slot_usage"]["basic"]["decisions"]) == {"final", "tiny"}
    gate, ck = G.load_gate(out / "checkpoints" / "best.pt", "cpu")
    assert ck["slots"] == G.read_slots(cfg_file) and ck["iteration"] == best_it
    sets = T.skill_sets("validation", 1, 10, ("basic", "advanced", "extra"), ice=True)
    ev = G.validate(gate, G.Experts(ck["slots"], "cpu"), sets, "cpu", stall_ticks=50)
    assert ev["success"] == pytest.approx(summary["best"]["validation_success"])
    assert ev["counts"] == summary["best"]["validation_counts"]
    with pytest.raises(ValueError, match="exists"):
        G.train(cfg)


def scripted_eval(counts: dict, success: float) -> dict:
    return {"success": success, "groups": {g: c / n for g, (c, n) in counts.items()},
            "counts": {g: {"completed": c, "courses": n} for g, (c, n) in counts.items()}, "usage": {},
            "per_level": {}, "per_course": {}, "courses": sum(n for _, n in counts.values()), "seconds": 0.0}


def test_selection_is_exact_and_the_test_is_played_once_with_best(tmp_path, monkeypatch):
    """best.pt: the highest four-group validation mean compared exactly, so two equal means whose floats differ in
    the last bit are a tie that keeps the earlier; --test-once plays the test sets once, after training, with
    best.pt's gate, and records best.test_success (as parkour.train) next to test_eval."""
    monkeypatch.setattr(G, "RUNS", tmp_path / "runs")
    efile = experts_file(tmp_path / "experts.json", [("a", "basic", tiny_expert(tmp_path / "a.pt", 0)),
                                                     ("b", "all", tiny_expert(tmp_path / "b.pt", 1))])
    first = {"basic": (10, 12), "ice": (5, 6), "advanced": (3, 6), "extra": (8, 12)}
    same = {"basic": (8, 12), "ice": (6, 6), "advanced": (3, 6), "extra": (8, 12)}  # two basic wins moved to one ice
    lower = {"basic": (8, 12), "ice": (5, 6), "advanced": (3, 6), "extra": (8, 12)}
    x = T.macro_score({g: c / n for g, (c, n) in first.items()})
    script = [scripted_eval(first, x), scripted_eval(same, math.nextafter(x, 1.0)), scripted_eval(lower, x - 0.05),
              scripted_eval(first, 0.123)]
    calls = []

    def scripted(gate, experts, sets, device, batch_size=600, stall_ticks=200, fixed_slot=None):
        calls.append({"sets": course_hashes(sets), "stall": stall_ticks,
                      "gate": {k: v.clone() for k, v in gate.state_dict().items()}})
        return script[len(calls) - 1]

    monkeypatch.setattr(G, "validate", scripted)
    cfg = G.GateConfig(run="sel", device="cpu", experts=str(efile), num_envs=16, rollout=24, iterations=3,
                       eval_every=1, val_per_level=1, eval_per_level=1, pool_size=16, pool_refresh_every=0,
                       stall_ticks=40, hidden=(8,), test_once=True)
    out = G.train(cfg)
    summary = json.loads((out / "summary.json").read_text())
    best = torch.load(out / "checkpoints" / "best.pt", weights_only=False)
    assert summary["best"]["iteration"] == 0 and best["iteration"] == 0
    assert len(calls) == 4  # three validations, then the test once
    val = course_hashes(T.skill_sets("validation", 1, 10, G.VALIDATION_GROUPS, ice=True))
    test = course_hashes(T.skill_sets("test", 1, 10, G.VALIDATION_GROUPS, ice=True))
    assert [c["sets"] for c in calls] == [val, val, val, test]
    assert all(c["stall"] == 200 for c in calls)  # the validation's stall, not the training environments' 40
    assert all(torch.equal(calls[3]["gate"][k], v) for k, v in best["gate"].items())  # best.pt's gate
    assert not all(torch.equal(calls[2]["gate"][k], v) for k, v in best["gate"].items())  # not the last one
    assert summary["test_eval"]["iteration"] == 0 and summary["best"]["test_success"] == 0.123
    assert summary["best"]["test_groups"] == script[3]["groups"]
    assert json.loads((out / "test_eval.json").read_text())["success"] == 0.123


def test_eval_command_writes_its_file_and_plays_a_runs_test_once(tmp_path):
    a, b = tiny_expert(tmp_path / "a.pt", 0), tiny_expert(tmp_path / "b.pt", 1)
    slots = G.read_slots(experts_file(tmp_path / "experts.json", [("x", "basic", a), ("y", "all", b)]))
    run = tmp_path / "runs" / "r"
    (run / "checkpoints").mkdir(parents=True)
    (run / "config.json").write_text("{}")
    torch.manual_seed(0)
    ck = run / "checkpoints" / "best.pt"
    torch.save(G.checkpoint(G.Gate(2, (8,)), slots, 4, {}), ck)
    (run / "summary.json").write_text(json.dumps({"best": {"iteration": 4, "validation_success": 0.5}}))
    common = ["--checkpoint", str(ck), "--device", "cpu", "--skills", "basic", "--per-level", "1",
              "--stall-ticks", "30"]
    ev = G.evaluate_cmd(common)  # validation by default, written next to the checkpoint
    assert json.loads((run / "eval_validation_it4.json").read_text())["success"] == ev["success"]
    assert ev["courses"] == 18 and set(ev["counts"]) == {"basic", "ice"}
    with pytest.raises(SystemExit, match="exists"):
        G.evaluate_cmd(common)
    assert G.evaluate_cmd(common + ["--force"])["success"] == ev["success"]
    # The gate's test sets: once per run, recorded in the run as --test-once records it.
    t = G.evaluate_cmd(common + ["--split", "test"])
    summary = json.loads((run / "summary.json").read_text())
    assert summary["test_eval"]["success"] == t["success"] and summary["best"]["test_success"] == t["success"]
    assert json.loads((run / "test_eval.json").read_text())["split"] == "test"
    for extra in ([], ["--out", str(tmp_path / "elsewhere.json")]):
        with pytest.raises(SystemExit, match="tested"):
            G.evaluate_cmd(common + ["--split", "test"] + extra)
    assert not (tmp_path / "elsewhere.json").exists()
    # One expert alone on the test sets is not the gate's test; swapped experts on them need --force.
    G.evaluate_cmd(common + ["--split", "test", "--fixed-slot", "y"])
    assert (run / "eval_test_it4_fixed_y.json").is_file()
    other = G.read_slots(experts_file(tmp_path / "other.json", [("x", "basic", a), ("y", "all", a)]))
    assert other[1]["sha1"] != slots[1]["sha1"]
    fresh = tmp_path / "runs" / "s"
    (fresh / "checkpoints").mkdir(parents=True)
    (fresh / "config.json").write_text("{}")
    torch.save(G.checkpoint(G.Gate(2, (8,)), slots, 4, {}), fresh / "checkpoints" / "best.pt")
    swapped = ["--checkpoint", str(fresh / "checkpoints" / "best.pt"), "--experts", str(tmp_path / "other.json"),
               "--swap", "--device", "cpu", "--skills", "basic", "--per-level", "1", "--stall-ticks", "30"]
    with pytest.raises(SystemExit, match="swapped"):
        G.evaluate_cmd(swapped + ["--split", "test"])
    G.evaluate_cmd(swapped)
    assert (fresh / "eval_validation_it4_swapped.json").is_file() and not G.tested(fresh)


def test_eval_one_expert_alone_from_another_folder(tmp_path, monkeypatch):
    """`python -m parkour.gate eval --fixed-slot NAME` plays the default experts file's slot from any folder (the file
    is found relative to the repository root, like its checkpoints) and, with no run folder, writes no file."""
    if not (ROOT / "runs" / "combined-distill" / "checkpoints" / "best.pt").exists():
        pytest.skip("the experts' checkpoints are on the training PC")
    monkeypatch.chdir(tmp_path)
    ev = G.evaluate_cmd(["--fixed-slot", "final", "--device", "cpu", "--skills", "basic", "--per-level", "1",
                         "--stall-ticks", "60"])
    assert ev["courses"] == 18 and [s["slot"] for s in ev["slots"]] == [
        "final", "g995-ice", "neo", "g995-extra", "combined-distill", "skills-hh-ft", "neo-combined-r1"]
    assert all(u["decisions"]["final"] == 1.0 for u in ev["usage"].values())
    assert list(tmp_path.iterdir()) == []


def test_thermal_guard_waits_while_the_gpu_is_hot():
    steps = iter([70.0, 66.0, 60.0, 59.0, 57.0, 60.0, 66.0, None])
    hold = G.ThermalGuard(hot=65, cool=58, read=lambda: next(steps))
    flags = []
    for _ in range(8):
        hold.sample()
        flags.append(hold.heat.is_set())
    assert flags == [True, True, True, True, False, False, True, True]  # on above 65, off below 58; None: no reading
    readings = [66.0, 60.0, 59.0, 57.0]  # once the work waits
    seen = []
    guard = None

    def read():
        t = 70.0 if guard is None or guard.pauses == 0 else (readings.pop(0) if readings else 40.0)
        seen.append(t)
        return t

    guard = G.ThermalGuard(hot=65, cool=58, every=0.01, read=read).start()
    try:
        assert guard.heat.is_set()  # 70 C when it starts
        guard.wait()  # 66, 60 and 59 keep it waiting (the hysteresis); 57 releases it
        assert 57.0 in seen and guard.pauses == 1 and guard.waited > 0
        guard.wait()  # cool: returns at once
        assert guard.pauses == 1
    finally:
        guard.stop()
    with G.thermal_guard("cpu", 65, 58) as g:  # nothing on the CPU
        assert g is None and G._GUARD is None


def test_the_watchdog_pauses_the_gate():
    from scripts import gpu_watchdog as W
    assert W.pausable("python -m parkour.gate train --run gate-r1")
    assert W.pausable("python -m parkour.gate eval --checkpoint runs/gate-r1/checkpoints/best.pt")
    assert not W.pausable("python scripts/keep_awake.py --pattern parkour.gate")


def test_defaults_are_combined_distills_training_courses_and_the_measurements_validation():
    """The training courses of skills mode with runs/combined-distill/config.json's settings (on the training PC)."""
    d = G.GateConfig()
    assert G.validation_spec(d) == dict(G.MEASUREMENT, as_measurement=True)
    assert (d.gamma, d.lam, d.num_envs, d.rollout, d.iterations, d.eval_every) == (0.995, 0.95, 8192, 64, 150, 10)
    path = ROOT / "runs" / "combined-distill" / "config.json"
    if not path.exists():
        pytest.skip("runs/combined-distill is on the training PC")
    ref = json.loads(path.read_text())
    tc = G.course_config(d)
    for key in ("skills", "ice_prob", "obstacle_prob", "pool_size", "pool_refresh_every", "pool_refresh_frac",
                "jumps_min", "jumps_max", "turn_prob", "curriculum", "wide_neo", "seed", "compact_ice", "stall_ticks",
                "eval_per_level", "val_per_level", "eval_jumps"):
        assert getattr(tc, key) == ref[key], key
    assert dataclasses.asdict(tc.reward) == ref["reward"]
    slots = json.loads((ROOT / "configs" / "gate_experts.json").read_text())["slots"]
    assert [s["slot"] for s in slots] == ["final", "g995-ice", "neo", "g995-extra", "combined-distill", "skills-hh-ft",
                                          "neo-combined-r1"]
