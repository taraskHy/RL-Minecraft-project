import copy
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest

from minecraft_rl.resources import (
    ACTION_NAMES, N_ACTIONS, ResourceAction as A, ResourceAgent,
    ResourceSimConfig, ResourceSimulator, allowed_actions, potential,
    reward_transition, state_from_observation, train_simulator,
)
from minecraft_rl.resources import (
    CONDITIONS, MASKED, STATE_VERSIONS, UNMASKED, WOOD_STAGE, StageSpec, encode_state,
)


def observation(**changes):
    result = {
        "inventory_counts": {"logs": 0, "planks": 0, "sticks": 0,
                             "tables": 0, "pickaxes": 0},
        "nearby_log": True, "nearby_table": False, "table_placed": False,
        "health": 20, "food": 20, "position": {"x": 0, "y": 64, "z": 0},
    }
    for key, value in changes.items():
        if key in result["inventory_counts"]:
            result["inventory_counts"][key] = value
        else:
            result[key] = value
    return result


S = state_from_observation(observation())
T = state_from_observation(observation(logs=1, nearby_log=False))


class ResourceAgentTests(unittest.TestCase):
    def test_bounded_state_and_missing_values_are_not_invented(self):
        state = state_from_observation(observation(logs=100, planks=100, sticks=100,
                                                    tables=10, pickaxes=3))
        self.assertEqual(state[:5], (3, 9, 2, 1, 1))
        missing = observation()
        del missing["inventory_counts"]["logs"]
        with self.assertRaises(ValueError):
            state_from_observation(missing)

    def test_observation_finite_counts_flags_and_health(self):
        invalid = [observation(logs=True), observation(planks=-1),
                   observation(nearby_table=1), observation(health=float("nan")),
                   observation(food=float("inf")), observation(health=21),
                   observation(position={"x": 0, "y": True, "z": 0})]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                state_from_observation(value)

    def test_mask_keeps_competing_craft_orders_and_limits_collection(self):
        choices = allowed_actions(state_from_observation(observation(logs=1, planks=4)))
        self.assertEqual(set(choices), {A.CHOP, A.CRAFT_PLANKS, A.CRAFT_STICKS, A.CRAFT_TABLE})
        enough_wood = state_from_observation(observation(logs=3, nearby_log=False))
        self.assertEqual(allowed_actions(enough_wood), (A.CRAFT_PLANKS,))
        ready = state_from_observation(observation(planks=3, sticks=4,
                                                  table_placed=True, nearby_table=True))
        self.assertEqual(allowed_actions(ready), (A.CRAFT_PICKAXE,))

    def test_q_learning_ignores_infeasible_high_values(self):
        agent = ResourceAgent(alpha=0.5, gamma=0.9)
        agent.q[T] = [2.0, 100.0, 4.0] + [200.0] * (N_ACTIONS - 3)
        error = agent.update(S, A.CHOP, 1, T, False)
        self.assertAlmostEqual(error, 4.6)
        self.assertAlmostEqual(agent.q[S][A.CHOP], 2.3)

    def test_sarsa_uses_actual_next_feasible_action(self):
        agent = ResourceAgent("sarsa", alpha=0.5, gamma=0.9)
        agent.q[T] = [2.0, 100.0, 4.0] + [200.0] * (N_ACTIONS - 3)
        agent.update(S, A.CHOP, 1, T, False, A.SEARCH)
        self.assertAlmostEqual(agent.q[S][A.CHOP], 1.4)
        with self.assertRaises(ValueError):
            agent.update(S, A.CHOP, 1, T, False, A.CRAFT_PICKAXE)

    def test_terminal_zero_but_time_limit_bootstraps(self):
        for algorithm in ("q_learning", "sarsa"):
            with self.subTest(algorithm=algorithm):
                agent = ResourceAgent(algorithm, alpha=1, gamma=0.9)
                agent.q[T] = [10.0] * N_ACTIONS
                agent.update(S, A.CHOP, 1, T, True)
                self.assertEqual(agent.q[S][A.CHOP], 1)
                # A truncated transition passes terminated=False, preserving bootstrap.
                agent.update(S, A.CHOP, 1, T, False, A.SEARCH)
                self.assertEqual(agent.q[S][A.CHOP], 10)

    def test_frozen_action_selection_never_inserts_rows(self):
        agent = ResourceAgent(seed=10)
        for epsilon in (0, 0.2, 1):
            self.assertIn(agent.act(T, epsilon), allowed_actions(T))
        self.assertEqual(agent.q, {})

    def test_checkpoint_roundtrip_and_reject_wrong_schema(self):
        agent = ResourceAgent("sarsa", alpha=0.3)
        agent.update(S, A.CHOP, 0.4, T, False, A.SEARCH)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resources.json"
            agent.save(path, {"source": "simulator"})
            self.assertEqual(ResourceAgent.load(path).q, agent.q)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["metadata"]["source"], "simulator")
            for field, bad in (("environment", "crafting_arena_v1"),
                               ("actions", list(reversed(ACTION_NAMES))),
                               ("state_caps", [999] * 9)):
                broken = copy.deepcopy(payload)
                broken[field] = bad
                path.write_text(json.dumps(broken))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    ResourceAgent.load(path)
            for bad in (True, float("nan"), float("inf")):
                broken = copy.deepcopy(payload)
                broken["q"][0]["values"][0] = bad
                path.write_text(json.dumps(broken))
                with self.assertRaises(ValueError):
                    ResourceAgent.load(path)

    def test_initial_q_sets_untried_values_and_survives_a_checkpoint(self):
        agent = ResourceAgent(initial_q=-1)
        self.assertEqual(agent.values(S), [-1.0] * N_ACTIONS)
        agent.update(S, A.CHOP, 0.5, T, False)
        # The new row starts from -1, and the target bootstraps on T's untried values, also -1.
        self.assertAlmostEqual(agent.q[S][A.CHOP], -1 + 0.2 * (0.5 + 0.97 * -1 + 1))
        self.assertEqual(agent.q[S][A.SEARCH], -1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pessimistic.json"
            agent.save(path)
            loaded = ResourceAgent.load(path)
            self.assertEqual((loaded.initial_q, loaded.values(T)), (-1.0, [-1.0] * N_ACTIONS))
            # A policy saved before initial_q was recorded started from 0.
            payload = json.loads(path.read_text())
            del payload["initial_q"]
            path.write_text(json.dumps(payload))
            self.assertEqual(ResourceAgent.load(path).initial_q, 0.0)
            for bad in (True, "low", 101, float("nan")):
                payload["initial_q"] = bad
                path.write_text(json.dumps(payload))
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    ResourceAgent.load(path)
        for bad in (True, float("nan"), float("inf"), -101, "0"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ResourceAgent(initial_q=bad)
        trained, _ = train_simulator(2, condition="unmasked", schedule="linear", initial_q=-0.5)
        self.assertEqual(trained.initial_q, -0.5)

    def test_reject_nonfinite_parameters_and_updates(self):
        for kwargs in ({"alpha": float("nan")}, {"gamma": float("inf")},
                       {"alpha": True}, {"gamma": 1}, {"seed": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ResourceAgent(**kwargs)
        agent = ResourceAgent()
        with self.assertRaises(ValueError):
            agent.act(S, float("nan"))
        with self.assertRaises(ValueError):
            agent.update(S, A.CHOP, float("nan"), T, False)
        with self.assertRaises(ValueError):
            agent.update(S, True, 1, T, False)


class ResourceSimulatorTests(unittest.TestCase):
    def test_all_recipes_consume_materials_and_table_must_be_made(self):
        env = ResourceSimulator(ResourceSimConfig(search_success=1, chop_success=1,
                                                  macro_success=1, max_steps=50))
        obs = env.reset(1)
        self.assertEqual(sum(obs["inventory_counts"].values()), 0)
        obs, _, terminal, _, info = env.step(A.CRAFT_PICKAXE)
        self.assertFalse(terminal)
        self.assertEqual(info["event"], "masked_action")
        for _ in range(3):
            if not obs["nearby_log"]:
                obs, *_ = env.step(A.SEARCH)
            obs, *_ = env.step(A.CHOP)
        self.assertEqual(obs["inventory_counts"]["logs"], 3)
        for _ in range(3):
            obs, *_ = env.step(A.CRAFT_PLANKS)
        self.assertEqual(obs["inventory_counts"]["planks"], 12)
        self.assertEqual(state_from_observation(obs)[1], 9)
        obs, *_ = env.step(A.CRAFT_STICKS)
        self.assertEqual(obs["inventory_counts"]["sticks"], 4)
        obs, *_ = env.step(A.CRAFT_TABLE)
        self.assertEqual(obs["inventory_counts"]["tables"], 1)
        obs, *_ = env.step(A.PLACE_TABLE)
        self.assertEqual(obs["inventory_counts"]["tables"], 0)
        self.assertTrue(obs["table_placed"])
        obs, _, terminated, truncated, info = env.step(A.CRAFT_PICKAXE)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["success"])
        self.assertEqual(obs["inventory_counts"], {
            "logs": 0, "planks": 3, "sticks": 2, "tables": 0, "pickaxes": 1,
        })
        with self.assertRaises(RuntimeError):
            env.step(A.SEARCH)

    def test_failed_actions_do_not_generate_inventory(self):
        env = ResourceSimulator(ResourceSimConfig(search_success=0, chop_success=0,
                                                  macro_success=0, max_steps=2))
        first = env.reset(1)
        obs, _, _, truncated, _ = env.step(A.CHOP)
        self.assertEqual(obs["inventory_counts"], first["inventory_counts"])
        self.assertFalse(truncated)
        _, _, terminated, truncated, _ = env.step(A.CHOP)
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_observe_returns_independent_inventory_copy(self):
        env = ResourceSimulator()
        observed = env.reset(1)
        observed["inventory_counts"]["pickaxes"] = 100
        observed["position"]["x"] = 999
        self.assertEqual(env.observe()["inventory_counts"]["pickaxes"], 0)
        self.assertEqual(env.observe()["position"]["x"], 0)

    def test_shaping_telescopes_and_terminal_potential_is_zero(self):
        before = observation(logs=1)
        middle = observation(planks=4)
        gamma = 0.9
        _, first = reward_transition(before, middle, gamma=gamma)
        _, second = reward_transition(middle, before, gamma=gamma)
        # A return to the same state cannot earn repeatable positive shaping.
        self.assertAlmostEqual(first["shaping_reward"] + gamma * second["shaping_reward"],
                               (gamma * gamma - 1) * potential(before))
        self.assertLess(first["shaping_reward"] + gamma * second["shaping_reward"], 0)
        reward, terminal = reward_transition(before, middle, gamma=gamma, success=True)
        self.assertAlmostEqual(terminal["shaping_reward"], -potential(before))
        self.assertAlmostEqual(reward, 9.99 - potential(before))
        # A timeout keeps the following potential, just like ordinary continuation.
        self.assertAlmostEqual(first["shaping_reward"], gamma * potential(middle) - potential(before))

    def test_seeded_training_reproducible_and_explicit_simulator_label(self):
        for algorithm in ("q_learning", "sarsa"):
            first, rows1 = train_simulator(30, seed=4, algorithm=algorithm)
            second, rows2 = train_simulator(30, seed=4, algorithm=algorithm)
            self.assertEqual(first.q, second.q)
            self.assertEqual(rows1, rows2)
            self.assertTrue(first.q)
            self.assertTrue(all(row["backend"] == "resource-macro-simulator" for row in rows1))
            self.assertTrue(any(row["success"] for row in rows1))

    def test_training_rejects_mismatched_discount_and_invalid_sizes(self):
        with self.assertRaises(ValueError):
            train_simulator(1, gamma=0.9, config=ResourceSimConfig(gamma=0.97))
        for value in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                train_simulator(value)
        for kwargs in ({"max_steps": True}, {"search_success": float("nan")},
                       {"gamma": 1}, {"reward_mode": "other"}):
            with self.assertRaises(ValueError):
                ResourceSimConfig(**kwargs)


ROOT = Path(__file__).resolve().parents[1]
STAGE1_POLICY = ROOT / "results" / "pc_resource_runs" / "87673b8" / "mx-unmasked-q-train" / "policy.json"
# SHA-256 of the text ResourceAgent.save wrote at 409560e, before stage specifications existed, for
# golden_agents() with GOLDEN_METADATA. Text is compared with "\n" line ends, so it holds on every platform.
GOLDEN_METADATA = {"source": "golden", "episode": 3}
GOLDEN_POLICY_SHA256 = {
    "masked-q_learning": "5b2cbd179f2d830473388e3d355cb0236a499a566ac6cfe45f5dd7edaf4f0ee5",
    "masked-sarsa": "4fa803c64303a3ddeac669ebc13a5e9228de141f1b83c4a7c920bc7c52b71e17",
    "unmasked-q_learning": "22bed71634776b2535ee2ebb9be51f7d3d7b9a8fb1809b74702393ab5370e6df",
    "unmasked-sarsa": "33625b1a9ebc0434811dc6ddb7d50654bb187fd51e974c7c297aea0d028e0f28",
    "unmasked_history-q_learning": "f06c6ff2fd2d39d068f29ced0d56cec20c2859c9637a0e70815547e7e022db35",
    "unmasked_history-sarsa": "73311458eff17774c8bf7423eacc59392f05b2ff0c59291b12ac5239fca569c1",
}
# The stage-1 checkpoint loaded and saved again with its own metadata, by the code at 409560e.
STAGE1_RESAVED_SHA256 = "4bfa894322463c854c580fc203cfdac9a34b9e87812ecbe9929b5f475bce837b"
WOOD_RESULT_FOLDERS = ("pc_resource_runs", "parallel_benchmark", "resource_smoke")


def golden_agents():
    """Deterministic wood agents for every condition and both learners; the Q values are arbitrary."""
    for name, (mask, version) in CONDITIONS.items():
        for algorithm in ("q_learning", "sarsa"):
            agent = ResourceAgent(algorithm, alpha=0.3, gamma=0.9, seed=5, action_mask=mask, state_version=version,
                                  initial_q=-0.5 if algorithm == "sarsa" else 0.0)
            for k in range(3):
                base = (k, 2 * k, min(2, k), k % 2, 0, 1, k % 2, 1, 2)
                agent.q[encode_state(base, version, bool(k % 2))] = [
                    round(0.1 * (k + 1) * (a - 3.5), 6) + 1 / 3 for a in range(N_ACTIONS)]
            yield f"{name}-{algorithm}", agent


def legacy_save(agent, path, metadata=None):
    """ResourceAgent.save exactly as it was at 409560e, before stage specifications."""
    fields, caps = STATE_VERSIONS[agent.state_version]
    payload = {
        "format_version": 1, "environment": agent.state_version,
        "algorithm": agent.algorithm, "alpha": agent.alpha, "gamma": agent.gamma,
        "initial_q": agent.initial_q,
        "actions": list(ACTION_NAMES), "state_fields": list(fields),
        "state_caps": list(caps), "action_mask": agent.action_mask,
        "condition": agent.condition, "metadata": metadata or {},
        "q": [{"state": list(s), "values": v} for s, v in sorted(agent.q.items())],
    }
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def text_sha256(path):
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()


class StageSpecificationTests(unittest.TestCase):
    def test_the_wood_stage_is_the_default_and_wraps_the_existing_tables(self):
        self.assertIs(ResourceAgent().stage, WOOD_STAGE)
        self.assertEqual((WOOD_STAGE.name, WOOD_STAGE.actions, WOOD_STAGE.n_actions), ("wood", ACTION_NAMES, N_ACTIONS))
        self.assertIs(WOOD_STAGE.state_versions, STATE_VERSIONS)
        self.assertIs(WOOD_STAGE.conditions, CONDITIONS)
        self.assertEqual(N_ACTIONS, 8)
        defaults = ResourceAgent()
        self.assertEqual((defaults.action_mask, defaults.state_version, defaults.condition),
                         (MASKED, "resource_v1", "masked"))
        self.assertEqual(ResourceAgent(action_mask=UNMASKED).condition, "unmasked")
        self.assertEqual(ResourceAgent(action_mask=UNMASKED, state_version="resource_v2").condition, "unmasked_history")
        self.assertEqual(ResourceAgent(action_mask=MASKED, state_version="resource_v1").condition, "masked")
        self.assertEqual(ResourceAgent().allowed(S), allowed_actions(S))

    def test_wood_policy_files_are_byte_identical_to_the_format_before_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            new, old = Path(directory) / "new.json", Path(directory) / "old.json"
            labels = []
            for label, agent in golden_agents():
                with self.subTest(policy=label):
                    agent.save(new, GOLDEN_METADATA)
                    legacy_save(agent, old, GOLDEN_METADATA)
                    self.assertEqual(new.read_bytes(), old.read_bytes())
                    self.assertEqual(text_sha256(new), GOLDEN_POLICY_SHA256[label])
                    self.assertNotIn("stage", json.loads(new.read_text(encoding="utf-8")))
                    loaded = ResourceAgent.load(new)
                    self.assertEqual((loaded.q, loaded.condition, loaded.initial_q),
                                     (agent.q, agent.condition, agent.initial_q))
                    labels.append(label)
            self.assertEqual(labels, list(GOLDEN_POLICY_SHA256))

    def test_copied_or_unpickled_wood_agents_still_save_the_format_before_stages(self):
        # copy.deepcopy and pickle give the agent a new StageSpec object; the file layout follows the stage's name.
        clones = {"deepcopy": copy.deepcopy, "pickle": lambda agent: pickle.loads(pickle.dumps(agent))}
        with tempfile.TemporaryDirectory() as directory:
            original, copied = Path(directory) / "original.json", Path(directory) / "copied.json"
            for label, agent in golden_agents():
                agent.save(original, GOLDEN_METADATA)
                for how, clone in clones.items():
                    with self.subTest(policy=label, clone=how):
                        twin = clone(agent)
                        self.assertIsNot(twin.stage, WOOD_STAGE)
                        twin.save(copied, GOLDEN_METADATA)
                        self.assertEqual(copied.read_bytes(), original.read_bytes())
                        self.assertEqual(text_sha256(copied), GOLDEN_POLICY_SHA256[label])
                        self.assertNotIn("stage", json.loads(copied.read_text(encoding="utf-8")))
                        self.assertEqual(ResourceAgent.load(copied).q, agent.q)
            # The retained stage-1 checkpoint, loaded and then copied, still saves the 409560e text (read only).
            checkpoint = ResourceAgent.load(STAGE1_POLICY)
            metadata = json.loads(STAGE1_POLICY.read_text(encoding="utf-8"))["metadata"]
            for how, clone in clones.items():
                with self.subTest(policy="stage1", clone=how):
                    clone(checkpoint).save(copied, metadata)
                    self.assertEqual(text_sha256(copied), STAGE1_RESAVED_SHA256)

    def test_the_retained_stage1_checkpoint_loads_unchanged(self):
        # Read only: the accepted stage-1 policy (design section 5); nothing under results/ is written.
        digest = "7d0fcce8653df8d4afc0867b321fd9670a8c64ab6ef03a876c64b9cd1c9e0349"
        self.assertEqual(hashlib.sha256(STAGE1_POLICY.read_bytes()).hexdigest(), digest)
        agent = ResourceAgent.load(STAGE1_POLICY)
        payload = json.loads(STAGE1_POLICY.read_text(encoding="utf-8"))
        self.assertNotIn("initial_q", payload)
        self.assertEqual((agent.condition, agent.algorithm, agent.state_version, agent.alpha, agent.gamma,
                          agent.initial_q), ("unmasked", "q_learning", "resource_v1", 0.2, 0.97, 0.0))
        self.assertEqual(len(agent.q), 180)
        self.assertEqual(agent.q, {tuple(entry["state"]): entry["values"] for entry in payload["q"]})
        with tempfile.TemporaryDirectory() as directory:
            new, old = Path(directory) / "new.json", Path(directory) / "old.json"
            agent.save(new, payload["metadata"])
            legacy_save(agent, old, payload["metadata"])
            self.assertEqual(new.read_bytes(), old.read_bytes())
            self.assertEqual(text_sha256(new), STAGE1_RESAVED_SHA256)
        self.assertEqual(hashlib.sha256(STAGE1_POLICY.read_bytes()).hexdigest(), digest)

    def test_every_retained_wood_policy_still_loads(self):
        paths = sorted(path for folder in WOOD_RESULT_FOLDERS for path in (ROOT / "results" / folder).rglob("*.json")
                       if path.name in ("policy.json", "initial_policy.json"))
        self.assertGreaterEqual(len(paths), 100)
        versions = set()
        for path in paths:
            with self.subTest(policy=str(path.relative_to(ROOT))):
                payload = json.loads(path.read_text(encoding="utf-8"))
                agent = ResourceAgent.load(path)
                versions.add((agent.state_version, agent.condition))
                self.assertEqual(agent.q, {tuple(entry["state"]): entry["values"] for entry in payload["q"]})
                self.assertEqual(agent.initial_q, payload.get("initial_q", 0.0))
        self.assertEqual(versions, {("resource_v1", "masked"), ("resource_v1", "unmasked"),
                                    ("resource_v2", "unmasked_history")})

    def test_a_policy_saved_by_another_stage_is_refused_with_a_clear_error(self):
        toy = StageSpec("toy", ("LEFT", "RIGHT"), {"toy_v1": (("x",), (1,))}, {"toy_free": (UNMASKED, "toy_v1")},
                        lambda state, mask, version: (0, 1), "toy_free")
        toy_agent = ResourceAgent(stage=toy)
        toy_agent.update((0,), 1, 1.0, (1,), True)
        wood_agent = ResourceAgent()
        wood_agent.update(S, A.CHOP, 0.4, T, False)
        with tempfile.TemporaryDirectory() as directory:
            toy_path, wood_path = Path(directory) / "toy.json", Path(directory) / "wood.json"
            toy_agent.save(toy_path)
            wood_agent.save(wood_path)
            self.assertEqual(json.loads(toy_path.read_text(encoding="utf-8"))["stage"], "toy")
            self.assertEqual(ResourceAgent.load(toy_path, stage=toy).q, toy_agent.q)
            with self.assertRaisesRegex(ValueError, "saved by the toy stage .*a wood-stage agent cannot load it"):
                ResourceAgent.load(toy_path)
            with self.assertRaisesRegex(ValueError, "saved by the wood stage .*a toy-stage agent cannot load it"):
                ResourceAgent.load(wood_path, stage=toy)
            payload = json.loads(wood_path.read_text(encoding="utf-8"))
            payload["stage"] = "stone"
            wood_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "saved by the stone stage"):
                ResourceAgent.load(wood_path)
            # Naming the wood stage explicitly is the same as leaving the key out.
            payload["stage"] = "wood"
            wood_path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(ResourceAgent.load(wood_path).q, wood_agent.q)
            for bad in ([], 7):
                payload["environment"] = bad
                wood_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(environment=bad), self.assertRaises(ValueError):
                    ResourceAgent.load(wood_path)

    def test_a_stage_can_refuse_policy_states_its_encoder_never_produces(self):
        # The wood stage checks caps only, as before stages.
        self.assertIsNone(WOOD_STAGE.consistent)
        toy = StageSpec("toy", ("LEFT", "RIGHT"), {"toy_v1": (("x", "y"), (1, 1))}, {"toy_free": (UNMASKED, "toy_v1")},
                        lambda state, mask, version: (0, 1), "toy_free", lambda state, version: state != (1, 0))
        agent = ResourceAgent(stage=toy)
        agent.q[(0, 0)] = [0.5, 0.0]
        agent.q[(1, 1)] = [0.0, 0.5]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.json"
            agent.save(path)
            self.assertEqual(ResourceAgent.load(path, stage=toy).q, agent.q)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["q"].append({"state": [1, 0], "values": [0.0, 0.0]})
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"the toy stage never produces the state \[1, 0\]"):
                ResourceAgent.load(path, stage=toy)


if __name__ == "__main__":
    unittest.main()
