import copy
import json
from pathlib import Path
import tempfile
import unittest

from minecraft_rl.resources import (
    ACTION_NAMES, N_ACTIONS, ResourceAction as A, ResourceAgent,
    ResourceSimConfig, ResourceSimulator, allowed_actions, potential,
    reward_transition, state_from_observation, train_simulator,
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


if __name__ == "__main__":
    unittest.main()
