"""The masked, unmasked, and unmasked_history conditions must differ only where declared."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from minecraft_rl.resources import (
    CONDITIONS, MASKED, N_ACTIONS, UNMASKED, ResourceAction as A, ResourceAgent, ResourceSimConfig,
    ResourceSimulator, actions_for, allowed_actions, condition_name, encode_state, state_from_observation,
    train_simulator,
)


def observation(**changes):
    result = {"inventory_counts": {"logs": 0, "planks": 0, "sticks": 0, "tables": 0, "pickaxes": 0},
              "nearby_log": True, "nearby_table": False, "table_placed": False,
              "health": 20, "food": 20, "position": {"x": 0, "y": 64, "z": 0}}
    for key, value in changes.items():
        if key in result["inventory_counts"]:
            result["inventory_counts"][key] = value
        else:
            result[key] = value
    return result


START = state_from_observation(observation(nearby_log=False))
NEAR_LOG = state_from_observation(observation())


class ConditionTests(unittest.TestCase):
    def test_three_named_conditions_fix_mask_and_state_version_together(self):
        self.assertEqual(CONDITIONS, {"masked": (MASKED, "resource_v1"), "unmasked": (UNMASKED, "resource_v1"),
                                      "unmasked_history": (UNMASKED, "resource_v2")})
        self.assertEqual(condition_name(MASKED, "resource_v1"), "masked")
        for bad in ((MASKED, "resource_v2"), ("other", "resource_v1"), (UNMASKED, "resource_v3")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                condition_name(*bad)
            with self.assertRaises(ValueError):
                ResourceAgent(action_mask=bad[0], state_version=bad[1])

    def test_unmasked_allows_all_eight_actions_and_masked_is_unchanged(self):
        for state in (START, NEAR_LOG, state_from_observation(observation(planks=3, sticks=2, table_placed=True, nearby_table=True))):
            self.assertEqual(actions_for(state, UNMASKED), tuple(range(N_ACTIONS)))
            self.assertEqual(actions_for(state, MASKED), allowed_actions(state))
        self.assertEqual(ResourceAgent().allowed(START), (A.SEARCH,))
        self.assertEqual(ResourceAgent(action_mask=UNMASKED).allowed(START), tuple(range(N_ACTIONS)))
        with self.assertRaises(ValueError):
            actions_for(START, "other")

    def test_history_bit_is_appended_only_for_resource_v2(self):
        self.assertEqual(encode_state(START), START)
        self.assertEqual(encode_state(START, "resource_v2"), START + (0,))
        self.assertEqual(encode_state(START, "resource_v2", True), START + (1,))
        for bad in ((START, "resource_v2", 1), (START + (0,), "resource_v2", False), (START, "resource_v9", False)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                encode_state(*bad)
        agent = ResourceAgent(action_mask=UNMASKED, state_version="resource_v2")
        self.assertEqual(agent.condition, "unmasked_history")
        self.assertEqual(agent.values(START + (1,)), [0.0] * N_ACTIONS)
        with self.assertRaises(ValueError):
            agent.values(START)  # a resource_v1 state is not a resource_v2 state

    def test_unmasked_learners_update_on_infeasible_actions_and_masked_learners_refuse_them(self):
        unmasked = ResourceAgent(action_mask=UNMASKED, alpha=0.5, gamma=0.9)
        td = unmasked.update(START, A.CRAFT_PICKAXE, -0.01, START, False)
        self.assertAlmostEqual(td, -0.01)
        self.assertAlmostEqual(unmasked.values(START)[A.CRAFT_PICKAXE], -0.005)
        # Q-learning bootstraps over all eight actions without the mask.
        unmasked.q[NEAR_LOG] = [0.0] * N_ACTIONS
        unmasked.q[NEAR_LOG][A.CRAFT_TABLE] = 2.0
        self.assertAlmostEqual(unmasked.update(START, A.SEARCH, 0.0, NEAR_LOG, False), 0.9 * 2.0)
        with self.assertRaises(ValueError):
            ResourceAgent(alpha=0.5).update(START, A.CRAFT_PICKAXE, -0.01, START, False)
        sarsa = ResourceAgent("sarsa", action_mask=UNMASKED, alpha=0.5, gamma=0.9)
        sarsa.update(START, A.CHOP, -0.01, START, False, next_action=A.CRAFT_TABLE)
        with self.assertRaises(ValueError):
            ResourceAgent("sarsa", alpha=0.5).update(NEAR_LOG, A.CHOP, -0.01, START, False, next_action=A.CRAFT_TABLE)
        for _ in range(50):
            self.assertIn(ResourceAgent("random", action_mask=UNMASKED).act(START), range(N_ACTIONS))

    def test_policy_files_carry_their_condition_and_reject_mixed_ones(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            for name, (mask, version) in CONDITIONS.items():
                agent = ResourceAgent("q_learning", action_mask=mask, state_version=version)
                state = encode_state(START, version, True)
                agent.q[state] = [float(i) for i in range(N_ACTIONS)]
                agent.save(path)
                payload = json.loads(path.read_text())
                self.assertEqual((payload["condition"], payload["action_mask"], payload["environment"]), (name, mask, version))
                loaded = ResourceAgent.load(path)
                self.assertEqual((loaded.condition, loaded.q), (name, agent.q))
            # A file that claims the mask is off but carries v1 caps under the v2 name is refused.
            broken = copy.deepcopy(payload)
            broken["state_caps"] = broken["state_caps"][:9]
            path.write_text(json.dumps(broken))
            with self.assertRaises(ValueError):
                ResourceAgent.load(path)
            broken = copy.deepcopy(payload)
            broken["action_mask"] = MASKED  # masked with resource_v2 is not a declared condition
            path.write_text(json.dumps(broken))
            with self.assertRaises(ValueError):
                ResourceAgent.load(path)
            # Checkpoints written before conditions existed still load as masked.
            legacy = ResourceAgent()
            legacy.q[START] = [1.0] * N_ACTIONS
            legacy.save(path)
            payload = json.loads(path.read_text())
            del payload["condition"]
            path.write_text(json.dumps(payload))
            self.assertEqual(ResourceAgent.load(path).condition, "masked")


class UnmaskedSimulatorTests(unittest.TestCase):
    def test_infeasible_actions_change_nothing_and_a_second_table_is_never_placed(self):
        env = ResourceSimulator(ResourceSimConfig(search_success=1, chop_success=1, macro_success=1, action_mask=UNMASKED))
        start = env.reset(3)
        for action in (A.CRAFT_PLANKS, A.CRAFT_STICKS, A.CRAFT_TABLE, A.PLACE_TABLE, A.APPROACH_TABLE, A.CRAFT_PICKAXE):
            following, reward, terminated, truncated, info = env.step(int(action))
            self.assertEqual(info["event"], "unavailable")
            self.assertEqual(following["inventory_counts"], start["inventory_counts"])
            self.assertFalse(terminated)
        # Collect enough wood for two tables, place one, then try to place the second.
        for action in (A.SEARCH, A.CHOP, A.SEARCH, A.CHOP, A.CRAFT_PLANKS, A.CRAFT_PLANKS, A.CRAFT_TABLE,
                       A.CRAFT_TABLE, A.PLACE_TABLE):
            env.step(int(action))
        self.assertEqual(env.observe()["inventory_counts"]["tables"], 1)
        _, _, _, _, info = env.step(int(A.PLACE_TABLE))
        self.assertEqual(info["event"], "unavailable")
        self.assertEqual(env.observe()["inventory_counts"]["tables"], 1)
        # The masked simulator still refuses the same redundant action before it happens.
        masked = ResourceSimulator(ResourceSimConfig(search_success=1, chop_success=1, macro_success=1))
        masked.reset(3)
        self.assertEqual(masked.step(int(A.CRAFT_PICKAXE))[4]["event"], "masked_action")

    def test_training_follows_the_condition_and_rejects_mismatches(self):
        for name in CONDITIONS:
            agent, rows = train_simulator(5, seed=1, condition=name, schedule="linear", epsilon_start=0.6, epsilon_end=0.1)
            self.assertEqual(agent.condition, name)
            self.assertTrue(all(row["condition"] == name for row in rows))
            self.assertAlmostEqual(rows[0]["epsilon"], 0.6)
            self.assertAlmostEqual(rows[-1]["epsilon"], 0.1)
        history_agent, _ = train_simulator(5, seed=1, condition="unmasked_history")
        self.assertTrue(all(len(state) == 10 for state in history_agent.q))
        for kwargs in ({"condition": "other"}, {"schedule": "other"},
                       {"condition": "unmasked", "config": ResourceSimConfig(action_mask=MASKED)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                train_simulator(2, **kwargs)
        with self.assertRaises(ValueError):
            ResourceSimConfig(action_mask="other")


if __name__ == "__main__":
    unittest.main()
