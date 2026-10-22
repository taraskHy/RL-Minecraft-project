"""Behavior checks for the educational crafting MDP."""

import unittest

from minecraft_rl.environment import (
    ACTION_NAMES, INITIAL_LOG_MASK, LAVA, LOG_POSITIONS, N_ACTIONS, SIZE,
    WORKBENCH, Action, CraftingEnv, EnvConfig, potential,
)


class CraftingEnvTests(unittest.TestCase):
    def make_env(self, **kwargs):
        options = {"slip_probability": 0.0, "reward_mode": "sparse"}
        options.update(kwargs)
        env = CraftingEnv(EnvConfig(**options))
        env.reset(seed=7)
        return env

    def test_reset_contains_every_observable_state_variable(self):
        env = self.make_env()
        starts = set()
        for seed in range(200):
            state = env.reset(seed=seed)
            self.assertEqual(len(state), 6)
            self.assertTrue(all(type(value) is int for value in state))
            self.assertEqual(state[2:], (INITIAL_LOG_MASK, 0, 0, 0))
            self.assertNotIn(state[:2], (*LOG_POSITIONS, LAVA))
            starts.add(state[:2])
        self.assertIn(WORKBENCH, starts)
        self.assertEqual(len(starts), SIZE * SIZE - len(LOG_POSITIONS) - 1)

    def test_movement_respects_walls_and_uncut_logs(self):
        env = self.make_env()
        env.state = (0, 1, 3, 0, 0, 0)
        for action in (Action.WEST, Action.EAST):
            state, reward, terminated, truncated, info = env.step(action)
            self.assertEqual(state, (0, 1, 3, 0, 0, 0))
            self.assertEqual(info["event"], "blocked")
            self.assertEqual(reward, -0.01)
            self.assertFalse(terminated or truncated)
        env.step(Action.CHOP)
        self.assertEqual(env.step(Action.EAST)[0][:2], (1, 1))

    def test_log_can_only_be_harvested_once_and_only_when_adjacent(self):
        env = self.make_env()
        env.state = (0, 0, 3, 0, 0, 0)
        self.assertEqual(env.step(Action.CHOP)[0], env.state)
        self.assertEqual(env.state[2:], (3, 0, 0, 0))
        env.step(Action.SOUTH)
        self.assertEqual(env.step(Action.CHOP)[0], (0, 1, 2, 1, 0, 0))
        for _ in range(3):
            self.assertEqual(env.step(Action.CHOP)[0], (0, 1, 2, 1, 0, 0))

    def test_recipe_quantities_and_pickaxe_require_workbench(self):
        env = self.make_env()
        env.state = (0, 0, 0, 2, 0, 0)
        self.assertEqual(env.step(Action.CRAFT_PLANKS)[0][3:], (1, 4, 0))
        self.assertEqual(env.step(Action.CRAFT_STICKS)[0][3:], (1, 2, 4))
        self.assertEqual(env.step(Action.CRAFT_PLANKS)[0][3:], (0, 6, 4))
        state, _, terminated, _, info = env.step(Action.CRAFT_PICKAXE)
        self.assertEqual(state[3:], (0, 6, 4))
        self.assertFalse(terminated)
        self.assertEqual(info["event"], "not_at_workbench")
        env.state = (*WORKBENCH, *state[2:])
        state, reward, terminated, truncated, info = env.step(Action.CRAFT_PICKAXE)
        self.assertEqual(state[3:], (0, 3, 2))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(reward, 9.99)
        self.assertTrue(info["success"])
        self.assertFalse(info["death"])

    def test_failed_recipes_do_not_create_or_consume_resources(self):
        env = self.make_env()
        env.state = (*WORKBENCH, 0, 0, 1, 1)
        before = env.state
        for action in (Action.CRAFT_PLANKS, Action.CRAFT_STICKS, Action.CRAFT_PICKAXE):
            state, reward, terminated, _, _ = env.step(action)
            self.assertEqual(state, before)
            self.assertEqual(reward, -0.01)
            self.assertFalse(terminated)

    def test_complete_task_from_uncut_logs(self):
        env = self.make_env()
        env.state = (0, 1, 3, 0, 0, 0)
        actions = (Action.CHOP, Action.CRAFT_PLANKS, Action.CRAFT_STICKS,
                   Action.EAST, Action.EAST, Action.EAST, Action.CHOP,
                   Action.CRAFT_PLANKS, Action.EAST,
                   Action.SOUTH, Action.SOUTH, Action.SOUTH, Action.CRAFT_PICKAXE)
        for action in actions:
            state, _, terminated, truncated, info = env.step(action)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["success"])
        self.assertEqual(state, (4, 4, 0, 0, 3, 2))
        self.assertEqual(info["steps"], len(actions))

    def test_lava_is_terminal_and_time_limit_is_only_truncation(self):
        env = self.make_env(max_steps=1)
        env.state = (1, 3, 3, 0, 0, 0)
        state, reward, terminated, truncated, info = env.step(Action.EAST)
        self.assertEqual(state[:2], LAVA)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(reward, -10.01)
        self.assertTrue(info["death"])
        self.assertFalse(info["success"])
        env.reset(seed=1)
        _, reward, terminated, truncated, info = env.step(Action.CRAFT_PICKAXE)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(reward, -0.01)
        self.assertFalse(info["death"] or info["success"])

    def test_step_requires_reset_before_start_and_after_episode(self):
        env = CraftingEnv(EnvConfig(max_steps=1))
        with self.assertRaises(RuntimeError):
            env.step(Action.NORTH)
        env.reset(seed=7)
        env.step(Action.CRAFT_PLANKS)
        with self.assertRaises(RuntimeError):
            env.step(Action.CRAFT_PLANKS)
        env.reset(seed=7)
        env.state = (*WORKBENCH, 0, 0, 3, 2)
        env.step(Action.CRAFT_PICKAXE)
        with self.assertRaises(RuntimeError):
            env.step(Action.CRAFT_PICKAXE)

    def test_potential_uses_current_inventory_and_can_decrease(self):
        self.assertEqual(potential((0, 0, 3, 0, 0, 0)), 0)
        self.assertEqual(potential((0, 0, 2, 0, 4, 4)), 3)
        env = self.make_env(reward_mode="shaped", gamma=0.97)
        env.state = (0, 0, 0, 0, 4, 4)
        state, reward, _, _, info = env.step(Action.CRAFT_STICKS)
        self.assertEqual(potential(state), 2)
        self.assertAlmostEqual(info["shaping_reward"], 0.97 * 2 - 3)
        self.assertAlmostEqual(reward, info["base_reward"] + info["shaping_reward"])

    def test_discounted_shaping_telescopes_including_terminal(self):
        gamma = 0.91
        env = self.make_env(reward_mode="shaped", gamma=gamma)
        env.state = (*WORKBENCH, 0, 2, 0, 0)
        initial_potential = potential(env.state)
        actions = (Action.CRAFT_PLANKS, Action.CRAFT_STICKS,
                   Action.CRAFT_PLANKS, Action.CRAFT_PICKAXE)
        shaping_sum = 0.0
        for index, action in enumerate(actions):
            state, _, terminated, _, info = env.step(action)
            shaping_sum += gamma ** index * info["shaping_reward"]
            final_potential = 0.0 if terminated else potential(state)
            self.assertAlmostEqual(shaping_sum,
                                   -initial_potential + gamma ** (index + 1) * final_potential)
        self.assertTrue(terminated)
        self.assertAlmostEqual(shaping_sum, -initial_potential)

    def test_truncation_preserves_potential_for_bootstrapping(self):
        env = self.make_env(reward_mode="shaped", gamma=0.9, max_steps=1)
        env.state = (0, 1, 3, 0, 0, 0)
        state, _, terminated, truncated, info = env.step(Action.CHOP)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(potential(state), 1)
        self.assertAlmostEqual(info["shaping_reward"], 0.9)

    def test_seed_controls_initial_position_and_slips(self):
        config = EnvConfig(slip_probability=1.0, max_steps=50)
        first, second = CraftingEnv(config), CraftingEnv(config)
        self.assertEqual(first.reset(seed=1234), second.reset(seed=1234))
        actual_positions = set()
        for action in (Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST) * 10:
            a, b = first.step(action), second.step(action)
            self.assertEqual(a, b)
            actual_positions.add(a[0][:2])
            if a[2] or a[3]:
                break
        self.assertGreater(len(actual_positions), 1)
        self.assertEqual(first.reset(seed=1234), second.reset(seed=1234))

    def test_invalid_config_action_seed_and_fixture_are_rejected(self):
        invalid = ({"reward_mode": "dense"}, {"gamma": float("nan")},
                   {"gamma": -0.1}, {"gamma": True}, {"slip_probability": 1.1},
                   {"max_steps": 0}, {"max_steps": 1.5}, {"max_steps": True})
        for kwargs in invalid:
            with self.subTest(config=kwargs), self.assertRaises(ValueError):
                EnvConfig(**kwargs)
        env = self.make_env()
        for action in (-1, N_ACTIONS, 0.0, "NORTH", True):
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.step(action)
        for seed in ("seed", 1.5, True):
            with self.subTest(seed=seed), self.assertRaises(TypeError):
                env.reset(seed=seed)
        for state in ((0, 0), (0, 0, 4, 0, 0, 0), (0, 0, 3, -1, 0, 0),
                      (1, 1, 3, 0, 0, 0), (SIZE, 0, 0, 0, 0, 0)):
            with self.subTest(state=state), self.assertRaises(ValueError):
                env.state = state
        self.assertEqual(ACTION_NAMES, tuple(action.name for action in Action))

    def test_render_exposes_map_and_inventory(self):
        env = self.make_env()
        env.state = (0, 0, 3, 1, 4, 2)
        text = env.render()
        self.assertEqual(text.splitlines()[0], "A . . . .")
        self.assertIn("logs=1 planks=4 sticks=2", text)
        self.assertIn("T", text)
        self.assertIn("L", text)
        self.assertIn("W", text)


if __name__ == "__main__":
    unittest.main()
