"""Stone stage: state stone_v1, reward, reference mask, hand-off, agent, and the SIMULATOR ONLY dynamics."""

import copy
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import tempfile
import unittest
from unittest import mock

from minecraft_rl import stone
from minecraft_rl.resources import MASKED, UNMASKED, WOOD_STAGE, ResourceAgent, state_from_observation
from minecraft_rl.stone import (
    HANDOFF_REQUIREMENTS, PICKAXE_MATERIALS, PICKAXE_STICKS, STONE_ACTION_NAMES, STONE_CAPS, STONE_CONDITION,
    STONE_ENVIRONMENT, STONE_FIELDS, STONE_LIVE_STATES, STONE_MATERIALS, STONE_STAGE, StoneAction as A,
    StoneSimConfig, StoneSimulator, bridge_action, consistent, evaluate_stone_policy, handoff_failures,
    linear_epsilon, stone_potential, stone_reference_actions, stone_reward_transition, stone_state_from_observation,
    stone_state_potential, train_stone_simulator,
)

ROOT = Path(__file__).resolve().parents[1]
STAGE1_POLICY = ROOT / "results" / "pc_resource_runs" / "87673b8" / "mx-unmasked-q-train" / "policy.json"
ALL_STATES = tuple(itertools.product(*(range(cap + 1) for cap in STONE_CAPS)))
HEALTH_FOR_BIN = {0: 0, 1: 6, 2: 20}


def observation(**changes):
    """A bridge-like stone observation; inventory keys go into inventory_counts."""
    result = {
        "inventory_counts": {"logs": 0, "planks": 3, "sticks": 2, "tables": 0, "pickaxes": 1,
                             "stone_materials": 0, "stone_pickaxes": 0},
        "nearby_stone": False, "stone_visible": False, "nearby_table": True, "table_placed": True,
        "health": 20, "food": 20, "held_item": "wooden_pickaxe", "event": "crafted_wooden_pickaxe",
        "position": {"x": 0.5, "y": 64, "z": 0.5},
    }
    for key, value in changes.items():
        if key in result["inventory_counts"]:
            result["inventory_counts"][key] = value
        else:
            result[key] = value
    return result


def at(items=0, stone=False, table=False, visible=None, health=20):
    """An observation with the given stone_v1 values; stone in reach is visible unless stated."""
    return observation(stone_materials=items, nearby_stone=stone, nearby_table=table,
                       stone_visible=stone if visible is None else visible, health=health)


def for_state(state):
    items, near_stone, visible, near_table, health = state
    return at(items, bool(near_stone), bool(near_table), bool(visible), HEALTH_FOR_BIN[health])


def pinned_minecraft_data():
    """The pinned minecraft-data 3.117.0 files for pc 1.21.4, from NODE_PATH or this checkout; None if absent."""
    roots = [Path(p) for p in os.environ.get("NODE_PATH", "").split(os.pathsep) if p]
    roots.append(ROOT / "minecraft_bridge" / "node_modules")
    for root in roots:
        for package in (root / "minecraft-data",
                        root / ".pnpm" / "minecraft-data@3.117.0" / "node_modules" / "minecraft-data"):
            try:
                if json.loads((package / "package.json").read_text(encoding="utf-8")).get("version") != "3.117.0":
                    continue
                data = package / "minecraft-data" / "data"
                paths = json.loads((data / "dataPaths.json").read_text(encoding="utf-8"))["pc"]["1.21.4"]
                return {kind: json.loads((data / paths[kind] / f"{kind}.json").read_text(encoding="utf-8"))
                        for kind in ("items", "blocks", "recipes")}
            except (OSError, ValueError, KeyError):
                continue
    return None


class StoneStateTests(unittest.TestCase):
    def test_five_actions_map_to_bridge_indices_8_to_12(self):
        self.assertEqual(STONE_ACTION_NAMES, ("SEARCH_STONE", "DIG_DOWN", "MINE_STONE", "RETURN_TO_TABLE",
                                              "CRAFT_STONE_PICKAXE"))
        self.assertEqual([int(a) for a in A], [0, 1, 2, 3, 4])
        self.assertEqual([bridge_action(int(a)) for a in A], [8, 9, 10, 11, 12])
        for bad in (-1, 5, True, 2.0, "0"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                bridge_action(bad)

    def test_stage_spec_fields_caps_and_the_48_consistent_live_states(self):
        self.assertEqual((STONE_STAGE.name, STONE_STAGE.actions, STONE_STAGE.n_actions), ("stone", STONE_ACTION_NAMES, 5))
        self.assertEqual(dict(STONE_STAGE.state_versions), {"stone_v1": (STONE_FIELDS, STONE_CAPS)})
        self.assertEqual(STONE_FIELDS, ("stone_items", "nearby_stone", "stone_visible", "nearby_table", "health_bin"))
        self.assertEqual(STONE_CAPS, (3, 1, 1, 1, 2))
        self.assertEqual(dict(STONE_STAGE.conditions), {"stone_unmasked": (UNMASKED, "stone_v1")})
        self.assertEqual(len(ALL_STATES), 96)
        self.assertEqual(sum(state[4] > 0 for state in ALL_STATES), 64)
        self.assertEqual(len(STONE_LIVE_STATES), 48)
        self.assertEqual(len(set(STONE_LIVE_STATES)), 48)
        self.assertEqual(set(STONE_LIVE_STATES), {s for s in ALL_STATES if s[4] and not (s[1] == 1 and s[2] == 0)})
        # Every observation the encoder accepts lands in a consistent state.
        for items, stone_, visible, table, health in itertools.product(range(6), (False, True), (False, True),
                                                                        (False, True), (0, 0.5, 10, 10.5, 20)):
            obs = at(items, stone_, table, visible, health)
            if stone_ and not visible:
                continue
            state = stone_state_from_observation(obs)
            self.assertTrue(consistent(state))
            self.assertEqual(state in STONE_LIVE_STATES, health > 0)

    def test_encoder_reads_exactly_five_values_and_caps_materials(self):
        minimal = {"inventory_counts": {"stone_materials": 2}, "nearby_stone": True, "stone_visible": True,
                   "nearby_table": False, "health": 15}
        self.assertEqual(stone_state_from_observation(minimal), (2, 1, 1, 0, 2))
        for materials in range(12):
            self.assertEqual(stone_state_from_observation(at(materials))[0], min(3, materials))
        # Wood counts, the pickaxe count, food, position, and every other field play no part.
        changed = at(1, True, True)
        changed["inventory_counts"].update(logs=9, planks=0, sticks=0, pickaxes=0, stone_pickaxes=4)
        changed.update(food=0, position=None, held_item=None, table_placed=False, nearby_log=True)
        self.assertEqual(stone_state_from_observation(changed), stone_state_from_observation(at(1, True, True)))

    def test_health_is_binned_exactly_as_the_wood_state(self):
        wood = {"inventory_counts": {"logs": 0, "planks": 0, "sticks": 0, "tables": 0, "pickaxes": 0},
                "nearby_log": False, "nearby_table": False, "table_placed": False,
                "food": 20, "position": {"x": 0, "y": 64, "z": 0}}
        for health in (0, 0.01, 0.5, 1, 5, 9.99, 10, 10.0001, 10.5, 11, 19.5, 20):
            with self.subTest(health=health):
                expected = state_from_observation({**wood, "health": health})[-1]
                self.assertEqual(stone_state_from_observation(at(health=health))[-1], expected)

    def test_missing_or_invalid_fields_raise(self):
        broken = []
        for field in ("nearby_stone", "stone_visible", "nearby_table", "health", "inventory_counts"):
            missing = at()
            del missing[field]
            broken.append(missing)
        no_materials = at()
        del no_materials["inventory_counts"]["stone_materials"]
        broken += [no_materials, at(items=True), at(items=-1), at(items=1.0), at(stone=1), at(table=None),
                   at(visible="yes"), at(health=float("nan")), at(health=21), at(health=-1), at(health=True),
                   "observation", None]
        for value in broken:
            with self.subTest(value=value), self.assertRaises(ValueError):
                stone_state_from_observation(value)

    def test_stone_in_reach_but_not_visible_is_rejected(self):
        # The bridge sets stone_visible whenever nearby_stone is set; a contradiction is not repaired.
        with self.assertRaises(ValueError):
            stone_state_from_observation(at(stone=True, visible=False))
        self.assertEqual(stone_state_from_observation(at(stone=False, visible=True))[1:3], (0, 1))
        self.assertFalse(consistent((0, 1, 0, 0, 2)))

    def test_diagnostic_only_never_changes_the_state(self):
        plain = at(2, True, False)
        expected = stone_state_from_observation(plain)
        for diagnostics in ({}, {"depth_to_stone": 4, "stair_refusals": {"+x": "cave_below"}},
                            {"nearby_stone": False, "stone_visible": False, "health": 0,
                             "inventory_counts": {"stone_materials": 3}}, None, [1, 2]):
            with self.subTest(diagnostics=diagnostics):
                self.assertEqual(stone_state_from_observation({**plain, "diagnostic_only": diagnostics}), expected)
                self.assertAlmostEqual(stone_potential({**plain, "diagnostic_only": diagnostics}), stone_potential(plain))

        class Guarded(dict):
            def __getitem__(self, key):
                if key == "diagnostic_only":
                    raise AssertionError("diagnostic_only was read")
                return super().__getitem__(key)

            def get(self, key, default=None):
                if key == "diagnostic_only":
                    raise AssertionError("diagnostic_only was read")
                return super().get(key, default)

        guarded = Guarded({**plain, "diagnostic_only": {"depth_to_stone": 2}})
        self.assertEqual(stone_state_from_observation(guarded), expected)
        stone_reward_transition(guarded, guarded)


class StoneRewardTests(unittest.TestCase):
    def test_potential_ranges_from_0_to_1_1(self):
        values = {state: stone_state_potential(state) for state in ALL_STATES}
        self.assertTrue(all(-1e-12 <= value <= 1.1 + 1e-12 for value in values.values()))
        self.assertAlmostEqual(min(values.values()), 0.0)
        self.assertAlmostEqual(max(values.values()), 1.1)
        self.assertAlmostEqual(stone_state_potential((3, 0, 0, 1, 2)), 1.1)
        self.assertAlmostEqual(stone_state_potential((0, 0, 1, 0, 2)), 0.0)
        # Materials beyond three add nothing; stone in reach adds nothing at three.
        self.assertAlmostEqual(stone_potential(at(7, True, False)), 0.9)
        self.assertAlmostEqual(stone_potential(at(2, True, True)), 0.7)
        for state in filter(consistent, ALL_STATES):
            self.assertAlmostEqual(stone_potential(for_state(state)), values[state])

    def test_every_row_of_the_shaped_reward_table(self):
        rows = [
            ("reaching stone with 0 items", at(0), at(0, stone=True), {}, 0.087),
            ("mining 0 to 1, stone still in reach", at(0, True), at(1, True), {}, 0.278),
            ("mining 1 to 2, stone still in reach", at(1, True), at(2, True), {}, 0.269),
            ("mining 2 to 3, away from the table", at(2, True), at(3, True), {}, 0.163),
            ("returning to the table with 3 items", at(3), at(3, table=True), {}, 0.157),
            ("crafting the stone pickaxe", at(3, table=True), at(0, table=True), {"success": True}, 8.89),
            ("leaving the table with 3 items", at(3, table=True), at(3), {}, -0.237),
            ("leaving stone with 1 item", at(1, True), at(1), {}, -0.119),
        ]
        for name, before, after, flags, printed in rows:
            with self.subTest(row=name):
                reward, parts = stone_reward_transition(before, after, gamma=0.97, **flags)
                # The printed values are exact, so they hold well past their printed precision.
                self.assertAlmostEqual(reward, printed, places=9)
                self.assertAlmostEqual(parts["base_reward"] + parts["shaping_reward"], reward)
        for state in STONE_LIVE_STATES:
            obs = for_state(state)
            phi = stone_state_potential(state)
            unchanged, _ = stone_reward_transition(obs, obs)
            self.assertAlmostEqual(unchanged, -0.01 - 0.03 * phi)
            self.assertTrue(-0.043 - 1e-12 <= unchanged <= -0.01 + 1e-12)
            died, parts = stone_reward_transition(obs, {**obs, "health": 0}, death=True)
            self.assertAlmostEqual(died, -10.01 - phi)
            self.assertAlmostEqual(parts["shaping_reward"], -phi)

    def test_progress_is_positive_and_a_first_update_from_zero_is_positive(self):
        progress = [(at(0), at(0, True)), (at(0, True), at(1, True)), (at(1, True), at(2, True)),
                    (at(2, True), at(3, True)), (at(3), at(3, table=True))]
        for before, after in progress:
            reward, _ = stone_reward_transition(before, after)
            self.assertGreater(reward, 0)
            agent = ResourceAgent(stage=STONE_STAGE)
            state, following = stone_state_from_observation(before), stone_state_from_observation(after)
            agent.update(state, int(A.MINE_STONE), reward, following, False)
            self.assertAlmostEqual(agent.values(state)[A.MINE_STONE], 0.2 * reward)
            self.assertGreater(agent.values(state)[A.MINE_STONE], 0)

    def test_terminal_potential_is_zero_and_truncation_is_not_termination(self):
        before, after = at(2, True), at(3, True, True)
        _, success = stone_reward_transition(before, after, success=True)
        self.assertAlmostEqual(success["shaping_reward"], -stone_potential(before))
        self.assertAlmostEqual(success["base_reward"], 9.99)
        # A time limit is an ordinary continuation: the following potential stays.
        _, ordinary = stone_reward_transition(before, after)
        self.assertAlmostEqual(ordinary["shaping_reward"], 0.97 * stone_potential(after) - stone_potential(before))
        sparse, parts = stone_reward_transition(before, after, reward_mode="sparse")
        self.assertEqual((sparse, parts["shaping_reward"]), (-0.01, 0.0))
        # Shaping telescopes: a round trip cannot earn repeatable positive reward.
        _, back = stone_reward_transition(after, before)
        self.assertLess(ordinary["shaping_reward"] + 0.97 * back["shaping_reward"], 0)

    def test_reward_rejects_invalid_arguments(self):
        obs = at()
        for kwargs in ({"gamma": 1}, {"gamma": float("nan")}, {"gamma": True}, {"reward_mode": "other"},
                       {"success": True, "death": True}, {"success": 1}, {"death": None}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                stone_reward_transition(obs, obs, **kwargs)
        with self.assertRaises(ValueError):
            stone_reward_transition(obs, at(stone=True, visible=False))


class StoneReferenceMaskTests(unittest.TestCase):
    def test_reference_mask_rules(self):
        self.assertEqual(stone.STONE_REFERENCE_MASK, "stone_reference_v1")
        for state in STONE_LIVE_STATES:
            items, near_stone, _, near_table, _ = state
            if items < 3 and not near_stone:
                expected = {A.SEARCH_STONE, A.DIG_DOWN}
            elif items < 3:
                expected = {A.MINE_STONE}
            elif not near_table:
                expected = {A.RETURN_TO_TABLE}
            else:
                expected = {A.CRAFT_STONE_PICKAXE}
            self.assertEqual(set(stone_reference_actions(state)), expected)

    def test_the_agent_never_consults_the_reference_mask(self):
        def refuse(state):
            raise AssertionError("the reference mask chose an action")

        with mock.patch.object(stone, "stone_reference_actions", refuse):
            for algorithm in ("q_learning", "sarsa", "random"):
                agent = ResourceAgent(algorithm, seed=3, stage=STONE_STAGE)
                for state in STONE_LIVE_STATES:
                    self.assertEqual(agent.allowed(state), (0, 1, 2, 3, 4))
                    for epsilon in (0, 0.3, 1):
                        self.assertIn(agent.act(state, epsilon), range(5))
        # Training with a different diagnostic mask learns exactly the same; only the diagnostic count changes.
        for algorithm in ("q_learning", "sarsa"):
            baseline, rows = train_stone_simulator(20, seed=2, algorithm=algorithm)
            with mock.patch.object(stone, "stone_reference_actions", lambda state: (int(A.CRAFT_STONE_PICKAXE),)):
                other, other_rows = train_stone_simulator(20, seed=2, algorithm=algorithm)
            self.assertEqual(baseline.q, other.q)
            strip = [{k: v for k, v in row.items() if k != "outside_reference_choices"} for row in rows]
            self.assertEqual(strip, [{k: v for k, v in row.items() if k != "outside_reference_choices"}
                                     for row in other_rows])
            self.assertNotEqual([r["outside_reference_choices"] for r in rows],
                                [r["outside_reference_choices"] for r in other_rows])


class HandoffTests(unittest.TestCase):
    def test_a_typical_wood_success_may_start_the_stone_stage(self):
        # 114 of 124 retained successes ended with 0 logs, 3 planks and 2 sticks.
        self.assertEqual(handoff_failures(observation()), ())
        self.assertEqual(handoff_failures(observation(planks=1, sticks=6, stone_materials=2)), ())

    def test_each_unmet_requirement_is_named(self):
        cases = [
            (observation(event="crafted_stick"), "wooden_pickaxe_not_crafted"),
            (observation(event="already_completed"), "wooden_pickaxe_not_crafted"),
            (observation(held_item="stick"), "wooden_pickaxe_not_held"),
            (observation(held_item=None), "wooden_pickaxe_not_held"),
            (observation(pickaxes=0), "wooden_pickaxe_not_held"),
            (observation(sticks=1), "sticks_short"),
            (observation(table_placed=False, nearby_table=True), "table_not_placed"),
            (observation(nearby_table=False), "table_not_in_reach"),
            (observation(stone_materials=3), "stone_materials_already_held"),
        ]
        for obs, requirement in cases:
            with self.subTest(requirement=requirement):
                self.assertEqual(handoff_failures(obs), (requirement,))
        everything = observation(event="observe", held_item=None, sticks=0, table_placed=False,
                                 nearby_table=False, stone_materials=5)
        self.assertEqual(handoff_failures(everything), HANDOFF_REQUIREMENTS)

    def test_an_incomplete_observation_raises_instead_of_failing_a_requirement(self):
        broken = []
        for field in ("event", "table_placed", "nearby_table", "inventory_counts"):
            missing = observation()
            del missing[field]
            broken.append(missing)
        for field in ("pickaxes", "sticks", "stone_materials"):
            missing = observation()
            del missing["inventory_counts"][field]
            broken.append(missing)
        broken += [observation(sticks=True), observation(sticks=-1), observation(nearby_table=1),
                   observation(held_item=3), observation(event=None), []]
        for value in broken:
            with self.subTest(value=value), self.assertRaises(ValueError):
                handoff_failures(value)

    def test_requirements_match_the_pinned_minecraft_data(self):
        data = pinned_minecraft_data()
        if data is None:
            self.skipTest("minecraft-data 3.117.0 not found (set NODE_PATH to the bridge's node_modules)")
        items = {item["id"]: item["name"] for item in data["items"]}
        ids = {name: number for number, name in items.items()}
        recipes = data["recipes"][str(ids["stone_pickaxe"])]
        self.assertEqual(len(recipes), 3)
        top_rows = set()
        for recipe in recipes:
            shape = recipe["inShape"]
            # Three rows: the 2x2 inventory grid cannot hold it, so the table must be in reach.
            self.assertEqual((len(shape), [len(row) for row in shape]), (3, [3, 3, 3]))
            self.assertEqual(len(set(shape[0])), 1)
            top_rows.add(items[shape[0][0]])
            cells = [cell for row in shape for cell in row if cell is not None]
            self.assertEqual(cells.count(shape[0][0]), PICKAXE_MATERIALS)
            self.assertEqual(cells.count(ids["stick"]), PICKAXE_STICKS)
            self.assertEqual(len(cells), PICKAXE_MATERIALS + PICKAXE_STICKS)
        self.assertEqual(top_rows, set(STONE_MATERIALS))
        # The wooden pickaxe can harvest a block that drops each counted material.
        blocks = {block["name"]: block for block in data["blocks"]}
        wooden = str(ids["wooden_pickaxe"])
        sources = {"stone": "cobblestone", "cobblestone": "cobblestone", "deepslate": "cobbled_deepslate",
                   "cobbled_deepslate": "cobbled_deepslate", "blackstone": "blackstone"}
        for name, drop in sources.items():
            with self.subTest(block=name):
                self.assertIn(wooden, blocks[name].get("harvestTools") or {})
                self.assertEqual([items[d] for d in blocks[name]["drops"]], [drop])
        self.assertEqual(set(sources.values()), set(STONE_MATERIALS))
        for lookalike in ("granite", "andesite", "diorite", "tuff", "calcite", "sandstone", "terracotta",
                          "mossy_cobblestone"):
            self.assertNotIn(items[blocks[lookalike]["drops"][0]], STONE_MATERIALS)


class StoneAgentTests(unittest.TestCase):
    S = (0, 0, 0, 1, 2)
    T = (0, 1, 1, 0, 2)

    def test_a_stone_agent_has_five_unmasked_actions_in_every_state(self):
        agent = ResourceAgent(stage=STONE_STAGE)
        self.assertEqual((agent.condition, agent.action_mask, agent.state_version),
                         (STONE_CONDITION, UNMASKED, STONE_ENVIRONMENT))
        self.assertEqual(agent.values(self.S), [0.0] * 5)
        for state in ALL_STATES:
            self.assertEqual(agent.allowed(state), (0, 1, 2, 3, 4))
        for bad in ((MASKED, STONE_ENVIRONMENT), (UNMASKED, "resource_v1"), (MASKED, "resource_v1")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ResourceAgent(stage=STONE_STAGE, action_mask=bad[0], state_version=bad[1])
        for bad in ((0, 0, 0, 0, 2, 0), (4, 0, 0, 0, 2), (0, 0, 0, 0, 3), (0, 0, 0, 0, 0, 0, 0, 0, 2), [0, 0, 0, 0, 2]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                agent.values(bad)
        with self.assertRaises(TypeError):
            ResourceAgent(stage="stone")

    def test_updates_bootstrap_over_all_five_actions(self):
        agent = ResourceAgent(alpha=0.5, gamma=0.9, stage=STONE_STAGE)
        agent.q[self.T] = [1.0, 2.0, 4.0, -1.0, 3.0]
        self.assertAlmostEqual(agent.update(self.S, int(A.DIG_DOWN), 0.1, self.T, False), 0.1 + 0.9 * 4.0)
        sarsa = ResourceAgent("sarsa", alpha=0.5, gamma=0.9, stage=STONE_STAGE)
        sarsa.q[self.T] = [1.0, 2.0, 4.0, -1.0, 3.0]
        sarsa.update(self.S, int(A.DIG_DOWN), 0.1, self.T, False, int(A.RETURN_TO_TABLE))
        self.assertAlmostEqual(sarsa.values(self.S)[A.DIG_DOWN], 0.5 * (0.1 - 0.9))
        terminal = ResourceAgent(alpha=1, stage=STONE_STAGE)
        terminal.q[self.T] = [9.0] * 5
        terminal.update(self.S, int(A.CRAFT_STONE_PICKAXE), 8.89, self.T, True)
        self.assertAlmostEqual(terminal.values(self.S)[A.CRAFT_STONE_PICKAXE], 8.89)
        for bad in (5, -1, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                agent.update(self.S, bad, 0.0, self.T, False)

    def test_save_and_load_record_the_stone_stage(self):
        agent = ResourceAgent("sarsa", alpha=0.3, seed=4, stage=STONE_STAGE, initial_q=-0.5)
        agent.update(self.S, int(A.SEARCH_STONE), 0.087, self.T, False, int(A.MINE_STONE))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stone.json"
            agent.save(path, {"source": "stone simulator"})
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(list(payload)[:3], ["format_version", "stage", "environment"])
            self.assertEqual((payload["stage"], payload["environment"], payload["actions"], payload["state_fields"],
                              payload["state_caps"], payload["action_mask"], payload["condition"]),
                             ("stone", "stone_v1", list(STONE_ACTION_NAMES), list(STONE_FIELDS), list(STONE_CAPS),
                              "none", "stone_unmasked"))
            loaded = ResourceAgent.load(path, stage=STONE_STAGE)
            self.assertIs(loaded.stage, STONE_STAGE)
            self.assertEqual((loaded.q, loaded.algorithm, loaded.alpha, loaded.initial_q, loaded.condition),
                             (agent.q, "sarsa", 0.3, -0.5, STONE_CONDITION))
            for field, bad in (("actions", list(reversed(STONE_ACTION_NAMES))), ("state_caps", [3, 1, 1, 1, 3]),
                               ("state_fields", list(STONE_FIELDS[:4])), ("action_mask", MASKED),
                               ("environment", "stone_v2"), ("format_version", 2)):
                broken = copy.deepcopy(payload)
                broken[field] = bad
                path.write_text(json.dumps(broken), encoding="utf-8")
                with self.subTest(field=field), self.assertRaises(ValueError):
                    ResourceAgent.load(path, stage=STONE_STAGE)
            for values in ([0.0] * 8, [0.0] * 4, [0.0, 0.0, float("nan"), 0.0, 0.0], [True] * 5):
                broken = copy.deepcopy(payload)
                broken["q"][0]["values"] = values
                path.write_text(json.dumps(broken), encoding="utf-8")
                with self.subTest(values=values), self.assertRaises(ValueError):
                    ResourceAgent.load(path, stage=STONE_STAGE)

    def test_a_policy_saved_by_the_other_stage_is_refused(self):
        stone_agent = ResourceAgent(stage=STONE_STAGE)
        stone_agent.q[self.S] = [0.1] * 5
        wood_agent = ResourceAgent(action_mask=UNMASKED)
        wood_agent.q[(0,) * 8 + (2,)] = [0.1] * 8
        with tempfile.TemporaryDirectory() as directory:
            stone_path, wood_path = Path(directory) / "stone.json", Path(directory) / "wood.json"
            stone_agent.save(stone_path)
            wood_agent.save(wood_path)
            with self.assertRaisesRegex(ValueError, "saved by the stone stage .*a wood-stage agent cannot load it"):
                ResourceAgent.load(stone_path)
            with self.assertRaisesRegex(ValueError, "saved by the wood stage .*a stone-stage agent cannot load it"):
                ResourceAgent.load(wood_path, stage=STONE_STAGE)
            # A stone file that drops or forges its stage key is still refused by both stages.
            payload = json.loads(stone_path.read_text(encoding="utf-8"))
            for stage_value in (None, "wood"):
                forged = {k: v for k, v in payload.items() if k != "stage"}
                if stage_value:
                    forged["stage"] = stage_value
                stone_path.write_text(json.dumps(forged), encoding="utf-8")
                for stage in (WOOD_STAGE, STONE_STAGE):
                    with self.subTest(stage_value=stage_value, stage=stage.name), self.assertRaises(ValueError):
                        ResourceAgent.load(stone_path, stage=stage)
            with self.assertRaises(TypeError):
                ResourceAgent.load(wood_path, stage="wood")
        # The retained stage-1 checkpoint is a wood policy: the stone stage refuses it (read only).
        with self.assertRaisesRegex(ValueError, "wood stage"):
            ResourceAgent.load(STAGE1_POLICY, stage=STONE_STAGE)
        self.assertEqual(len(ResourceAgent.load(STAGE1_POLICY).q), 180)

    def test_initial_q_is_carried(self):
        agent = ResourceAgent(stage=STONE_STAGE, initial_q=-1)
        self.assertEqual(agent.values(self.T), [-1.0] * 5)
        agent.update(self.S, int(A.DIG_DOWN), 0.087, self.T, False)
        self.assertAlmostEqual(agent.values(self.S)[A.DIG_DOWN], -1 + 0.2 * (0.087 + 0.97 * -1 + 1))
        self.assertEqual(agent.values(self.S)[A.SEARCH_STONE], -1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stone.json"
            agent.save(path)
            loaded = ResourceAgent.load(path, stage=STONE_STAGE)
            self.assertEqual((loaded.initial_q, loaded.values(self.T)), (-1.0, [-1.0] * 5))
            payload = json.loads(path.read_text(encoding="utf-8"))
            del payload["initial_q"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(ResourceAgent.load(path, stage=STONE_STAGE).initial_q, 0.0)
        for initial_q in (0.0, -0.5, -1.0):
            trained, _ = train_stone_simulator(3, initial_q=initial_q)
            self.assertEqual(trained.initial_q, initial_q)
        with self.assertRaises(ValueError):
            train_stone_simulator(3, initial_q=float("nan"))


def sim(**changes):
    return StoneSimulator(StoneSimConfig(**changes))


# Every macro succeeds and nothing hurts: the bot can always dig to stone and mine it.
CERTAIN = {"q_d": 1.0, "dig_good": 1.0, "p_mine": 1.0, "p_ret": 1.0, "p_craft": 1.0, "p_damage": 0.0, "p_death": 0.0}


class StoneSimulatorTests(unittest.TestCase):
    def test_reset_is_the_handoff_state(self):
        env = sim()
        for seed in range(50):
            obs = env.reset(seed)
            self.assertEqual(obs["inventory_counts"], {"logs": 0, "planks": 3, "sticks": 2, "tables": 0,
                                                       "pickaxes": 1, "stone_materials": 0, "stone_pickaxes": 0})
            state = stone_state_from_observation(obs)
            self.assertEqual((state[0], state[1], state[3], state[4]), (0, 0, 1, 2))
            self.assertEqual(bool(state[2]), env.hidden_terrain()["visible_at_start"])
            self.assertEqual(handoff_failures({**obs, "event": "crafted_wooden_pickaxe"}), ())
        self.assertEqual(StoneSimConfig().max_steps, 30)
        with self.assertRaises(RuntimeError):
            StoneSimulator().observe()
        with self.assertRaises(RuntimeError):
            StoneSimulator().step(0)

    def test_materials_come_only_from_mine_stone(self):
        env = sim(p_death=0.0)
        rng = random.Random(7)
        mined = 0
        for episode in range(300):
            obs = env.reset(episode)
            done = False
            while not done:
                action = rng.randrange(5)
                following, _, terminated, truncated, info = env.step(action)
                change = following["inventory_counts"]["stone_materials"] - obs["inventory_counts"]["stone_materials"]
                if change > 0:
                    self.assertEqual((action, change, info["event"]), (A.MINE_STONE, 1, "cobblestone_collected"))
                    mined += 1
                elif change < 0:
                    self.assertEqual((action, change, info["event"]), (A.CRAFT_STONE_PICKAXE, -3, "crafted_stone_pickaxe"))
                if action == A.MINE_STONE and change == 0:
                    self.assertIn(info["event"], ("no_reachable_stone", "stone_unreachable"))
                obs, done = following, terminated or truncated
        self.assertGreater(mined, 100)

    def test_a_craft_uses_three_materials_and_two_sticks_at_the_table(self):
        env = sim(**CERTAIN)
        obs = env.reset(11)
        while obs["inventory_counts"]["stone_materials"] < 4:
            action = A.MINE_STONE if obs["nearby_stone"] else A.DIG_DOWN
            obs, _, terminated, _, info = env.step(int(action))
            self.assertFalse(terminated)
        self.assertFalse(obs["nearby_table"])
        before, _, _, _, info = env.step(int(A.CRAFT_STONE_PICKAXE))
        self.assertEqual(info["event"], "not_at_own_table")
        self.assertEqual(before["inventory_counts"], obs["inventory_counts"])
        before, _, _, _, info = env.step(int(A.RETURN_TO_TABLE))
        self.assertEqual(info["event"], "table_reached")
        self.assertFalse(before["nearby_stone"])
        after, reward, terminated, truncated, info = env.step(int(A.CRAFT_STONE_PICKAXE))
        self.assertEqual((terminated, truncated, info["success"], info["event"]), (True, False, True, "crafted_stone_pickaxe"))
        self.assertEqual(after["inventory_counts"], {**before["inventory_counts"], "stone_materials": 1,
                                                     "sticks": 0, "stone_pickaxes": 1})
        self.assertEqual(after["held_item"], "stone_pickaxe")
        self.assertAlmostEqual(reward, 9.99 - stone_potential(before))
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_crafting_needs_the_table_three_materials_and_two_sticks(self):
        env = sim(**CERTAIN)
        env.reset(1)
        _, _, terminated, _, info = env.step(int(A.CRAFT_STONE_PICKAXE))
        self.assertEqual((info["event"], terminated), ("missing_materials", False))
        env.reset(1)
        env._observation["inventory_counts"].update(stone_materials=3, sticks=1)
        _, _, terminated, _, info = env.step(int(A.CRAFT_STONE_PICKAXE))
        self.assertEqual((info["event"], terminated), ("missing_materials", False))
        env.reset(1)
        env._observation["inventory_counts"]["stone_materials"] = 3
        env._observation["nearby_table"] = False
        _, _, terminated, _, info = env.step(int(A.CRAFT_STONE_PICKAXE))
        self.assertEqual((info["event"], terminated), ("not_at_own_table", False))
        failing = sim(p_craft=0.0)
        failing.reset(1)
        failing._observation["inventory_counts"]["stone_materials"] = 3
        obs, _, terminated, _, info = failing.step(int(A.CRAFT_STONE_PICKAXE))
        self.assertEqual((info["event"], terminated, obs["inventory_counts"]["stone_materials"]),
                         ("craft_not_confirmed", False, 3))

    def test_refusals_change_nothing(self):
        def check(env, action, event):
            before = env.observe()
            after, reward, terminated, _, info = env.step(int(action))
            self.assertEqual(info["event"], event)
            self.assertEqual(after, before)
            self.assertFalse(terminated)
            self.assertAlmostEqual(reward, -0.01 - 0.03 * stone_potential(before))

        env = sim(**CERTAIN)
        env.reset(5)
        check(env, A.MINE_STONE, "no_reachable_stone")
        check(env, A.RETURN_TO_TABLE, "table_reached")
        check(env, A.CRAFT_STONE_PICKAXE, "missing_materials")
        env.step(int(A.DIG_DOWN))
        check(env, A.SEARCH_STONE, "stone_already_reachable")
        check(env, A.DIG_DOWN, "stone_already_reachable")
        check(env, A.CRAFT_STONE_PICKAXE, "not_at_own_table")
        bad_ground = sim(q_d=0.0, dig_bad=0.0)
        bad_ground.reset(5)
        check(bad_ground, A.DIG_DOWN, "stair_refused")
        failing = sim(q_d=1.0, dig_good=1.0, p_mine=0.0, p_ret=0.0, p_damage=0.0, p_death=0.0)
        failing.reset(5)
        failing.step(int(A.DIG_DOWN))
        check(failing, A.MINE_STONE, "stone_unreachable")
        check(failing, A.RETURN_TO_TABLE, "table_path_blocked")

    def test_search_and_return_follow_the_declared_dynamics(self):
        # Without surface stone nothing is ever visible; a search still leaves the table.
        env = sim(q_s=0.0)
        env.reset(2)
        obs, *_ = env.step(int(A.SEARCH_STONE))
        self.assertEqual((obs["stone_visible"], obs["nearby_stone"], obs["nearby_table"]), (False, False, False))
        reveal = sim(q_s=1.0, p_v=0.0, p_rev=1.0, p_nav=1.0)
        reveal.reset(2)
        obs, _, _, _, info = reveal.step(int(A.SEARCH_STONE))
        self.assertEqual((info["event"], obs["stone_visible"], obs["nearby_stone"]), ("explored;stone_visible", True, False))
        obs, _, _, _, info = reveal.step(int(A.SEARCH_STONE))
        self.assertEqual((info["event"], obs["nearby_stone"]), ("stone_reached", True))
        blocked = sim(q_s=1.0, p_v=1.0, p_nav=0.0, p_block=1.0)
        obs = blocked.reset(2)
        self.assertTrue(obs["stone_visible"])
        obs, _, _, _, info = blocked.step(int(A.SEARCH_STONE))
        self.assertEqual((info["event"], obs["stone_visible"], obs["nearby_table"]), ("stone_path_blocked", False, False))
        # A return redraws visibility from the world's surface stone.
        back = sim(q_s=1.0, p_v=1.0, p_nav=0.0, p_block=1.0, p_ret=1.0)
        back.reset(2)
        back.step(int(A.SEARCH_STONE))
        obs, _, _, _, info = back.step(int(A.RETURN_TO_TABLE))
        self.assertEqual((info["event"], obs["nearby_table"], obs["stone_visible"]), ("table_reached", True, True))

    def test_damage_death_and_the_step_limit(self):
        dying = sim(**{**CERTAIN, "p_death": 1.0})
        before = dying.reset(3)
        obs, reward, terminated, truncated, info = dying.step(int(A.DIG_DOWN))
        self.assertEqual((terminated, truncated, info["death"], obs["health"]), (True, False, True, 0))
        self.assertAlmostEqual(reward, -10.01 - stone_potential(before))
        hurt = sim(**{**CERTAIN, "p_damage": 1.0, "damage": 12})
        hurt.reset(3)
        obs, _, terminated, _, info = hurt.step(int(A.DIG_DOWN))
        self.assertEqual((obs["health"], terminated, info["event"]), (8, False, "stair_steps;stone_exposed;took_damage"))
        self.assertEqual(stone_state_from_observation(obs)[4], 1)
        limited = sim(max_steps=2)
        limited.reset(3)
        _, _, terminated, truncated, _ = limited.step(int(A.MINE_STONE))
        self.assertEqual((terminated, truncated), (False, False))
        before = limited.observe()
        after, reward, terminated, truncated, _ = limited.step(int(A.MINE_STONE))
        self.assertEqual((terminated, truncated), (False, True))
        # Truncation keeps the following potential, like any non-terminal step.
        self.assertAlmostEqual(reward, -0.01 + 0.97 * stone_potential(after) - stone_potential(before))
        with self.assertRaises(RuntimeError):
            limited.step(0)

    def test_seeded_runs_reproduce(self):
        actions = [1, 2, 0, 2, 2, 3, 4, 0, 1, 2, 2, 2, 3, 4] * 3
        runs = []
        for _ in range(2):
            env = StoneSimulator()
            trace = [env.reset(42)]
            for action in actions:
                step = env.step(action)
                trace.append(step)
                if step[2] or step[3]:
                    break
            runs.append(trace)
        self.assertEqual(runs[0], runs[1])
        for algorithm in ("q_learning", "sarsa"):
            first, rows1 = train_stone_simulator(25, seed=4, algorithm=algorithm)
            second, rows2 = train_stone_simulator(25, seed=4, algorithm=algorithm)
            self.assertEqual((first.q, rows1), (second.q, rows2))
            other, _ = train_stone_simulator(25, seed=5, algorithm=algorithm)
            self.assertNotEqual(first.q, other.q)
            self.assertTrue(all(row["backend"] == "stone-macro-simulator" for row in rows1))
        self.assertEqual(evaluate_stone_policy(first, 30, 9), evaluate_stone_policy(second, 30, 9))

    def test_hidden_terrain_draws_follow_the_declared_probabilities(self):
        n = 20_000
        for config in (StoneSimConfig(), StoneSimConfig(q_s=0.2, q_d=0.6, p_v=0.3),
                       StoneSimConfig(q_s=0.8, q_d=0.9, p_v=0.7)):
            env = StoneSimulator(config)
            draws = []
            for seed in range(n):
                obs = env.reset(seed)
                terrain = env.hidden_terrain()
                self.assertEqual(obs["stone_visible"], terrain["visible_at_start"])
                draws.append(terrain)
            surface = [d for d in draws if d["surface_stone"]]
            self.assertFalse(any(d["visible_at_start"] for d in draws if not d["surface_stone"]))
            for name, count, total, p in (("q_s", len(surface), n, config.q_s),
                                          ("p_v", sum(d["visible_at_start"] for d in surface), len(surface), config.p_v),
                                          ("q_d", sum(d["dig_good"] for d in draws), n, config.q_d)):
                with self.subTest(config=config, probability=name):
                    # Within five binomial standard errors (a false alarm about once in 1.7 million).
                    self.assertLessEqual(abs(count / total - p), 5 * math.sqrt(p * (1 - p) / total))
            # Surface stone and good digging ground are drawn independently.
            good_given_surface = sum(d["dig_good"] for d in surface) / len(surface)
            self.assertLessEqual(abs(good_given_surface - config.q_d),
                                 5 * math.sqrt(config.q_d * (1 - config.q_d) / len(surface)))

    def test_dig_down_rates_and_face_sizes_follow_the_declared_values(self):
        env = sim(p_damage=0.0, p_death=0.0, p_mine=1.0)
        outcomes = {True: [], False: []}
        faces = []
        for seed in range(12_000):
            env.reset(seed)
            good = env.hidden_terrain()["dig_good"]
            obs, *_ = env.step(int(A.DIG_DOWN))
            outcomes[good].append(obs["nearby_stone"])
            if obs["nearby_stone"]:
                size = 0
                while obs["nearby_stone"]:
                    obs, _, done, truncated, _ = env.step(int(A.MINE_STONE))
                    size += 1
                    if done or truncated:
                        break
                faces.append(size)
        for good, p in ((True, 0.8), (False, 0.1)):
            rate = statistics.mean(outcomes[good])
            self.assertLessEqual(abs(rate - p), 5 * math.sqrt(p * (1 - p) / len(outcomes[good])))
        self.assertEqual(set(faces), set(range(1, 7)))
        for size in range(1, 7):
            share = faces.count(size) / len(faces)
            self.assertLessEqual(abs(share - 1 / 6), 5 * math.sqrt((1 / 6) * (5 / 6) / len(faces)))

    def test_training_uses_the_live_agent_reward_and_linear_schedule(self):
        agent, rows = train_stone_simulator(11, seed=1)
        self.assertIsInstance(agent, ResourceAgent)
        self.assertIs(agent.stage, STONE_STAGE)
        self.assertEqual([round(row["epsilon"], 12) for row in rows], [round(0.6 - 0.05 * i, 12) for i in range(11)])
        # The live session's formula: epsilon + (epsilon_end - epsilon) * (episode - 1) / max(1, episodes - 1).
        for episode in range(1, 12):
            self.assertEqual(linear_epsilon(episode - 1, 11, 0.6, 0.1), 0.6 + (0.1 - 0.6) * (episode - 1) / max(1, 10))
        self.assertEqual(linear_epsilon(0, 1, 0.6, 0.1), 0.6)
        self.assertTrue(all(row["condition"] == STONE_CONDITION and row["steps"] <= 30 for row in rows))
        self.assertTrue(all(set(row) >= {"surface_stone", "visible_at_start", "dig_good", "outside_reference_choices"}
                            for row in rows))
        self.assertTrue(all(len(state) == 5 for state in agent.q))
        # Every simulator reward is the live stone reward of its own observations.
        env = StoneSimulator()
        obs = env.reset(8)
        for action in (1, 2, 2, 0, 2, 3, 4):
            following, reward, terminated, _, info = env.step(action)
            expected, _ = stone_reward_transition(obs, following, success=info["success"], death=info["death"])
            self.assertEqual(reward, expected)
            if terminated:
                break
            obs = following

    def test_training_and_configuration_reject_invalid_values(self):
        for kwargs in ({"episodes": 0}, {"episodes": True}, {"episodes": 2.5}, {"algorithm": "random"},
                       {"epsilon_start": 0.1, "epsilon_end": 0.6}, {"epsilon_start": float("nan")},
                       {"gamma": 1}, {"alpha": 0}):
            episodes = kwargs.pop("episodes", 2)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                train_stone_simulator(episodes, **kwargs)
        with self.assertRaises(ValueError):
            train_stone_simulator(2, gamma=0.9, config=StoneSimConfig(gamma=0.97))
        with self.assertRaises(TypeError):
            StoneSimulator({"max_steps": 30})
        for kwargs in ({"max_steps": 0}, {"max_steps": True}, {"q_s": 1.5}, {"p_nav": float("nan")},
                       {"dig_good": -0.1}, {"p_damage": 0.9, "p_death": 0.2}, {"damage": 0}, {"damage": 21},
                       {"face_max": 0}, {"gamma": 1}, {"reward_mode": "other"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                StoneSimConfig(**kwargs)
        with self.assertRaises(ValueError):
            StoneSimulator().reset(1.5)
        env = StoneSimulator()
        env.reset(1)
        for bad in (5, -1, True, "0"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                env.step(bad)
        with self.assertRaises(ValueError):
            evaluate_stone_policy(ResourceAgent(), 5, 0)

    def test_a_short_training_run_beats_random_play_in_the_central_corner(self):
        # SIMULATOR ONLY. 40 training episodes (the design's estimate of N), 30-decision limit, default config.
        random_rates = [evaluate_stone_policy(ResourceAgent("random", seed=seed, stage=STONE_STAGE), 200,
                                              1_000_000 + 1000 * seed)["success_rate"] for seed in range(5)]
        for algorithm in ("q_learning", "sarsa"):
            rates = []
            for seed in range(5):
                agent, _ = train_stone_simulator(40, seed=seed, algorithm=algorithm)
                rates.append(evaluate_stone_policy(agent, 200, 1_000_000 + 1000 * seed)["success_rate"])
            with self.subTest(algorithm=algorithm):
                self.assertGreater(statistics.mean(rates), statistics.mean(random_rates) + 0.2)


if __name__ == "__main__":
    unittest.main()
