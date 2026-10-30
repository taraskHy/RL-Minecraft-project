"""The game's collision order at wall corners (``physics.simulate(..., larger_first=True)``) and the neo planner's
``corner_limit``: prismarine-physics moves along x before z, the game along the larger component first, and past a
wall's corner the two can end apart; more than 0.25 blocks apart, a server logs "moved wrongly" and sends the bot
back if its box started the tick clear of every block."""
import gzip
import json
import math
import unittest
from unittest import mock

import numpy as np
import torch

from parkour import courses as C
from parkour import neo_assist as NA
from parkour import physics
from parkour.env import YAW_STEPS_DEG, CoursePool, action_to_controls
from parkour.neo_assist import AssistConfig, Rules, config_meta, corner_gap, courses_from_file, plan_once
from parkour.paths import ROOT

NO_KEYS = [1, 1, 0, 0, 3]  # move none, strafe none, no jump, no sprint, no turn


def corner_course() -> C.Course:
    """A wall cell at (10, 5, 10) beside the corner of a 3x3 platform whose top is y = 5."""
    platforms = np.array([[11, 13, 11, 13, 4], [11, 13, 20, 22, 4]], dtype=np.int64)
    return C.Course(platforms, np.array([12.5, 5.0, 12.5]), 0.0, extra=np.array([[10, 5, 10]], dtype=np.int64))


def standing(pos, vel) -> dict:
    return {"course": 0, "cp": 0, "pos": list(pos), "vel": list(vel), "yaw": 0.0, "on_ground": True, "jump_ticks": 0,
            "collided_h": False, "collided_v": False, "last_action": NO_KEYS}


# On the ground just south of the wall's corner (box 0.02 from its z face, overlapping its x range by 0.05), moving
# mostly north (-z) and a little east: x then z clears the wall, z then x is stopped by it.
PAST_THE_CORNER = standing((11.25, 5.0, 11.32), (0.07, -0.0784, -0.29))


class CollisionOrder(unittest.TestCase):
    def setUp(self):
        self.pool = CoursePool([corner_course()], "cpu", C.GRID)

    def step(self, arrival, larger_first):
        s = NA._state_rows([arrival], "cpu")
        a = torch.tensor([NO_KEYS])
        return physics.simulate(self.pool.world, torch.tensor([0]), s, action_to_controls(a), larger_first=larger_first)

    def test_the_orders_part_only_past_a_corner(self):
        xz, zx = self.step(PAST_THE_CORNER, False), self.step(PAST_THE_CORNER, True)
        self.assertAlmostEqual(float(xz.pos[0, 2]), 11.03, places=9)  # x first: past the corner
        self.assertAlmostEqual(float(zx.pos[0, 2]), 11.30, places=9)  # z first: stopped by the wall's z face
        self.assertAlmostEqual(float(xz.pos[0, 0]), float(zx.pos[0, 0]), places=12)
        self.assertTrue(bool(zx.collided_h[0]) and not bool(xz.collided_h[0]))
        for vel in ((0.29, -0.0784, -0.07), (0.07, -0.0784, 0.29), (0.0, -0.0784, -0.29)):  # x larger, or no corner
            a, b = self.step(dict(PAST_THE_CORNER, vel=list(vel)), False), self.step(dict(PAST_THE_CORNER, vel=list(vel)), True)
            self.assertTrue(torch.equal(a.pos, b.pos), vel)

    def test_corner_gap_and_the_planner_drop(self):
        self.assertAlmostEqual(corner_gap(self.pool, PAST_THE_CORNER, [NO_KEYS], "cpu"), 0.27, places=9)
        self.assertEqual(corner_gap(self.pool, standing((12.5, 5.0, 12.5), (0.0, -0.0784, -0.29)), [NO_KEYS] * 3, "cpu"),
                         0.0)
        plan = np.array([NO_KEYS])
        calls = []

        def search(pool, rules, arrivals, device, cfg, seed, prior=None, horizon=32):  # lands on every call
            calls.append(seed)
            return [[plan]], [cfg.iterations]

        rules = Rules(self.pool)
        with mock.patch.object(NA, "search", search), mock.patch.object(NA, "neo_request", lambda *a: False):
            seq, rounds, restarts = plan_once(None, self.pool, rules, PAST_THE_CORNER, "takeoff", 0,
                                              AssistConfig(iterations=2), 32, "cpu")
            self.assertTrue(np.array_equal(seq, plan))  # off: the landing is played
            st = {}
            seq, rounds, restarts = plan_once(None, self.pool, rules, PAST_THE_CORNER, "takeoff", 0,
                                              AssistConfig(iterations=2, corner_limit=0.25), 32, "cpu", stats=st)
            self.assertEqual((seq, rounds, st), (None, 2, {"corner_dropped": 1}))  # an ordinary jump: no restart
            seq, _, _ = plan_once(None, self.pool, rules, PAST_THE_CORNER, "takeoff", 0,
                                  AssistConfig(iterations=2, corner_limit=0.3), 32, "cpu")
            self.assertIsNotNone(seq)  # 0.27 apart: under a limit of 0.3
        calls.clear()
        with mock.patch.object(NA, "search", search), mock.patch.object(NA, "neo_request", lambda *a: True):
            st = {}
            seq, rounds, restarts = plan_once(None, self.pool, rules, PAST_THE_CORNER, "takeoff", 0,
                                              AssistConfig(iterations=2, corner_limit=0.25, restarts=3), 32, "cpu",
                                              stats=st)
            self.assertEqual((seq, rounds, restarts, st, len(calls)), (None, 8, 3, {"corner_dropped": 4}, 4))
        self.assertNotIn("corner_limit", config_meta(AssistConfig()))
        self.assertEqual(config_meta(AssistConfig(corner_limit=0.25))["corner_limit"], 0.25)
        with self.assertRaises(ValueError):  # it acts per request: batch 1 only
            NA.evaluate_assisted(None, self.pool, "cpu", AssistConfig(corner_limit=0.25, batch=2))

    def test_the_two_moves_a_server_sent_back(self):
        # Real Minecraft, planner with plan_start, test courses 5 and 57: the server corrected the bot at tick 111 and
        # 141 ("moved wrongly"). Replayed from the offline bot's identical recording: x then z gives the recorded
        # position, the game's order ends 0.264 and 0.255 blocks away.
        d = ROOT / "results" / "games" / "single41-offline-planstart"
        cs = courses_from_file(json.loads((d / "courses.json").read_text())["courses"])
        eps = {e["index"]: e for e in (json.loads(x) for x in gzip.open(d / "episodes.jsonl.gz", "rt") if x.strip())}
        pool = CoursePool(cs, "cpu", C.GRID)
        steps = torch.tensor(YAW_STEPS_DEG, dtype=torch.float64) * (math.pi / 180)
        for i, tick, want in ((5, 111, 0.2637), (57, 141, 0.2548)):
            ep, init = eps[i], eps[i]["init"]
            s = physics.PlayerState.at(torch.tensor([init["pos"]], dtype=torch.float64),
                                       torch.tensor([init["yaw"]], dtype=torch.float64))
            s.vel = torch.tensor([init["vel"]], dtype=torch.float64)
            s.on_ground = torch.tensor([bool(init["onGround"])])
            s.jump_ticks = torch.tensor([int(init["jumpTicks"])])
            course = torch.tensor([i])
            for t in range(tick):
                a = torch.tensor([ep["actions"][t]])
                s.yaw = torch.remainder(s.yaw + steps[a[:, 4]] + math.pi, 2 * math.pi) - math.pi
                game = physics.simulate(pool.world, course, s, action_to_controls(a), larger_first=True)
                s = physics.simulate(pool.world, course, s, action_to_controls(a))
                np.testing.assert_allclose(s.pos[0].numpy(), ep["traj"][t][:3], atol=1e-9)
            gap = float(torch.linalg.norm(s.pos[0, [0, 2]] - game.pos[0, [0, 2]]))
            self.assertAlmostEqual(gap, want, places=4)
            self.assertGreater(gap, 0.25)


if __name__ == "__main__":
    unittest.main()
