"""Watch-session guards and readiness checks without launching Minecraft."""

from contextlib import ExitStack
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from infrastructure import watch
from minecraft_rl.environment import EnvConfig


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class WatchTests(unittest.TestCase):
    def test_invalid_viewer_or_timeout_cannot_start_server(self):
        bad_names = ["", "@a", "Alex\nstop", "Alice Bob", "x" * 17, "rlcoursebot", None]
        with patch.object(watch, "running_server") as start:
            for viewer in bad_names:
                with self.subTest(viewer=viewer), self.assertRaises(ValueError):
                    watch.run_watch(viewer=viewer, policy=Path("unused.json"))
            for timeout in (0, -1, 601, float("nan"), float("inf"), True, "30"):
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    watch.run_watch(viewer="Alex", policy=Path("unused.json"), join_timeout=timeout)
            start.assert_not_called()

    def test_missing_viewer_records_invalid_run_without_policy_actions(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as patches:
            root = Path(temporary)
            policy = root / "policy.json"
            policy.write_text("{}", encoding="utf-8")
            node = root / "node.exe"
            node.touch()
            log = root / "server.log"
            log.touch()
            server = {"port": 25565, "process": SimpleNamespace(poll=lambda: None), "log_file": log}
            clock = Clock()
            agent = MagicMock(algorithm="sarsa")
            env = MagicMock()
            env.__enter__.return_value = env
            env.connection_info = {"backend": "test"}
            env.reset.return_value = (0, 0, 3, 0, 0, 0)
            env.observe.return_value = {"state": [0, 0, 3, 0, 0, 0], "inventory": {}}
            env.players.return_value = {}
            manager = MagicMock()
            manager.__enter__.return_value = server
            patches.enter_context(patch.object(watch, "ROOT", root))
            patches.enter_context(patch.object(watch.TabularAgent, "load", return_value=agent))
            patches.enter_context(patch.object(watch, "load_env_config", return_value=EnvConfig()))
            patches.enter_context(patch.object(watch, "runtime_paths", return_value={"node": node}))
            start = patches.enter_context(patch.object(watch, "running_server", return_value=manager))
            patches.enter_context(patch.object(watch, "MinecraftEnv", return_value=env))
            patches.enter_context(patch.object(watch.time, "monotonic", clock.monotonic))
            patches.enter_context(patch.object(watch.time, "sleep", clock.sleep))
            patches.enter_context(patch("builtins.print"))
            with self.assertRaisesRegex(RuntimeError, "did not join"):
                watch.run_watch(viewer="Alex", policy=policy, join_timeout=1, once=True)
            start.assert_called_once_with("arena", mirror_output=True)
            agent.act.assert_not_called()
            env.step.assert_not_called()
            env.reset.assert_called_once_with(seed=1_000_000)
            env.__exit__.assert_called_once()
            manager.__exit__.assert_called_once()
            outcomes = list((root / "runs").glob("watch-*/episode-001/outcome.json"))
            self.assertEqual(len(outcomes), 1)
            outcome = json.loads(outcomes[0].read_text(encoding="utf-8"))
            self.assertEqual(outcome["status"], "invalid")
            self.assertIsNone(outcome["success"])
            self.assertEqual(outcome["recorded_steps"], 0)

    def test_spectator_mode_and_old_teleport_ack_are_not_readiness(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "server.log"
            log.write_text("[Server thread/INFO]: Teleported Alex to 2.500000, 66.000000, 4.500000\n", encoding="utf-8")
            server = {"process": SimpleNamespace(stdin=io.StringIO(), poll=lambda: None), "log_file": log}
            env = MagicMock()
            env.players.return_value = {"Alex": {"gamemode": 3}}
            clock = Clock()
            with patch.object(watch.time, "monotonic", clock.monotonic), patch.object(watch.time, "sleep", clock.sleep):
                with self.assertRaisesRegex(RuntimeError, "Could not verify"):
                    watch._position_viewer(env, server, "Alex")
            self.assertIn("gamemode spectator Alex\n", server["process"].stdin.getvalue())
            self.assertNotIn("op ", server["process"].stdin.getvalue())

    def test_new_teleport_ack_and_spectator_mode_allow_invisible_viewer(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "server.log"
            log.touch()

            class Console(io.StringIO):
                def flush(self):
                    with log.open("a", encoding="utf-8") as output:
                        output.write("[Server thread/INFO]: Teleported Alex to 2.500000, 66.000000, 4.500000\n")

            server = {"process": SimpleNamespace(stdin=Console(), poll=lambda: None), "log_file": log}
            env = MagicMock()
            # Spectators need no visible entity in the crafting bot's client.
            env.players.return_value = {"Alex": {"gamemode": 3}}
            watch._position_viewer(env, server, "Alex")
            self.assertEqual(server["process"].stdin.getvalue().count("tp Alex "), 1)

    def test_disconnect_during_countdown_stops_before_policy(self):
        server = {"process": SimpleNamespace(poll=lambda: None)}
        env = MagicMock()
        env.players.side_effect = [{"Alex": {"gamemode": 3}}, {}]
        with patch.object(watch.time, "sleep"), patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "disconnected"):
                watch._countdown(env, server, "Alex")


if __name__ == "__main__":
    unittest.main()
