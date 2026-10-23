"""Fresh-world lifecycle checks using fake children, without Minecraft startup."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from infrastructure import resource_server, server


class FreshResourceServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.run = self.root / "runs" / "resource-test"
        for module in (resource_server, server):
            root_patch = patch.object(module, "ROOT", self.root)
            root_patch.start()
            self.addCleanup(root_patch.stop)
        consent = self.root / ".local" / "minecraft-eula-consent.json"
        consent.parent.mkdir()
        consent.write_text(json.dumps({"schema": 1, "accepted": True,
                                       "url": server.EULA_URL}), encoding="utf-8")

    def _fake_command(self, *, ready=True, exit_code=None):
        script = self.root / "fake_server.py"
        if exit_code is not None:
            code = f"import sys\nprint('Startup failed', flush=True)\nsys.exit({exit_code})\n"
        else:
            code = "import sys\nfrom pathlib import Path\n"
            if ready:
                code += "print('[Server thread/INFO]: Done (0.1s)! For help, type help', flush=True)\n"
            code += (
                "for line in sys.stdin:\n"
                "    if line.strip() == 'stop':\n"
                "        Path('saved-by-fake-server').write_text('saved')\n"
                "        print('Saving worlds', flush=True)\n"
                "        break\n"
            )
        script.write_text(code, encoding="utf-8")
        return [sys.executable, "-u", str(script)]

    def test_new_natural_survival_configuration_and_owned_shutdown(self):
        with patch.object(server, "server_command", return_value=self._fake_command()) as command, \
                patch.object(server, "_check_port") as port:
            with resource_server.fresh_resource_server(
                    world_seed=-123456, run_directory=self.run, episode=1) as info:
                self.assertEqual(info["directory"], self.run / "worlds" / "episode-0001")
                self.assertEqual(info["world_seed"], -123456)
                self.assertEqual(info["port"], 25567)
                props = server._properties(info["directory"] / "server.properties")
                self.assertEqual(props["server-ip"], "127.0.0.1")
                self.assertEqual(props["level-seed"], "-123456")
                self.assertEqual(props["level-type"], "minecraft:normal")
                self.assertEqual(props["level-name"], "world")
                self.assertEqual(props["gamemode"], "survival")
                self.assertEqual(props["difficulty"], "normal")
                self.assertEqual(props["spawn-monsters"], "true")
                self.assertEqual(props["enable-rcon"], "false")
                self.assertEqual(props["enable-command-block"], "false")
                self.assertEqual(props["initial-enabled-packs"], "vanilla")
                self.assertEqual(json.loads((info["directory"] / "ops.json").read_text()), [])
                self.assertFalse((info["directory"] / "world").exists())
                self.assertEqual(server._properties(info["directory"] / "eula.txt")["eula"], "true")
                marker = json.loads((info["directory"] / server.MARKER_NAME).read_text())
                self.assertTrue(marker["fresh_world"])
                self.assertEqual(marker["episode"], 1)
            command.assert_called_once_with("survival")
            port.assert_called_once_with(25567)
        self.assertEqual(info["process"].returncode, 0)
        self.assertEqual((info["directory"] / "saved-by-fake-server").read_text(), "saved")
        self.assertIn("Saving worlds", info["log_file"].read_text())

    def test_existing_episode_is_never_reused_or_overwritten(self):
        for empty in (True, False):
            with self.subTest(empty=empty):
                episode = 1 if empty else 2
                directory = self.run / "worlds" / f"episode-{episode:04d}"
                directory.mkdir(parents=True)
                if not empty:
                    (directory / "world").mkdir()
                    (directory / "world" / "level.dat").write_bytes(b"precious evidence")
                before = sorted(path.relative_to(directory) for path in directory.rglob("*"))
                with patch.object(resource_server.subprocess, "Popen") as popen:
                    with self.assertRaisesRegex(server.ServerError, "already exists"):
                        with resource_server.fresh_resource_server(
                                world_seed=1, run_directory=self.run, episode=episode):
                            self.fail("Existing world yielded")
                    popen.assert_not_called()
                self.assertEqual(before, sorted(path.relative_to(directory) for path in directory.rglob("*")))
                if not empty:
                    self.assertEqual((directory / "world" / "level.dat").read_bytes(), b"precious evidence")

    def test_invalid_paths_and_numeric_inputs_do_not_create_run(self):
        inputs = [
            {"run_directory": self.root / "outside"},
            {"run_directory": self.root / "runs"},
            {"run_directory": self.run / ".." / ".." / "outside"},
            {"episode": 0}, {"episode": True}, {"episode": 1.5},
            {"world_seed": True}, {"world_seed": 2**63},
            {"world_seed": -(2**63) - 1}, {"world_seed": "5"},
            {"timeout": 0}, {"timeout": float("inf")}, {"timeout": True},
        ]
        for override in inputs:
            arguments = {"run_directory": self.run, "episode": 1, "world_seed": 123}
            arguments.update(override)
            with self.subTest(override=override):
                with patch.object(resource_server.subprocess, "Popen") as popen:
                    with self.assertRaises((ValueError, server.ServerError)):
                        with resource_server.fresh_resource_server(**arguments):
                            self.fail("Invalid settings yielded")
                    popen.assert_not_called()
                self.assertFalse(self.run.exists())

    def test_linked_worlds_directory_is_rejected(self):
        worlds = self.run / "worlds"
        with patch.object(resource_server, "_is_link", side_effect=lambda path: path == worlds):
            with self.assertRaisesRegex(server.ServerError, "linked world"):
                with resource_server.fresh_resource_server(
                        world_seed=1, run_directory=self.run, episode=1):
                    self.fail("Linked path yielded")
        self.assertFalse(self.run.exists())

    def test_port_conflict_is_checked_before_creating_any_episode_files(self):
        with patch.object(server, "server_command", return_value=["unused"]), \
                patch.object(server, "_check_port", side_effect=server.ServerError("already in use")), \
                patch.object(resource_server.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(server.ServerError, "already in use"):
                with resource_server.fresh_resource_server(
                        world_seed=1, run_directory=self.run, episode=1):
                    self.fail("Occupied port yielded")
            popen.assert_not_called()
        self.assertFalse(self.run.exists())

    def test_missing_eula_is_rejected_without_recording_acceptance_or_creating_world(self):
        consent = self.root / ".local" / "minecraft-eula-consent.json"
        consent.write_text('{"accepted": false}', encoding="utf-8")
        with patch.object(resource_server.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(server.ServerError, "accept-eula"):
                with resource_server.fresh_resource_server(
                        world_seed=1, run_directory=self.run, episode=1):
                    self.fail("Missing acceptance yielded")
            popen.assert_not_called()
        self.assertEqual(consent.read_text(), '{"accepted": false}')
        self.assertFalse(self.run.exists())

    def test_context_error_saves_world_and_stops_only_owned_child(self):
        with patch.object(server, "server_command", return_value=self._fake_command()), \
                patch.object(server, "_check_port"):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with resource_server.fresh_resource_server(
                        world_seed=7, run_directory=self.run, episode=1) as info:
                    raise RuntimeError("training failed")
        self.assertEqual(info["process"].returncode, 0)
        self.assertEqual((info["directory"] / "saved-by-fake-server").read_text(), "saved")

    def test_startup_timeout_preserves_logs_and_stops_owned_child(self):
        with patch.object(server, "server_command", return_value=self._fake_command(ready=False)), \
                patch.object(server, "_check_port"):
            with self.assertRaisesRegex(server.ServerError, "did not become ready"):
                with resource_server.fresh_resource_server(
                        world_seed=1, run_directory=self.run, episode=1, timeout=0.3):
                    self.fail("Unready child yielded")
        directory = self.run / "worlds" / "episode-0001"
        self.assertEqual((directory / "saved-by-fake-server").read_text(), "saved")
        self.assertTrue((directory / "launcher.log").exists())

    def test_startup_failure_never_yields_and_reserves_failed_episode_for_diagnosis(self):
        with patch.object(server, "server_command", return_value=self._fake_command(exit_code=7)), \
                patch.object(server, "_check_port"):
            with self.assertRaisesRegex(server.ServerError, "exited before"):
                with resource_server.fresh_resource_server(
                        world_seed=1, run_directory=self.run, episode=1, timeout=5):
                    self.fail("Failed child yielded")
        directory = self.run / "worlds" / "episode-0001"
        self.assertIn("Startup failed", (directory / "launcher.log").read_text())
        self.assertTrue((directory / server.MARKER_NAME).exists())


if __name__ == "__main__":
    unittest.main()
