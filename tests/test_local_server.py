"""Checks for the server manager; no Minecraft download or EULA is required."""

from pathlib import Path
import json
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

from infrastructure import server


class ManagedServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.root_patch = patch.object(server, "ROOT", self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.temporary.cleanup()

    def test_offline_uuid_matches_known_minecraft_uuid(self):
        self.assertEqual(server.offline_uuid("Notch"),
                         "b50ad385-829d-3141-a216-7e7d7539ba7f")

    def test_profiles_are_separate_local_survival_worlds(self):
        arena = server.prepare_server("arena")
        survival = server.prepare_server("survival")
        self.assertNotEqual(arena["directory"], survival["directory"])
        self.assertNotEqual(arena["port"], survival["port"])
        for profile in (arena, survival):
            props = server._properties(profile["directory"] / "server.properties")
            self.assertEqual(props["server-ip"], "127.0.0.1")
            self.assertEqual(props["gamemode"], "survival")
            self.assertEqual(props["level-seed"], "")
            self.assertEqual(props["enable-rcon"], "false")
            self.assertEqual(props["enable-command-block"], "false")
            self.assertEqual(server._properties(profile["directory"] / "eula.txt")["eula"], "false")
        ops = json.loads((arena["directory"] / "ops.json").read_text())
        self.assertEqual(ops[0]["name"], "RLCourseBot")
        self.assertEqual(ops[0]["uuid"], server.offline_uuid("RLCourseBot"))
        self.assertEqual(json.loads((survival["directory"] / "ops.json").read_text()), [])

    def test_repeated_prepare_preserves_existing_world_and_extra_settings(self):
        profile = server.prepare_server()
        directory = profile["directory"]
        world = directory / profile["level_name"]
        world.mkdir()
        (world / "level.dat").write_bytes(b"sentinel world data")
        with (directory / "server.properties").open("a") as stream:
            stream.write("motd=Custom local description\n")
        server.prepare_server()
        self.assertEqual((world / "level.dat").read_bytes(), b"sentinel world data")
        self.assertEqual(server._properties(directory / "server.properties")["motd"],
                         "Custom local description")

    def test_foreign_nonempty_directory_is_left_untouched(self):
        directory = self.root / ".local" / "servers" / "arena"
        directory.mkdir(parents=True)
        (directory / "level.dat").write_bytes(b"other world")
        with self.assertRaisesRegex(server.ServerError, "unmarked"):
            server.prepare_server()
        self.assertEqual([path.name for path in directory.iterdir()], ["level.dat"])

    def test_incompatible_settings_are_not_rewritten(self):
        profile = server.prepare_server()
        properties = profile["directory"] / "server.properties"
        changed = properties.read_text().replace("server-ip=127.0.0.1", "server-ip=0.0.0.0")
        properties.write_text(changed)
        with self.assertRaisesRegex(server.ServerError, "server-ip"):
            server.prepare_server()
        self.assertEqual(properties.read_text(), changed)

    def test_survival_operator_privileges_are_rejected(self):
        profile = server.prepare_server("survival")
        ops = profile["directory"] / "ops.json"
        ops.write_text(json.dumps(server._expected_ops("arena")))
        with self.assertRaisesRegex(server.ServerError, "operator list"):
            server.prepare_server("survival")

    def test_consent_is_required_and_applies_to_existing_and_new_profiles(self):
        profile = server.prepare_server()
        with patch.object(server.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(server.ServerError, "accept-eula"):
                with server.running_server():
                    self.fail("Unaccepted EULA allowed startup")
            popen.assert_not_called()
        server.accept_eula()  # Isolated temporary test folder, never a real agreement.
        self.assertTrue(server.eula_accepted())
        self.assertEqual(server._properties(profile["directory"] / "eula.txt")["eula"], "true")
        survival = server.prepare_server("survival")
        self.assertTrue(survival["eula_accepted"])
        self.assertEqual(server._properties(survival["directory"] / "eula.txt")["eula"], "true")

    def test_server_command_uses_pinned_project_local_runtime(self):
        java = self.root / ".local" / "runtime" / "java" / "bin" / "java.exe"
        jar = self.root / ".local" / "server" / "server-1.21.4.jar"
        for file in (java, jar):
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b"test placeholder")
        command = server.server_command()
        self.assertEqual(command, [str(java), "-Xms512M", "-Xmx2048M", "-jar", str(jar), "nogui"])
        (self.root / ".local" / "machine.json").write_text('{"server_memory_mb": 4096}')
        self.assertIn("-Xmx4096M", server.server_command())

    def test_machine_heap_override_is_local_optional_and_validated(self):
        self.assertEqual(server.server_memory_mb(), 2048)
        configuration = self.root / ".local" / "machine.json"
        configuration.parent.mkdir(parents=True)
        for memory in (1024, 4096, 32768):
            with self.subTest(valid_memory=memory):
                configuration.write_text(json.dumps({"server_memory_mb": memory}))
                self.assertEqual(server.server_memory_mb(), memory)
        configuration.write_text("{}")
        self.assertEqual(server.server_memory_mb(), 2048)
        for memory in (True, False, 0, 1023, 32769, 4096.0, "4096", None):
            with self.subTest(invalid_memory=memory):
                configuration.write_text(json.dumps({"server_memory_mb": memory}))
                with self.assertRaisesRegex(server.ServerError, "integer from 1024 to 32768"):
                    server.server_memory_mb()
        for content in ("not json", "[]", "null"):
            with self.subTest(invalid_configuration=content):
                configuration.write_text(content)
                with self.assertRaises(server.ServerError):
                    server.server_memory_mb()

    def test_existing_port_is_rejected_without_touching_its_process(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            with self.assertRaisesRegex(server.ServerError, "already in use"):
                server._check_port(occupied.getsockname()[1])

    def _fake_process_command(self, *, ready: bool = True, exit_code: int | None = None):
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
        script.write_text(code)
        return [sys.executable, "-u", str(script)]

    def test_ready_server_is_saved_and_stopped_on_context_exception(self):
        server.prepare_server()
        server.accept_eula()
        command = self._fake_process_command()
        with patch.object(server, "server_command", return_value=command), \
                patch.object(server, "_check_port"):
            with self.assertRaisesRegex(RuntimeError, "task failed"):
                with server.running_server(timeout=5) as info:
                    self.assertIsNone(info["process"].poll())
                    self.assertTrue(info["log_file"].is_file())
                    raise RuntimeError("task failed")
        self.assertEqual(info["process"].returncode, 0)
        self.assertEqual((info["directory"] / "saved-by-fake-server").read_text(), "saved")
        self.assertIn("Saving worlds", info["log_file"].read_text())

    def test_startup_timeout_still_stops_owned_process(self):
        profile = server.prepare_server()
        server.accept_eula()
        command = self._fake_process_command(ready=False)
        with patch.object(server, "server_command", return_value=command), \
                patch.object(server, "_check_port"):
            with self.assertRaisesRegex(server.ServerError, "did not become ready"):
                with server.running_server(timeout=0.3):
                    self.fail("Unready process was yielded")
        self.assertEqual((profile["directory"] / "saved-by-fake-server").read_text(), "saved")

    def test_startup_exit_reports_log_and_never_yields(self):
        server.prepare_server()
        server.accept_eula()
        command = self._fake_process_command(exit_code=7)
        with patch.object(server, "server_command", return_value=command), \
                patch.object(server, "_check_port"):
            with self.assertRaisesRegex(server.ServerError, "exited before"):
                with server.running_server(timeout=5):
                    self.fail("Failed process was yielded")


if __name__ == "__main__":
    unittest.main()
