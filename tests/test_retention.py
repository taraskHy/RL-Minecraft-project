"""Retained evidence must be a byte-identical, digest-checked copy of the small result files only."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from infrastructure import retention


def make_run(root: Path, name: str, episodes: int = 1) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps({"mode": "online_training"}), encoding="utf-8")
    (run / "initial_policy.json").write_text("{\"q\": {}}", encoding="utf-8")
    (run / "policy.json").write_text("{\"q\": {\"s\": [1]}}", encoding="utf-8")
    (run / "episodes.csv").write_text("episode,success\n1,True\n", encoding="utf-8")
    (run / "summary.json").write_text("{}", encoding="utf-8")
    (run / "simulator_training.csv").write_text("episode\n", encoding="utf-8")
    for index in range(1, episodes + 1):
        episode = run / f"episode-{index:04d}"
        episode.mkdir()
        (episode / "start.json").write_text("{\"observation\": {}}", encoding="utf-8")
        (episode / "transitions.jsonl").write_text("{\"step\": 1}\n", encoding="utf-8")
        (episode / "steps.csv").write_text("step\n1\n", encoding="utf-8")
        (episode / "outcome.json").write_text("{\"status\": \"completed\"}", encoding="utf-8")
        if index == episodes and episodes > 1:
            (episode / "session_error.json").write_text("{\"status\": \"interrupted\"}", encoding="utf-8")
        world = run / "worlds" / episode.name
        (world / "logs").mkdir(parents=True)
        (world / "server.properties").write_text("level-seed=1\n", encoding="utf-8")
        (world / "ops.json").write_text("[]", encoding="utf-8")
        (world / "logs" / "latest.log").write_text("".join(f"line {i}\n" for i in range(300)), encoding="utf-8")
        (world / "world").mkdir()
        (world / "world" / "level.dat").write_bytes(b"\x00" * 64)
        (world / "server.jar").write_bytes(b"\x00" * 16)
        (world / "eula.txt").write_text("eula=true\n", encoding="utf-8")
        (world / "usercache.json").write_text("[]", encoding="utf-8")
        (world / "minecraft-eula-consent.json").write_text("{}", encoding="utf-8")  # decoy: must never be copied
    return run


class RetentionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = make_run(self.root / "runs", "pc-q-dev", episodes=2)

    def test_copies_only_compact_files_with_matching_digests(self):
        out = self.root / "results" / "pc_dev"
        manifest = retention.retain([self.run], out, "DEVELOPMENT EVIDENCE; label under test", log_tail_lines=50)
        self.assertEqual(manifest["label"], "DEVELOPMENT EVIDENCE; label under test")
        expected = {f"pc-q-dev/{name}" for name in ("metadata.json", "initial_policy.json", "policy.json", "episodes.csv",
                                                     "summary.json", "simulator_training.csv")}
        for episode in ("episode-0001", "episode-0002"):
            expected |= {f"pc-q-dev/{episode}/{name}" for name in ("start.json", "transitions.jsonl", "steps.csv", "outcome.json")}
            expected |= {f"pc-q-dev/worlds/{episode}/server.properties", f"pc-q-dev/worlds/{episode}/ops.json",
                         f"pc-q-dev/worlds/{episode}/server_tail.log"}
        expected.add("pc-q-dev/episode-0002/session_error.json")
        self.assertEqual(set(manifest["sha256"]), expected)
        self.assertEqual(manifest["files"], len(expected))
        for relative, digest in manifest["sha256"].items():
            self.assertEqual(hashlib.sha256((out / relative).read_bytes()).hexdigest(), digest)
        self.assertEqual((out / "pc-q-dev/policy.json").read_bytes(), (self.run / "policy.json").read_bytes())
        tail = (out / "pc-q-dev/worlds/episode-0001/server_tail.log").read_text(encoding="utf-8").splitlines()
        self.assertEqual((len(tail), tail[0], tail[-1]), (50, "line 250", "line 299"))
        # Worlds, jars, EULA files, user caches, and anything named like the consent file are never retained.
        for forbidden in ("world", "server.jar", "eula.txt", "usercache.json", "minecraft-eula-consent.json", "logs"):
            self.assertFalse((out / "pc-q-dev/worlds/episode-0001" / forbidden).exists(), forbidden)
        self.assertEqual(list(out.rglob("minecraft-eula-consent.json")), [])
        saved = json.loads((out / "retention.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["sha256"], manifest["sha256"])
        self.assertEqual(retention.verify(out), [])
        (out / "pc-q-dev/summary.json").write_text("{\"edited\": true}", encoding="utf-8")
        (out / "pc-q-dev/episode-0001/steps.csv").unlink()
        self.assertEqual(retention.verify(out), ["pc-q-dev/summary.json", "pc-q-dev/episode-0001/steps.csv"])

    def test_refuses_overwrites_duplicates_and_non_runs(self):
        out = self.root / "results" / "taken"
        out.mkdir(parents=True)
        (out / "README.md").write_text("existing", encoding="utf-8")
        with self.assertRaises(ValueError):
            retention.retain([self.run], out, "label")
        self.assertEqual((out / "README.md").read_text(encoding="utf-8"), "existing")
        with self.assertRaises(ValueError):
            retention.retain([self.run, self.root / "elsewhere" / "pc-q-dev"], self.root / "results" / "dup", "label")
        # Names that differ only in case collide on a case-insensitive file system.
        other = make_run(self.root / "other", "PC-Q-DEV")
        with self.assertRaises(ValueError):
            retention.retain([self.run, other], self.root / "results" / "case", "label")
        with self.assertRaises(ValueError):
            retention.retain([self.root / "runs"], self.root / "results" / "not-a-run", "label")
        with self.assertRaises(ValueError):
            retention.retain([self.run], self.root / "results" / "unlabeled", "  ")
        with self.assertRaises(ValueError):
            retention.retain([], self.root / "results" / "empty", "label")
        for name in ("dup", "case"):
            self.assertFalse((self.root / "results" / name).exists())

    def test_command_line_reports_errors_without_writing(self):
        out = self.root / "results" / "cli"
        self.assertEqual(retention.main([str(self.run), "--out", str(out), "--label", "DEVELOPMENT EVIDENCE"]), 0)
        self.assertTrue((out / "retention.json").is_file())
        self.assertEqual(retention.main([str(self.run), "--out", str(out), "--label", "again"]), 2)
        self.assertEqual(retention.main(["--verify", "--out", str(out)]), 0)
        (out / "pc-q-dev/policy.json").write_text("{}", encoding="utf-8")
        self.assertEqual(retention.main(["--verify", "--out", str(out)]), 1)
        for bad in (["--out", str(out)], [str(self.run), "--out", str(out)], ["--verify", str(self.run), "--out", str(out)],
                    ["--verify", "--out", str(out), "--label", "x"]):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                retention.main(bad)


    def test_a_schedule_folder_keeps_only_its_plan_and_record(self):
        schedule = self.root / "runs" / "px"
        (schedule / "logs").mkdir(parents=True)
        (schedule / "plan.json").write_text('{"name": "px"}', encoding="utf-8")
        (schedule / "schedule.json").write_text('{"plan_name": "px"}', encoding="utf-8")
        (schedule / "logs" / "px-a.log").write_text("long log\n", encoding="utf-8")
        nested = make_run(schedule, "px-a")
        out = self.root / "results" / "schedule"
        manifest = retention.retain([schedule, nested], out, "SCHEDULE EVIDENCE")
        for key in ("px/plan.json", "px/schedule.json", "px-a/metadata.json"):
            self.assertIn(key, manifest["sha256"])
        self.assertFalse(any(key.startswith("px/logs") or key.startswith("px/px-a") for key in manifest["sha256"]))
        self.assertEqual(retention.verify(out), [])


    def test_a_schedule_folder_brings_every_run_it_recorded(self):
        schedule = self.root / "runs" / "pq"
        schedule.mkdir(parents=True)
        (schedule / "plan.json").write_text('{"name": "pq"}', encoding="utf-8")
        units = [{"name": "pq-a", "parts": [{"folder": "pq-a"}, {"folder": "pq-a-part1"}]}]
        (schedule / "schedule.json").write_text(json.dumps({"units": units}), encoding="utf-8")
        make_run(schedule, "pq-a")
        make_run(schedule, "pq-a-part1")
        out = self.root / "results" / "expanded"
        manifest = retention.retain([schedule], out, "SCHEDULE EVIDENCE")
        for key in ("pq/plan.json", "pq/schedule.json", "pq-a/metadata.json", "pq-a-part1/episode-0001/outcome.json"):
            self.assertIn(key, manifest["sha256"])
        self.assertEqual(retention.verify(out), [])


if __name__ == "__main__":
    unittest.main()
