"""The generated snapshot in HANDOFF.md: only its block is rewritten, and the committed one is current."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from parkour import handoff as H


class Snapshot(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def write_results(self):
        res = self.dir / "results"
        (res / "games" / "x-1").mkdir(parents=True)
        (res / "games" / "y-1").mkdir()
        (res / "games" / "y-1" / "summary.json").write_text(json.dumps(
            {"mode": "offline", "checkpoint": "C:\\pc\\repo\\runs\\a\\best.pt", "courses": 4, "success": 0.75}))
        (res / "sub").mkdir()
        (res / "sub" / "f.json").write_text("{}")
        (res / "a.json").write_text("{}")
        (res / "b.json").write_text("{}")
        (res / "DATA.md").write_text("Files: `a.json` and the `x-*` games.\n")
        return res

    def test_the_repository_handoff_has_one_block(self):
        text = H.HANDOFF.read_text(encoding="utf-8")
        for marker in (H.BEGIN, H.DATA, H.END):
            self.assertEqual(text.count(marker), 1, marker)

    def test_committed_snapshot_matches_results(self):
        stored = H.stored_data(H.HANDOFF.read_text(encoding="utf-8"))
        self.assertEqual(stored, "\n".join(H.data_part()).strip(),
                         "results/ changed without `python -m parkour.handoff` (or `run.cmd report`)")

    def test_only_the_block_is_rewritten_and_twice_changes_nothing(self):
        res = self.write_results()
        p = self.dir / "HANDOFF.md"
        before, after = "# Title\n\nhand-written\n\n", "\n\n## Session log\n\n- entry\n"
        p.write_text(before + H.BEGIN + "\nstale text\n" + H.END + after)
        self.assertTrue(H.update(p, res))
        text = p.read_text()
        self.assertTrue(text.startswith(before + H.BEGIN))
        self.assertTrue(text.endswith(H.END + after))
        self.assertNotIn("stale text", text)
        self.assertFalse(H.update(p, res))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(H.check(p, res), 0)

    def test_line_endings_are_kept(self):
        res = self.write_results()
        p = self.dir / "HANDOFF.md"
        p.write_bytes(f"top\r\n{H.BEGIN}\r\n{H.END}\r\nbottom\r\n".encode())
        H.update(p, res)
        raw = p.read_bytes()
        self.assertEqual(raw.count(b"\n"), raw.count(b"\r\n"))

    def test_missing_markers_are_an_error(self):
        p = self.dir / "HANDOFF.md"
        p.write_text("no markers\n")
        with self.assertRaises(ValueError):
            H.update(p, self.write_results())

    def test_unindexed_names_what_the_data_index_misses(self):
        missing = H.unindexed(self.write_results())
        self.assertEqual(missing, ["`b.json`", "`games/y-1/`", "`sub/` (1 file)"])

    def test_checkpoint_paths_from_another_pc_are_shortened(self):
        self.assertEqual(H.short_path("C:\\pc\\repo\\runs\\a\\best.pt"), "runs/a/best.pt")
        self.assertEqual(H.short_path("results/final_policy.pt"), "results/final_policy.pt")
        rows = "\n".join(H.games_lines(self.write_results()))
        self.assertIn("| `y-1` | offline | `runs/a/best.pt` | 3/4 |", rows)


if __name__ == "__main__":
    unittest.main()
