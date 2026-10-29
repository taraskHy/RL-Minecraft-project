"""parkour.report rebuilds results/ from the training PC's runs/: elsewhere it must stop, not drop sections."""
import pytest

from parkour import report as R


def test_report_stops_without_the_training_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "RUNS", tmp_path / "runs")  # a fresh clone has no runs/ at all
    out = tmp_path / "results"
    with pytest.raises(SystemExit, match="parkour.handoff"):
        R.main(["--out", str(out)])
    assert not out.exists()
    (tmp_path / "runs").mkdir()  # nor does another PC's runs/ hold the main run
    with pytest.raises(SystemExit, match="main-s0"):
        R.main(["--out", str(out)])
    assert not out.exists()
    R.archive(tmp_path / "copy")  # and archiving tolerates a missing runs/
