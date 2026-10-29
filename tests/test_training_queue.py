import json
import pytest

pytest.importorskip('msvcrt')  # the local queue uses Windows process/lock APIs
from scripts import training_queue as Q


def test_state_replace_recovers_from_transient_windows_lock(monkeypatch, tmp_path):
    path = tmp_path/'state.json'
    path.write_text('{"old": true}')
    replace = Q.os.replace
    attempts = []

    def locked_twice(src, dst):
        attempts.append(1)
        assert json.loads(path.read_text()) == {'old': True}
        if len(attempts) < 3:
            raise PermissionError('file temporarily locked')
        replace(src, dst)

    monkeypatch.setattr(Q.os, 'replace', locked_twice)
    monkeypatch.setattr(Q.time, 'sleep', lambda _: None)
    assert Q.save(path, {'jobs': {'existing': 'running'}})
    assert len(attempts) == 3
    assert json.loads(path.read_text())['jobs']['existing'] == 'running'


def test_persistent_lock_preserves_state_and_next_save_can_recover(monkeypatch, tmp_path):
    path = tmp_path/'state.json'
    path.write_text('{"old": true}')
    replace = Q.os.replace

    def locked(src, dst):
        raise PermissionError('file locked')

    monkeypatch.setattr(Q.os, 'replace', locked)
    monkeypatch.setattr(Q.time, 'sleep', lambda _: None)
    assert not Q.save(path, {'jobs': {'existing': 'running'}})
    assert json.loads(path.read_text()) == {'old': True}
    monkeypatch.setattr(Q.os, 'replace', replace)
    assert Q.save(path, {'jobs': {'existing': 'completed'}})
    assert json.loads(path.read_text())['jobs']['existing'] == 'completed'


def test_jobs_start_from_their_own_checkpoint_and_seeds_keep_it(monkeypatch, tmp_path):
    assert Q.init_checkpoint({}, 'x/latest.pt') == 'results/final_policy.pt'
    assert Q.init_checkpoint({'init': 'runs/a/best.pt'}, 'x/latest.pt') == 'runs/a/best.pt'
    assert Q.init_checkpoint({'init': 'runs/a/best.pt', 'resume': True}, 'x/latest.pt') == 'x/latest.pt'
    monkeypatch.setattr(Q, 'RUNS', tmp_path)
    (tmp_path/'job').mkdir()
    (tmp_path/'job'/'summary.json').write_text(json.dumps(
        {'completed': True, 'best': {'validation_per_level': {str(i): {'success': 0.9} for i in range(6)}}}))
    spec = Q.next_seed({'jobs': {}}, {'jobs': [{'name': 'job', 'flags': ['--x'], 'init': 'runs/a/best.pt'}]})
    assert spec['name'] == 'job-s1' and spec['init'] == 'runs/a/best.pt' and spec['flags'] == ['--x', '--seed', '1']
