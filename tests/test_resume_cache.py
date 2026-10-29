"""Interrupted evaluations must not duplicate workers, mix checkpoints or repeat rows."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from parkour import neo_assist as assist
from parkour.run_cache import cache_lock, write_json_atomic
from scripts import extra_preview as preview


@pytest.fixture
def small_run(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'best.pt'
    checkpoint.write_bytes(b'fixed checkpoint')
    monkeypatch.setattr(assist, 'ROOT', tmp_path)
    monkeypatch.setattr(assist, 'PARTS_DIR', tmp_path / 'parts')
    monkeypatch.setattr(assist, 'course_set', lambda *args: [SimpleNamespace(to_json=lambda i=i: {'id': i}) for i in range(2)])
    monkeypatch.setattr(assist, 'load_model', lambda *args: object())
    monkeypatch.setattr(assist, 'CoursePool', lambda courses, *args: courses)
    calls = []

    def evaluate(model, courses, *args, **kwargs):
        calls.append(kwargs['course_offset'])
        per_course = [dict(success=True, fell=False, neos=0, neos_reached=0,
                           neos_passed=0, stopped_at_neo=False) for _ in courses]
        return dict(per_course=per_course, events=[], plans={}, search_seconds=0)

    monkeypatch.setattr(assist, 'evaluate_assisted', evaluate)
    args = ['--checkpoint', str(checkpoint), '--device', 'cpu', '--name', 'sample',
            '--out', str(tmp_path / 'rows.jsonl')]
    return tmp_path, checkpoint, args, calls


def test_restart_after_append_does_not_duplicate_or_recompute(small_run, monkeypatch):
    root, _, args, calls = small_run
    for part in ('1/2', '2/2'):
        assist.main(args + ['--part', part])
    real_write = assist.write_json_atomic

    def interrupted(path, value):
        if path.name == 'merged.json':
            raise OSError('interrupted after append')
        real_write(path, value)

    monkeypatch.setattr(assist, 'write_json_atomic', interrupted)
    with pytest.raises(OSError, match='after append'):
        assist.main(args + ['--parts', '2'])
    assert len((root / 'rows.jsonl').read_text().splitlines()) == 1
    monkeypatch.setattr(assist, 'write_json_atomic', real_write)
    assist.main(args + ['--parts', '2'])
    assist.main(args + ['--parts', '2'])
    assert len((root / 'rows.jsonl').read_text().splitlines()) == 1
    assert calls == [0, 1]
    assist.main(args + ['--parts', '2', '--out', str(root / 'another.jsonl'),
                       '--details', str(root / 'another-details.json')])
    assert len((root / 'another.jsonl').read_text().splitlines()) == 1
    assert (root / 'another-details.json').exists()
    assert calls == [0, 1]
    with pytest.raises(SystemExit, match='other settings'):
        assist.main(args + ['--parts', '2', '--brake', '1'])
    (root / 'parts/sample/1of2.json').rename(root / 'parts/sample/1of2.saved')
    with pytest.raises(SystemExit, match='refusing to repeat'):
        assist.main(args + ['--parts', '2'])
    assert calls == [0, 1]


def test_parts_dir_holds_the_whole_split_run(small_run, monkeypatch):
    import subprocess
    root, _, args, calls = small_run
    other = root / 'elsewhere'
    monkeypatch.setattr(assist.time, 'sleep', lambda *args: None)
    monkeypatch.setattr(assist.torch, 'set_num_threads', lambda *args: None)  # (the parts' --threads 1)
    commands = []

    def here(cmd, **kwargs):  # each part in this process, from the command line the manager gives it
        commands.append(cmd)
        assist.main(cmd[3:])
        return SimpleNamespace(poll=lambda: 0, returncode=0)

    monkeypatch.setattr(subprocess, 'Popen', here)
    assist.main(args + ['--parts', '2', '--parts-dir', str(other)])
    assert calls == [0, 1]
    assert [Path(c[c.index('--parts-dir') + 1]) for c in commands] == [other, other]
    assert sorted(p.name for p in (other / 'sample').glob('*.json')) == ['1of2.json', '2of2.json', 'merged.json']
    assert not (root / 'parts').exists()
    assert len((root / 'rows.jsonl').read_text().splitlines()) == 1
    assist.main(args + ['--parts', '2', '--parts-dir', str(other)])  # all cached: the same row, not run again
    assert calls == [0, 1] and len(commands) == 2


def test_changed_weights_reject_existing_part_before_evaluation(small_run):
    _, checkpoint, args, calls = small_run
    assist.main(args + ['--part', '1/2'])
    checkpoint.write_bytes(b'new weights, same filename')
    with pytest.raises(SystemExit, match='other settings'):
        assist.main(args + ['--part', '1/2'])
    assert calls == [0]


@pytest.mark.parametrize('flags', [
    ['--parts', '-1'], ['--jobs', '0'], ['--courses', '-1'],
    ['--parts', '2', '--part', '1/2'], ['--parts', '3'], ['--part', '1/3'],
])
def test_invalid_partition_cannot_evaluate(small_run, flags):
    _, _, args, calls = small_run
    with pytest.raises((SystemExit, ValueError)):
        assist.main(args + flags)
    assert calls == []


@pytest.mark.parametrize('spelling', ['1/2', '01/02', '+1/2'])
def test_part_lock_prevents_duplicate_work(small_run, spelling):
    root, _, args, calls = small_run
    with cache_lock(root / 'parts/sample/part-1of2.lock'):
        with pytest.raises(SystemExit, match='Another process owns'):
            assist.main(args + ['--part', spelling])
    assert calls == []
    assist.main(args + ['--part', '1/2'])
    assert calls == [0]


def test_worker_failure_cleans_up_other_children(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(assist, 'PARTS_DIR', tmp_path)
    (tmp_path / 'sample').mkdir()
    monkeypatch.setattr(assist, '_part_command', lambda *args: ['unused'])
    monkeypatch.setattr(assist.time, 'sleep', lambda *args: None)
    stopped = []
    failed = SimpleNamespace(poll=lambda: 1, returncode=1)
    running = SimpleNamespace(poll=lambda: None, pid=123, terminate=lambda: stopped.append('terminate'),
                              wait=lambda **kwargs: stopped.append('wait'))
    children = iter([failed, running])
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: next(children))
    monkeypatch.setattr(subprocess, 'run', lambda args, **kwargs: stopped.append(args))
    with pytest.raises(SystemExit, match='part 1 of 2 failed'):
        assist._run_parts(SimpleNamespace(name='sample', parts=2, jobs=2, parts_dir=str(tmp_path)), {}, 2)
    assert stopped[-1] == 'wait'
    assert stopped[0] == (['taskkill', '/PID', '123', '/T', '/F'] if assist.os.name == 'nt' else 'terminate')


def test_preview_rejects_changed_weights_device_and_final_kind(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'best.pt'
    checkpoint.write_bytes(b'first weights')
    args = SimpleNamespace(checkpoint=str(checkpoint), per_level=1, assist_per_level=0, family=2)
    monkeypatch.setattr(preview, 'ROOT', tmp_path)
    monkeypatch.setattr(preview, 'EXTRA', [{'name': 'kind', 'specs': []}])
    same = preview.preview_identity(args, 'cpu', [], {})
    cache = tmp_path / 'cache'
    cache.mkdir()
    kind = cache / 'kind.json'
    write_json_atomic(kind, {'same': same, 'policy_alone': 0, 'policy_alone_fell': 0,
                             'with_planner': {}, 'seconds': 0})
    checkpoint.write_bytes(b'different weights')
    changed = preview.preview_identity(args, 'cpu', [], {})
    assert changed != same
    with pytest.raises(SystemExit, match='other arguments'):
        preview.read_kind(kind, changed)
    with pytest.raises(SystemExit, match='other arguments'):
        preview._publish(args, 'cpu', [], {}, changed, cache)
    checkpoint.write_bytes(b'first weights')
    with pytest.raises(SystemExit, match='other arguments'):
        preview.read_kind(kind, preview.preview_identity(args, 'cuda', [], {}))
    assert not (tmp_path / 'results/extra/course_preview_family2.json').exists()
