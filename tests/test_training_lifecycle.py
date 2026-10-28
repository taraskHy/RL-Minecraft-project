import dataclasses
import json
import csv
from pathlib import Path
import pytest
import torch
from parkour import train as T


def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(T, 'RUNS', tmp_path)
    original = T.held_out_courses
    monkeypatch.setattr(T, 'held_out_courses', lambda levels, per_level, n_jumps, obstacle_prob=0, family=1:
                        original(levels, 1, 2, obstacle_prob=obstacle_prob, family=family))
    cfg = T.TrainConfig(run='tiny', device='cpu', threads=1, num_envs=4, rollout=2,
                        pool_size=8, pool_refresh_every=0, hidden=(8,), epochs=1, minibatches=1,
                        eval_every=1, iterations=3, obstacle_prob=0.3, test_once=True)
    return cfg


def evaluator(scores, calls):
    def evaluate(model, pool, device, stall_ticks):
        kind = 'validation' if pool.size == 12 else 'test'
        calls.append(kind)
        score = next(scores) if kind == 'validation' else 0.4
        return {'success': score, 'per_level': {i: {'success': score, 'progress': score, 'fell': 1-score, 'n': 1} for i in range(6)}}
    return evaluate


def test_pause_resume_and_test_selected_best_once(monkeypatch, tmp_path):
    cfg = setup(monkeypatch, tmp_path)
    out = tmp_path / cfg.run
    out.mkdir()
    (out / 'pause_requested').touch()
    # Existing directory is permitted only for initialization; use a weights checkpoint.
    init = tmp_path / 'initial.pt'
    torch.save({'model': T.ActorCritic(hidden=(8,)).state_dict()}, init)
    cfg.init_from = str(init)
    calls = []
    monkeypatch.setattr(T, 'evaluate', evaluator(iter([0.9, 0.8, 0.7]), calls))
    T.train(cfg)
    assert calls == ['validation']
    assert json.loads((out/'pause_ready.json').read_text())['iteration'] == 0
    before = (out/'checkpoints/best.pt').read_bytes()
    (out/'pause_requested').unlink()
    monkeypatch.chdir(tmp_path)
    resumed = dataclasses.replace(cfg, init_from='tiny/checkpoints/latest.pt')
    T.train(resumed)
    assert calls == ['validation', 'validation', 'validation', 'test', 'test']
    assert (out/'checkpoints/best.pt').read_bytes() == before
    summary = json.loads((out/'summary.json').read_text())
    assert summary['iteration'] == 2 and summary['best']['iteration'] == 0
    assert summary['test_eval']['iteration'] == 0
    assert 'success' not in summary['last_eval']
    assert [int(r['iteration']) for r in csv.DictReader((out/'train_log.csv').open())] == [0, 1, 2]
    assert len(list(csv.DictReader((out/'eval_log.csv').open()))) == 1
    T.train(resumed)
    assert len(calls) == 5
    with pytest.raises(ValueError, match='already been tested'):
        T.train(dataclasses.replace(resumed, iterations=4))


def test_existing_run_requires_own_checkpoint(monkeypatch, tmp_path):
    cfg = setup(monkeypatch, tmp_path)
    out = tmp_path/cfg.run
    out.mkdir()
    (out/'train_log.csv').write_text('existing evidence')
    with pytest.raises(ValueError, match='Existing run'):
        T.train(dataclasses.replace(cfg, init_from='external.pt'))
    assert (out/'train_log.csv').read_text() == 'existing evidence'

def test_batched_evaluation_matches_full():
    import numpy as np
    from parkour import courses as C
    from parkour.env import CoursePool
    from parkour.live import load_model
    from parkour.paths import ROOT
    cs = [C.generate(np.random.default_rng(k+300), C.load_levels(), k%6, 2, ice_prob=.5) for k in range(12)]
    model = load_model(ROOT/'results/final_policy.pt', 'cpu')
    model.eval()
    full = T.evaluate(model, CoursePool(cs, 'cpu', C.GRID), 'cpu', 200)
    batched = T.evaluate_course_batches(model, cs, 'cpu', 200, 5, compact_ice=True)
    assert batched['success'] == full['success']
    for level, row in full['per_level'].items():
        for key in ['success','fell','progress','ticks_success','n']:
            expected=row[key]
            actual=batched['per_level'][level][key]
            assert (actual == pytest.approx(expected) if expected is not None else actual is None), (level, key, actual, expected)
