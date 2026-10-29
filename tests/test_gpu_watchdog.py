from scripts import gpu_watchdog as G


def test_feedback_does_not_override_temperature_hysteresis():
    limits = G.Limits(65, 58, .7, .7)
    limits.update(40, .98)
    assert limits.duty < .5 and not limits.heat
    limits.update(66, .2)
    assert limits.heat
    limits.update(60, .2)
    assert limits.heat
    limits.update(58, .2)
    assert limits.heat
    limits.update(57, .2)
    assert not limits.heat
    for _ in range(1000):
        limits.update(40, 0)
    assert limits.duty == .7
    for _ in range(1000):
        limits.update(40, 1)
    assert limits.duty == .05


def test_gpu_pattern_ignores_keep_awake_arguments():
    assert G.GPU_JOBS.search('python -m parkour.train --run x')
    assert not G.GPU_JOBS.search('python scripts/keep_awake.py --pattern parkour.train')


def test_offline_work_is_paused_but_live_services_and_helpers_are_not():
    for command in (
        'python -m parkour.train --run x',
        'python -m parkour.live --offline --assist {}',
        'python -m parkour.neo_assist --serve',
        'python -m parkour.skills matrix --candidate a=b.pt',
        'python -m parkour.gate train --run gate-r1',
        'python -m parkour.gate eval --checkpoint runs/gate-r1/checkpoints/best.pt --split validation',
        'python -m parkour.route_train --run r',
        'python -m parkour.routing.train --run r',
    ):
        assert G.pausable(command), command
    for command in (
        'python -m parkour.live --assist {}',
        'python -m parkour.neo_assist --serve --realtime',
        'python scripts/keep_awake.py --pattern parkour.train',
        'python scripts/gpu_watchdog.py',
    ):
        assert not G.pausable(command), command
