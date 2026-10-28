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
