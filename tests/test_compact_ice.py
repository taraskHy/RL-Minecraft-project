import numpy as np
import torch
from parkour import courses as C, physics
from parkour.env import CoursePool


def test_compact_ice_exact_and_replace():
    cs = [C.generate(np.random.default_rng(k), C.load_levels(), k%6, 5, ice_prob=.5) for k in range(6)]
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    for dev in devices:
        full = CoursePool(cs, dev, C.GRID, ice=True)
        small = CoursePool(cs, dev, C.GRID, ice=True, compact_ice=True)
        assert small.slip.element_size() * 8 == full.slip.element_size()
        ids = torch.arange(6, device=dev)
        state = lambda: physics.PlayerState.at(full.spawn.clone(), full.spawn_yaw.clone())
        a, b = state(), state()
        g = torch.Generator().manual_seed(9)
        for i in range(120):
            action = (torch.rand(6, 6, generator=g) > .5).to(dev)
            a = physics.simulate(full.world, ids, a, action)
            b = physics.simulate(small.world, ids, b, action)
            for key in a.__dataclass_fields__:
                assert torch.equal(getattr(a,key), getattr(b,key)), (dev,i,key)
        replacement = C.generate(np.random.default_rng(99), C.load_levels(), 0, 4, ice_prob=1)
        full.replace(0, replacement); small.replace(0, replacement)
        grid = torch.cartesian_prod(torch.arange(64, device=dev), torch.arange(32, device=dev), torch.arange(64, device=dev))
        for index in range(6):
            assert torch.equal(full.world.slipperiness(index,*grid.T), small.world.slipperiness(index,*grid.T))
