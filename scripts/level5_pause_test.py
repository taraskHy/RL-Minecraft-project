"""Test-time pause: after each landing on a new platform, k ticks without input, then the policy again.

    python scripts/level5_pause_test.py K SET     # SET: val, valobs, or a number of level-5 courses (family 11)

Prints one JSON line. A negative result (28 Oct): on 2,000 level-5 courses of
the analysis family 11, pausing 3, 6 or 10 ticks lowers success from 90.2% to
80.0%, 72.3% and 70.0% (results/level5_pause_test.jsonl). Kept so it is not
tried again.
"""
import json, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, torch
from parkour import courses as C
from parkour.env import CoursePool, EnvConfig, ParkourEnv
from parkour.live import load_model
from parkour.train import held_out_courses
torch.set_num_threads(4)
k = int(sys.argv[1]); which = sys.argv[2]
model = load_model(__import__("pathlib").Path(__file__).resolve().parents[1] / "results" / "final_policy.pt", "cpu")
levels = C.load_levels()
if which == "val":
    cs = held_out_courses(levels, 100, 10, family=2)
elif which == "valobs":
    cs = held_out_courses(levels, 100, 10, obstacle_prob=0.5, family=5)
else:  # analysis family 11, level 5
    rng = np.random.default_rng([2026, 11, 5]); cs = [C.generate(rng, levels, 5, 10, seed=i) for i in range(int(which))]
pool = CoursePool(cs, "cpu", C.GRID); n = pool.size
env = ParkourEnv(pool, EnvConfig(num_envs=n, stall_ticks=200, max_ticks=4000), sampler=None)
obs = env.reset_all(torch.arange(n))
idle = torch.tensor([1, 1, 0, 0, 3])
finished = torch.zeros(n, dtype=torch.bool); success = torch.zeros(n, dtype=torch.bool); ticks = torch.zeros(n, dtype=torch.int64)
pause = torch.zeros(n, dtype=torch.int64); prev_cp = env.cp.clone()
with torch.no_grad():
    for t in range(4000):
        a, _, _ = model.act(obs, deterministic=True)
        a = torch.where((pause > 0)[:, None], idle[None, :], a)
        pause = (pause - 1).clamp(min=0)
        obs, _, _, _, _, info = env.step(a)
        first = info["done"] & ~finished
        success |= first & info["success"]; ticks = torch.where(first, info["ep_len"], ticks)
        finished |= info["done"]
        landed = (env.cp > prev_cp) & ~finished
        pause = torch.where(landed, torch.full_like(pause, k), pause)
        prev_cp = env.cp.clone()
        if t % 25 == 0 and bool(finished.all()):
            break
lv = pool.level.numpy(); s = success.numpy()
print(json.dumps({"pause": k, "set": which, "courses": n, "success": round(float(s.mean()), 4),
                  "per_level": {int(L): round(float(s[lv == L].mean()), 3) for L in sorted(set(lv.tolist()))},
                  "mean_ticks_success": round(float(ticks.numpy()[s].mean()), 1)}))
