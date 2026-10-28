"""Render the theatre's ghost races to MP4 videos.

    python scripts/render_videos.py [--races 0,1,...] [--fps 30] [--speed 1]
    python scripts/render_videos.py --replays 12,40 [--cam follow] [--caption TEXT]

Needs a headless Chrome with remote debugging on port 9333, for example:
    chrome --headless=new --remote-debugging-port=9333 --user-data-dir=TMP about:blank
Each race: results/videos/ghost_race_NN_<course>.mp4 (1920x1080, H.264), the whole
course in view, one coloured runner per training stage, with a legend.
Each replay: results/videos/replay_NN_<name>.mp4, chase camera by default; runs
without a course list show the route planner's plan.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT / ".local" / "runtime" / "node" / "node.exe"


def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", default=str(ROOT / "results" / "theatre.html"))
    ap.add_argument("--races", default="")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--replays", default="", help="replay indices to render instead of races")
    ap.add_argument("--cam", default="follow", help="camera for replays: follow, cinematic, overview")
    ap.add_argument("--caption", default="")
    a = ap.parse_args()
    html = Path(a.page).read_text(encoding="utf-8")
    data = json.loads(re.search(r"const DATA = (.*?);\r?\n", html, re.S).group(1))
    out_dir = ROOT / "results" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    url_base = Path(a.page).resolve().as_uri()
    jobs = []
    if a.replays:
        for k in (int(v) for v in a.replays.split(",") if v.strip()):
            name = data["replays"][k]["label"]
            slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            jobs.append((f"replay={k}&cam={a.cam}", a.caption or name, out_dir / f"replay_{k:02d}_{slug}.mp4"))
    else:
        for k in [int(v) for v in a.races.split(",") if v.strip()] or list(range(len(data["races"]))):
            name = data["races"][k]["name"].split(":")[0]
            slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            caption = (f"{name}: ten training stages of the same PPO agent run this unseen course at once "
                       f"(untrained to 400 updates). Recorded trajectories, Minecraft physics.")
            jobs.append((f"race={k}&cam=overview&names=1", caption, out_dir / f"ghost_race_{k:02d}_{slug}.mp4"))
    for query, caption, out in jobs:
        frames = Path(tempfile.mkdtemp(prefix="frames-"))
        url = f"{url_base}?render=1&{query}&caption={quote(caption)}"
        r = subprocess.run([str(NODE), str(ROOT / "bridge" / "render_frames.js"), "--url", url, "--out", str(frames),
                            "--fps", str(a.fps), "--speed", str(a.speed)], capture_output=True, text=True)
        if r.returncode != 0:
            print("render failed:", r.stderr[-500:])
            sys.exit(1)
        subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(frames / "frame_%05d.jpg"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow", "-movflags", "+faststart",
                        str(out)], check=True)
        shutil.rmtree(frames, ignore_errors=True)
        print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB", r.stdout.strip(), flush=True)


if __name__ == "__main__":
    main()
