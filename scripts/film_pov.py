"""Film the trained bot in real Minecraft from its own eyes.

    python scripts/film_pov.py [--checkpoint results/final_policy.pt] [--names tour,stairs,sprint]

For each named course of configs/showcase_courses.json: starts the local Minecraft server and
the live bot (parkour.live, rainbow blocks, prismarine-viewer in first person), streams the
viewer from a headless Chrome (remote debugging on port 9333, see render_videos.py), and cuts
the stream to the run itself using the bot's wall-clock marks. Writes
results/videos/real_minecraft_<name>.mp4 and, for several courses, real_minecraft_showcase.mp4.
The run's own record (episodes, replay check) stays in runs/film-<name>-<time>.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT / ".local" / "runtime" / "node" / "node.exe"
PYTHON = sys.executable
FPS = 30
LEAD_MS, TAIL_MS = 2500, 2800  # video before the first and after the last tick of the run


def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def film(course: dict, a, out_dir: Path) -> Path | None:
    name = course["name"]
    work = Path(tempfile.mkdtemp(prefix=f"film-{name}-"))
    course_file = work / "course.json"
    course_file.write_text(json.dumps({"courses": [course]}))
    run = ROOT / "runs" / f"film-{name}-{time.strftime('%m%d-%H%M%S')}"
    live = subprocess.Popen([PYTHON, "-m", "parkour.live", "--checkpoint", a.checkpoint, "--out", str(run),
                             "--courses-file", str(course_file), "--style", "rainbow", "--web-viewer", str(a.viewer_port),
                             "--first-person", "--hold", str(a.hold)],
                            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    viewer_up = threading.Event()
    marks = {}

    def read():
        for line in live.stdout:
            print("   ", line.rstrip(), flush=True)
            if "web viewer on" in line:
                viewer_up.set()
            m = re.search(r"film: (start|end) \d+ (\d+)", line)
            if m:
                marks[m.group(1)] = int(m.group(2))
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    if not viewer_up.wait(180):
        live.kill()
        print(f"{name}: the web viewer never came up")
        return None
    time.sleep(1.5)
    frames, stop = work / "frames", work / "stop"
    cap = subprocess.Popen([str(NODE), str(ROOT / "bridge" / "capture.js"), "--url", f"http://localhost:{a.viewer_port}",
                            "--out", str(frames), "--wait", "3", "--seconds", "600", "--screencast", "true",
                            "--stop-file", str(stop), "--width", "1920", "--height", "1080", "--caption", a.caption.format(name=name)],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    live.wait()
    reader.join(5)
    stop.touch()
    print("   capture:", cap.communicate(timeout=60)[0].strip())
    if "start" not in marks or "end" not in marks:
        print(f"{name}: no film marks in the bot output")
        return None
    times = json.loads((frames / "frames.json").read_text())
    seq = work / "seq"
    seq.mkdir()
    t0, t1 = marks["start"] - LEAD_MS, marks["end"] + TAIL_MS
    k, j = 0, 0
    for k in range(int((t1 - t0) * FPS / 1000)):
        t = t0 + k * 1000 / FPS
        while j + 1 < len(times) and times[j + 1]["t"] <= t:
            j += 1
        shutil.copyfile(frames / times[j]["file"], seq / f"{k:05d}.jpg")
    out = out_dir / f"real_minecraft_{name}.mp4"
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(seq / "%05d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow", "-movflags", "+faststart",
                    str(out)], check=True)
    shutil.rmtree(work, ignore_errors=True)
    result = json.loads((run / "summary.json").read_text())["episodes"][0]["result"] if (run / "summary.json").exists() else "?"
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(times)} frames streamed, {k + 1} in the video, run: {result})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="results/final_policy.pt")
    ap.add_argument("--courses-file", default="configs/showcase_courses.json")
    ap.add_argument("--names", default="tour,stairs,sprint")
    ap.add_argument("--viewer-port", type=int, default=3007)
    ap.add_argument("--hold", type=int, default=8000, help="ms on the start block while the viewer loads the world")
    ap.add_argument("--caption", default="Real Minecraft 1.21.4 · the trained bot's own view · course “{name}”, never seen in training")
    a = ap.parse_args()
    courses = {c["name"]: c for c in json.loads((ROOT / a.courses_file).read_text())["courses"]}
    out_dir = ROOT / "results" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    made = [p for p in (film(courses[n], a, out_dir) for n in a.names.split(",")) if p]
    if len(made) > 1:
        lst = out_dir / "concat.txt"
        lst.write_text("".join(f"file '{p.name}'\n" for p in made))
        subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                        "-movflags", "+faststart", str(out_dir / "real_minecraft_showcase.mp4")], check=True, cwd=str(out_dir))
        lst.unlink()
        print("wrote", out_dir / "real_minecraft_showcase.mp4")


if __name__ == "__main__":
    main()
