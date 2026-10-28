"""Pause GPU work while the GPU is hot (Windows).

    python scripts/gpu_watchdog.py [--hot 65] [--cool 58] [--every 10]
    python scripts/gpu_watchdog.py --hot 60 --cool 52 --pause-while VALORANT-Win64-Shipping.exe
    python scripts/gpu_watchdog.py --resume-all     # resume anything left suspended

Every ``--every`` seconds it reads the GPU temperature (nvidia-smi). Above
``--hot`` degrees it suspends every GPU job of this project (training,
demonstration and jump searches, planner and jump evaluations: python
processes running parkour.train, parkour.neo_demos, parkour.neo_dagger,
parkour.cem, parkour.planner, parkour.jumpeval, parkour.stress,
parkour.feasibility or parkour.neo_assist). With ``--pause-while`` (program
names as in Task Manager's details tab) they are also suspended while any of
those programs runs, e.g. a game, and resumed when it has closed and the GPU
is below ``--cool``.
Suspended processes keep all their state; nothing is lost. Below ``--cool``
degrees it resumes them. Jobs that start while paused are suspended too.
Live Minecraft runs are left alone (the bot plays in real time), and so is the
bots' planner service (parkour.neo_assist --serve). Every pause
and resume is logged with time and temperature to runs/gpu_watchdog.log.
"""
import argparse
import ctypes
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "runs" / "gpu_watchdog.log"
GPU_JOBS = re.compile(r"parkour\.(train|neo_demos|neo_dagger|neo_assist|cem|planner|jumpeval|stress|feasibility)\b"
                      r"|cem_momentum")
PROCESS_SUSPEND_RESUME = 0x0800


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def temperature() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return max(float(v) for v in out.split())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def running(names: list) -> bool:
    """Is any of these programs (image names such as VALORANT-Win64-Shipping.exe) running?"""
    if not names:
        return False
    try:
        out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    have = {line.split(",")[0].strip('"').lower() for line in out.splitlines() if line.strip()}
    return any(n.lower() in have for n in names)


def gpu_jobs() -> dict:
    """{pid: command line} of this project's GPU jobs."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60).stdout
    jobs = {}
    for line in out.splitlines():
        pid, _, cmd = line.partition("\t")
        # The bots' planner service (neo_assist --serve) answers the live bot, which plays in real time: left alone.
        if pid.strip().isdigit() and GPU_JOBS.search(cmd) and "gpu_watchdog" not in cmd and "--serve" not in cmd:
            jobs[int(pid)] = cmd.strip()
    return jobs


def _call(pid: int, fn) -> bool:
    k32, ntdll = ctypes.windll.kernel32, ctypes.windll.ntdll
    h = k32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
    if not h:
        return False
    try:
        return fn(ntdll, h) == 0
    finally:
        k32.CloseHandle(h)


def suspend(pid: int) -> bool:
    return _call(pid, lambda nt, h: nt.NtSuspendProcess(h))


def resume(pid: int) -> bool:
    return _call(pid, lambda nt, h: nt.NtResumeProcess(h))


def short(cmd: str) -> str:
    m = re.search(r"(parkour\.\w+.*|scripts/\S+.*)", cmd)
    return (m.group(1) if m else cmd)[:90]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hot", type=float, default=65.0)
    ap.add_argument("--cool", type=float, default=58.0)
    ap.add_argument("--every", type=float, default=10.0)
    ap.add_argument("--pause-while", type=lambda v: [x for x in v.split(",") if x], default=[],
                    help="comma-separated program names; GPU jobs stay suspended while one of them runs")
    ap.add_argument("--resume-all", action="store_true")
    a = ap.parse_args()
    LOG.parent.mkdir(exist_ok=True)
    if a.resume_all:
        for pid, cmd in gpu_jobs().items():
            log(f"resume (manual) pid {pid}: {resume(pid)} {short(cmd)}")
        return
    log(f"watchdog started: pause above {a.hot} C, resume below {a.cool} C, every {a.every:g} s"
        + (f", and while {', '.join(a.pause_while)} runs" if a.pause_while else ""))
    paused: dict = {}
    holding = for_heat = False
    deg = (lambda t: "temperature unknown" if t is None else f"{t:.0f} C")
    try:
        while True:
            t = temperature()
            game = running(a.pause_while)
            hot = t is not None and t > a.hot
            if not holding and (hot or game):
                for pid, cmd in gpu_jobs().items():
                    if suspend(pid):
                        paused[pid] = cmd
                holding, for_heat = True, hot
                why = f"at {deg(t)}" if hot else f"while {', '.join(a.pause_while)} runs ({deg(t)})"
                log(f"PAUSE {why}: suspended {len(paused)} processes: " + "; ".join(short(c) for c in set(paused.values())))
            elif holding and not game and ((t is not None and t < a.cool) or (t is None and not for_heat)):
                n = sum(resume(pid) for pid in paused)
                log(f"RESUME at {deg(t)}: resumed {n} of {len(paused)} processes")
                paused, holding = {}, False
            elif holding:
                for pid, cmd in gpu_jobs().items():  # jobs started while paused
                    if pid not in paused and suspend(pid):
                        paused[pid] = cmd
                        log(f"suspended a job started while paused ({deg(t)}): {short(cmd)}")
            time.sleep(a.every)
    finally:
        if paused:
            n = sum(resume(pid) for pid in paused)
            log(f"watchdog stopping: resumed {n} of {len(paused)} processes")


if __name__ == "__main__":
    sys.exit(main())
