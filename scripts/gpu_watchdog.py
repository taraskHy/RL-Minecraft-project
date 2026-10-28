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
from ctypes import wintypes
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "runs" / "gpu_watchdog.log"
GPU_JOBS = re.compile(r"(?:^|\s)-m\s+parkour\.(train|neo_demos|neo_dagger|neo_assist|cem|planner|jumpeval|stress|feasibility)\b"
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


def gpu_sample():
    """Slow telemetry belongs to the monitor thread, never the pacing loop."""
    out = subprocess.check_output([
        "nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits"], text=True, timeout=20)
    temp, util, used, total = map(float, out.splitlines()[0].split(','))
    return temp, util / 100, used / total


class Limits:
    def __init__(self, hot, cool, duty, target):
        self.hot, self.cool, self.maximum, self.target = hot, cool, duty, target
        self.heat = False
        # Headroom absorbs CUDA kernels already queued when the host is paused.
        self.duty = min(duty, 0.5) if target else duty

    def update(self, temp, utilization):
        if temp > self.hot:
            self.heat = True
        elif temp < self.cool:
            self.heat = False
        if self.target and not self.heat:
            desired = max(0.05, self.target - 0.07)
            if utilization > desired:
                self.duty = max(0.05, self.duty - min(0.08, (utilization - desired) * 0.3))
            elif utilization < desired - 0.03:
                self.duty = min(self.maximum, self.duty + 0.005)


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
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    ntdll.NtSuspendProcess.argtypes = [wintypes.HANDLE]
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
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
    ap.add_argument("--duty", type=float, default=1.0, help="shared compute duty for all GPU jobs together")
    ap.add_argument("--period", type=float, default=0.05,
                    help="shared pacing period in seconds (default 50 ms)")
    ap.add_argument("--utilization-target", type=float, default=0,
                    help="feedback target for measured GPU busy time; zero disables feedback")
    ap.add_argument("--resume-all", action="store_true")
    a = ap.parse_args()
    if not 0 < a.duty <= 1 or a.period <= 0 or not 0 <= a.utilization_target <= 1:
        ap.error("duty must be in (0, 1] and period must be positive")
    LOG.parent.mkdir(exist_ok=True)
    if a.resume_all:
        for pid, cmd in gpu_jobs().items():
            log(f"resume (manual) pid {pid}: {resume(pid)} {short(cmd)}")
        return
    log(f"watchdog started: pause above {a.hot} C, resume below {a.cool} C, every {a.every:g} s"
        + (f", and while {', '.join(a.pause_while)} runs" if a.pause_while else "") + f", shared duty {a.duty:.0%}")
    limits = Limits(a.hot, a.cool, a.duty, a.utilization_target)
    snapshot = {'jobs': {}, 'game': False, 'sample': None, 'checked': 0.0}
    stop = threading.Event()

    def monitor():
        last_log = 0.0
        while not stop.is_set():
            try:
                temp, util, memory = gpu_sample()
                jobs = gpu_jobs()
                game = running(a.pause_while)
                limits.update(temp, util)
                snapshot.update(jobs=jobs, game=game, sample=(temp, util, memory), checked=time.monotonic())
                if time.monotonic() - last_log >= 10:
                    log(f"sample: {temp:.1f} C, GPU {util:.0%}, memory {memory:.1%}, "
                        f"host duty {limits.duty:.1%}, processes {len(jobs)}, heat hold {limits.heat}")
                    last_log = time.monotonic()
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                log(f"monitor error (work held after stale telemetry): {exc}")
            stop.wait(min(a.every, 1.0) if a.duty < 1 else a.every)

    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    paused: dict = {}
    try:
        while True:
            now = time.monotonic()
            phase = now % a.period
            boundary = a.period * limits.duty
            stale = now - snapshot['checked'] > max(5, 2 * a.every)
            should_hold = limits.heat or snapshot['game'] or stale or phase >= boundary
            if should_hold:
                for pid, cmd in snapshot['jobs'].items():
                    if pid not in paused and suspend(pid):
                        paused[pid] = cmd
            else:
                for pid in list(paused):
                    if resume(pid) or pid not in snapshot['jobs']:
                        paused.pop(pid)
            # Absolute phase prevents telemetry/logging latency stretching an on window.
            deadline = now - phase + (boundary if phase < boundary else a.period)
            time.sleep(max(0.001, min(0.05, deadline - time.monotonic())))
    finally:
        stop.set()
        if paused:
            n = sum(resume(pid) for pid in paused)
            log(f"watchdog stopping: resumed {n} of {len(paused)} processes")


if __name__ == "__main__":
    sys.exit(main())
