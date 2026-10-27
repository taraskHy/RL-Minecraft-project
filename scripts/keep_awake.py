"""Keep Windows from idle-sleeping while training processes run.

    python scripts/keep_awake.py [--pattern parkour.train --pattern sweep.py]

Asks Windows for ES_SYSTEM_REQUIRED (the request a video player makes) for as
long as any process whose command line contains one of the patterns exists,
then exits. It changes no power setting; the display may still turn off.
"""
from __future__ import annotations

import argparse
import ctypes
import subprocess
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def running(patterns: list[str]) -> bool:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { $_.CommandLine }"],
        capture_output=True, text=True).stdout
    return any(p in line for line in out.splitlines() for p in patterns if "keep_awake" not in line)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pattern", action="append", default=None)
    ap.add_argument("--interval", type=float, default=60.0)
    a = ap.parse_args(argv)
    patterns = a.pattern or ["parkour.train", "sweep.py", "parkour.live"]
    kernel = ctypes.windll.kernel32
    print(time.strftime("%H:%M:%S"), "holding the system awake while", patterns, "run", flush=True)
    try:
        while running(patterns):
            kernel.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            time.sleep(a.interval)
    finally:
        kernel.SetThreadExecutionState(ES_CONTINUOUS)
    print(time.strftime("%H:%M:%S"), "no matching process left; released", flush=True)


if __name__ == "__main__":
    main()
