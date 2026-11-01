"""The course builder's session: the local server, the networks for the bot and the builder bot (``parkour.builder``).

    python -m parkour.builder [--viewer YourMinecraftName] [--port 25570] [--out runs/builder-NAME]
                              [--checkpoint results/final_policy.pt] [--skill results/neo_skill.pt]
                              [--router results/routing/router_policy.pt]

Join 127.0.0.1:25570 in Minecraft Java 1.21.4. The panel's buttons beside the spawn give the bot its commands
(no chat needed: ``bridge/builder_panel.js``); chat (!help) and lines typed in this console (help) work too. The
server needs the EULA accepted on this PC (``run.cmd accept-eula``). Ctrl+C ends the session; every run is in
OUT/runs.jsonl.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from .live import load_model
from .paths import BRIDGE, NODE, ROOT

BUILDER_PROPERTIES = {"enable-command-block": "true"}  # the panel's command blocks (the live runs' server keeps them off)


def run(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--viewer", default="", help="your Minecraft name (the player this console's commands act for "
                                                  "when several are online)")
    ap.add_argument("--port", type=int, default=25570)
    ap.add_argument("--out", default="")
    ap.add_argument("--checkpoint", default=str(ROOT / "results" / "final_policy.pt"))
    ap.add_argument("--skill", default=str(ROOT / "results" / "neo_skill.pt"))
    ap.add_argument("--router", default=str(ROOT / "results" / "routing" / "router_policy.pt"))
    ap.add_argument("--no-server", action="store_true", help="join a server that is already running")
    a = ap.parse_args(argv)
    out = Path(a.out or ROOT / "runs" / time.strftime("builder-%Y%m%d-%H%M%S")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    load_model(Path(a.checkpoint), dev).export(out / "policy.json", {"checkpoint": str(a.checkpoint)})
    load_model(Path(a.skill), dev).export(out / "skill.json", {"checkpoint": str(a.skill)})
    from .planner import PlannerConfig
    from .route_policy import load_router
    load_router(Path(a.router), dev).export(out / "router.json", {"checkpoint": str(a.router)})
    fill = torch.load(a.router, map_location="cpu", weights_only=False)["config"].get("fill", "greedy")
    planner = {**dataclasses.asdict(PlannerConfig()), "router": str(out / "router.json"), "fill": fill, "blocked": "keep"}
    (out / "session.json").write_text(json.dumps({"checkpoint": a.checkpoint, "skill": a.skill, "router": a.router,
                                                   "planner": planner}, indent=1))
    from .builder import demo_courses
    (out / "demo_courses.json").write_text(json.dumps({"courses": [c.to_json() for c in demo_courses()]}))
    cmd = [str(NODE), str(BRIDGE / "builder_bot.js"), "--policy", str(out / "policy.json"), "--skill", str(out / "skill.json"),
           "--planner", json.dumps(planner), "--python", sys.executable, "--root", str(ROOT), "--out", str(out),
           "--port", str(a.port), "--demo", str(out / "demo_courses.json"),
           "--converter-args", json.dumps(["--checkpoint", a.checkpoint, "--skill", a.skill, "--router", a.router])]
    if a.viewer:
        cmd += ["--viewer", a.viewer]
    print(f"Join 127.0.0.1:{a.port} in Minecraft Java 1.21.4{' as ' + a.viewer if a.viewer else ''}. The buttons by the "
          f"spawn give the bot its commands (no chat needed); chat !help, or type help here. Runs go to {out}. "
          f"Ctrl+C ends the session.", flush=True)
    if a.no_server:
        subprocess.run(cmd, cwd=str(BRIDGE))
        return
    from .server import Server
    with open(out / "server.log", "w") as slog, Server(a.port, log=slog, properties=BUILDER_PROPERTIES):
        try:
            subprocess.run(cmd, cwd=str(BRIDGE))
        except KeyboardInterrupt:
            pass
