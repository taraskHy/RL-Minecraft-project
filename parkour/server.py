"""A local Minecraft 1.21.4 server with an empty void world for parkour runs.

The server binds to 127.0.0.1 in offline mode, so only programs on this PC
can join. The bot account ``ParkourBot`` is an operator: it builds each course
with /fill and places itself on the start block with /tp. These commands are
the environment's reset, the same as placing blocks in the simulator; during
an episode the bot only uses ordinary movement controls.

The Minecraft EULA must be accepted by the user, once per machine:

    run.cmd accept-eula        (or: python -m parkour.server accept-eula --agree)

Nothing here records that decision on its own.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .paths import JAVA, LOCAL, SERVER_JAR

SERVER_DIR = LOCAL / "servers" / "parkour"
CONSENT = LOCAL / "minecraft-eula-consent.json"
EULA_URL = "https://aka.ms/MinecraftEULA"
BOT_NAME = "ParkourBot"
DEFAULT_PORT = 25570
COURSE_ORIGIN = (0, 80, 0)  # world position of course grid cell (0, 0, 0)
SPAWN_PAD = (-20, 100, -20)  # small platform where players first appear, away from the course area

PROPERTIES = {
    "online-mode": "false",
    "server-ip": "127.0.0.1",
    "level-name": "parkour",
    "level-type": "minecraft\\:flat",
    "generator-settings": '{"layers"\\:[{"block"\\:"minecraft\\:air","height"\\:1}],"biome"\\:"minecraft\\:the_void"}',
    "generate-structures": "false",
    "spawn-protection": "0",
    "allow-flight": "true",
    "difficulty": "peaceful",
    "spawn-monsters": "false",
    "spawn-animals": "false",
    "spawn-npcs": "false",
    "gamemode": "adventure",
    "view-distance": "8",
    "simulation-distance": "6",
    "max-players": "6",
    "motd": "Parkour RL (local)",
    "enforce-secure-profile": "false",
    "enable-rcon": "false",
    "sync-chunk-writes": "false",
}

STARTUP_COMMANDS = [
    "gamerule doDaylightCycle false", "time set noon", "gamerule doWeatherCycle false", "weather clear",
    "gamerule fallDamage false", "gamerule doImmediateRespawn true", "gamerule spawnRadius 0",
    "gamerule announceAdvancements false", "gamerule sendCommandFeedback false", "gamerule logAdminCommands false",
    "gamerule commandBlockOutput false", "gamerule doMobSpawning false",
    "fill {x0} {y} {z0} {x1} {y} {z1} minecraft:stone".format(x0=SPAWN_PAD[0] - 2, x1=SPAWN_PAD[0] + 2, y=SPAWN_PAD[1] - 1,
                                                             z0=SPAWN_PAD[2] - 2, z1=SPAWN_PAD[2] + 2),
    "setworldspawn {} {} {}".format(*SPAWN_PAD),
    "op " + BOT_NAME,  # ops.json already lists it; this also covers a UUID mismatch
]


def offline_uuid(name: str) -> str:
    """UUID the server assigns an offline-mode player (Java UUID.nameUUIDFromBytes)."""
    h = bytearray(hashlib.md5(("OfflinePlayer:" + name).encode()).digest())
    h[6] = (h[6] & 0x0F) | 0x30
    h[8] = (h[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(h)))


def eula_accepted() -> bool:
    try:
        return json.loads(CONSENT.read_text()).get("accepted") is True
    except (OSError, ValueError):
        return False


def accept_eula() -> None:
    CONSENT.parent.mkdir(parents=True, exist_ok=True)
    CONSENT.write_text(json.dumps({"schema": 1, "accepted": True, "url": EULA_URL,
                                   "accepted_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")


def prepare(port: int = DEFAULT_PORT) -> Path:
    if not eula_accepted():
        raise SystemExit(
            "The Minecraft EULA has not been accepted on this machine. Read " + EULA_URL +
            " and, if you agree, run:  run.cmd accept-eula")
    if not SERVER_JAR.exists() or not JAVA.exists():
        raise SystemExit("Minecraft server or Java runtime missing. Run setup.cmd first.")
    SERVER_DIR.mkdir(parents=True, exist_ok=True)
    (SERVER_DIR / "eula.txt").write_text("# Accepted by the user; record in .local/minecraft-eula-consent.json\neula=true\n")
    props = dict(PROPERTIES, **{"server-port": str(port), "query.port": str(port)})
    (SERVER_DIR / "server.properties").write_text("".join(f"{k}={v}\n" for k, v in props.items()))
    ops = [{"uuid": offline_uuid(BOT_NAME), "name": BOT_NAME, "level": 4, "bypassesPlayerLimit": False}]
    (SERVER_DIR / "ops.json").write_text(json.dumps(ops, indent=2))
    return SERVER_DIR


class Server:
    """Runs the server as a child process and forwards console commands."""

    def __init__(self, port: int = DEFAULT_PORT, memory_mb: int = 2048, log=None):
        self.port = port
        self.memory_mb = memory_mb
        self.proc: subprocess.Popen | None = None
        self.ready = threading.Event()
        self.lines: list[str] = []
        self.log = log

    def start(self, timeout: float = 240.0) -> "Server":
        prepare(self.port)
        cmd = [str(JAVA), f"-Xms{self.memory_mb // 2}M", f"-Xmx{self.memory_mb}M", "-jar", str(SERVER_JAR), "nogui"]
        self.proc = subprocess.Popen(cmd, cwd=SERVER_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self._pump, daemon=True).start()
        if not self.ready.wait(timeout):
            self.stop()
            raise RuntimeError("Minecraft server did not finish starting; see " + str(SERVER_DIR / "logs" / "latest.log"))
        for c in STARTUP_COMMANDS:
            self.command(c)
        time.sleep(1.0)
        return self

    def _pump(self):
        for line in self.proc.stdout:
            line = line.rstrip()
            self.lines.append(line)
            if self.log:
                self.log.write(line + "\n")
                self.log.flush()
            if "Done (" in line and "For help" in line:
                self.ready.set()
            if "FAILED TO BIND" in line.upper():
                print(line, file=sys.stderr)

    def command(self, text: str) -> None:
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    def stop(self, timeout: float = 60.0) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.command("stop")
                self.proc.wait(timeout)
            except Exception:
                self.proc.kill()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("accept-eula", help="record your acceptance of the Minecraft EULA on this machine")
    a.add_argument("--agree", action="store_true", help="confirm you have read and accept " + EULA_URL)
    r = sub.add_parser("run", help="run the server in this console until you type stop")
    r.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args(argv)
    if args.cmd == "accept-eula":
        if not args.agree:
            print("Read the Minecraft EULA at " + EULA_URL + ". If you accept it, rerun with --agree.")
            return 1
        accept_eula()
        print("Recorded your EULA acceptance in", CONSENT)
        return 0
    if args.cmd == "run":
        s = Server(args.port, log=sys.stdout).start()
        print(f"Server running on 127.0.0.1:{args.port}. Type commands; 'stop' ends it.")
        try:
            for line in sys.stdin:
                s.command(line.strip())
                if line.strip() == "stop":
                    break
        finally:
            s.stop()
        return 0


if __name__ == "__main__":
    sys.exit(main())
