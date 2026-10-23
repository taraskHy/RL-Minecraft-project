"""One natural, newly generated Survival world per resource-training episode.

Episode directories and their worlds are retained as evidence. This module never
resets a world, copies player data, or grants the bot operator privileges.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from typing import Any, Iterator

from . import server as managed_server


ROOT = Path(__file__).resolve().parents[1]
PORT = 25567
# Loopback ports for resource episodes. Runs that execute at the same time each
# take a different one; the arena (25565) and ordinary Survival (25566) stay apart.
PORTS = range(25567, 25600)
# At most this many resource servers at once on one machine.
MAX_PARALLEL_SLOTS = 8
# One resource-parallel schedule at a time for this user on this machine, from any checkout.
LOCK_PATH = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()) / "rl-minecraft" / "resource-parallel.lock"


def validate_port(port) -> int:
    if isinstance(port, bool) or not isinstance(port, int) or port not in PORTS:
        raise ValueError(f"Resource server port must be an integer from {PORTS.start} to {PORTS.stop - 1}")
    return port


def _lock(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def schedule_lock() -> Iterator[None]:
    """Hold the machine-wide schedule lock; the operating system releases it if this process dies."""
    path = LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        _lock(handle)
    except OSError as exc:
        handle.close()
        raise managed_server.ServerError("A resource-parallel schedule is already running on this machine; "
                                         "wait for it to finish") from exc
    try:
        yield
    finally:
        try:
            _unlock(handle)
        except OSError:
            pass
        handle.close()


def schedule_running() -> bool:
    """Whether a resource-parallel schedule holds the machine-wide lock right now."""
    try:
        with schedule_lock():
            return False
    except managed_server.ServerError:
        return True


def _is_link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _episode_directory(run_directory: Path, episode: int, world_seed: int) -> Path:
    if isinstance(episode, bool) or not isinstance(episode, int) or episode <= 0:
        raise ValueError("Episode must be a positive integer")
    if (isinstance(world_seed, bool) or not isinstance(world_seed, int)
            or not -(2**63) <= world_seed < 2**63):
        raise ValueError("World seed must be a signed 64-bit integer")
    runs = ROOT.resolve() / "runs"
    candidate = Path(run_directory).absolute()
    # Check before resolve() so links inside the workspace cannot hide themselves.
    for ancestor in (candidate, *candidate.parents):
        if _is_link(ancestor):
            raise managed_server.ServerError(f"Refusing a linked run directory: {ancestor}")
        if ancestor == ROOT.resolve():
            break
    candidate = candidate.resolve()
    if candidate == runs or not candidate.is_relative_to(runs):
        raise managed_server.ServerError(f"Resource run directory must be inside {runs}")
    worlds = candidate / "worlds"
    if _is_link(worlds):
        raise managed_server.ServerError(f"Refusing a linked world directory: {worlds}")
    directory = worlds / f"episode-{episode:04d}"
    if directory.exists() or _is_link(directory):
        raise managed_server.ServerError(
            f"Fresh-world episode directory already exists: {directory}. "
            "Choose a new run or episode; existing evidence was left untouched.")
    return directory


def _prepare(directory: Path, episode: int, world_seed: int, port: int = PORT) -> dict[str, Any]:
    # mkdir(exist_ok=False) also refuses a concurrent attempt to reuse this episode.
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise managed_server.ServerError(
            f"Fresh-world episode directory already exists: {directory}") from exc
    info = {
        "profile": "resource-training", "port": port, "level_name": "world",
        "world_seed": world_seed, "episode": episode, "directory": directory,
        "host": "127.0.0.1", "version": managed_server.MINECRAFT_VERSION,
        "eula_accepted": True,
    }
    marker = {
        "schema": 1, "profile": info["profile"], "port": port,
        "minecraft_version": info["version"], "world_seed": world_seed,
        "episode": episode, "fresh_world": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (directory / managed_server.MARKER_NAME).write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    properties = {
        "server-ip": "127.0.0.1", "server-port": str(port),
        "online-mode": "false", "enforce-secure-profile": "false",
        "level-name": "world", "level-seed": str(world_seed),
        "level-type": "minecraft:normal", "generator-settings": "{}",
        "gamemode": "survival", "force-gamemode": "true",
        "difficulty": "normal", "hardcore": "false",
        "enable-command-block": "false", "enable-rcon": "false",
        "enable-query": "false", "spawn-protection": "0",
        "allow-flight": "false", "max-players": "4", "view-distance": "8",
        "simulation-distance": "5", "generate-structures": "true",
        "spawn-monsters": "true", "initial-enabled-packs": "vanilla",
        "initial-disabled-packs": "",
    }
    (directory / "server.properties").write_text(
        "# New natural Survival training world; loopback access only.\n"
        + "\n".join(f"{key}={value}" for key, value in properties.items()) + "\n",
        encoding="utf-8")
    (directory / "ops.json").write_text("[]\n", encoding="utf-8")
    (directory / "eula.txt").write_text(
        f"# Existing explicit acceptance of {managed_server.EULA_URL}\n"
        "eula=true\n", encoding="utf-8")
    return info


@contextmanager
def fresh_resource_server(*, world_seed: int, run_directory: Path, episode: int,
                          mirror_output: bool = False, timeout: float = 120,
                          port: int = PORT
                          ) -> Iterator[dict[str, Any]]:
    """Start and own one fresh Survival server, saving its world on every exit.

    The user must have already accepted the Minecraft EULA. No directory is
    created until validation, runtime checks, and the port availability check
    succeed. A failed launch still reserves its episode directory for diagnosis.
    ``port`` is this server's loopback port; runs at the same time need
    different ones, and a port already in use is refused, never freed.
    """
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("Server startup timeout must be positive and finite")
    validate_port(port)
    directory = _episode_directory(run_directory, episode, world_seed)
    if not managed_server.eula_accepted():
        raise managed_server.ServerError(
            "Minecraft EULA has not been accepted. Read " + managed_server.EULA_URL
            + ", then run run.cmd accept-eula if you agree.")
    command = managed_server.server_command("survival")
    managed_server._check_port(port)
    info = _prepare(directory, episode, world_seed, port)
    log_file = directory / "launcher.log"
    print(f"Starting fresh Survival episode {episode} (world seed {world_seed}, port {port}); "
          f"startup log: {log_file}", flush=True)
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    ready = threading.Event()
    reader_finished = threading.Event()
    process: subprocess.Popen[str] | None = None
    reader: threading.Thread | None = None
    with log_file.open("w", encoding="utf-8", buffering=1) as log:
        try:
            process = subprocess.Popen(
                command, cwd=directory, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=flags,
            )

            def read_output() -> None:
                assert process is not None and process.stdout is not None
                try:
                    for line in process.stdout:
                        log.write(line)
                        if mirror_output:
                            print(line, end="", flush=True)
                        if "Done (" in line and "For help, type" in line:
                            ready.set()
                finally:
                    reader_finished.set()

            reader = threading.Thread(target=read_output, daemon=True,
                                      name=f"minecraft-resource-episode-{episode}-log")
            reader.start()
            deadline = time.monotonic() + timeout
            while not ready.is_set():
                if process.poll() is not None or reader_finished.is_set():
                    raise managed_server.ServerError(
                        f"Minecraft exited before becoming ready. Read {log_file}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise managed_server.ServerError(
                        f"Minecraft did not become ready within {timeout:g} seconds. "
                        f"Read {log_file}")
                ready.wait(min(remaining, 0.2))
            if process.poll() is not None:
                raise managed_server.ServerError(
                    f"Minecraft exited immediately after startup. Read {log_file}")
            yield {**info, "log_file": log_file, "process": process}
        finally:
            try:
                if process is not None:
                    managed_server._stop_server(process, log_file)
            finally:
                if reader is not None:
                    reader.join(timeout=10)
                if process is not None and process.stdout is not None:
                    process.stdout.close()
