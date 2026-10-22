"""Managed, localhost-only Minecraft servers for this project.

The arena is a disposable development world; the survival profile is a separate
ordinary survival world. Nothing here deletes or resets either world. Starting
a server requires the user's explicit, saved Minecraft EULA acceptance.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Iterator
import uuid
import warnings


ROOT = Path(__file__).resolve().parents[1]
MINECRAFT_VERSION = "1.21.4"
EULA_URL = "https://aka.ms/MinecraftEULA"
MARKER_NAME = ".rl-minecraft-managed.json"
PROFILES = {
    "arena": {"port": 25565, "level_name": "dev-arena"},
    "survival": {"port": 25566, "level_name": "fresh-survival"},
}


class ServerError(RuntimeError):
    """A managed server could not be prepared, started, or stopped safely."""


def _profile(profile: str) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError("Server profile must be 'arena' or 'survival'")
    return {"profile": profile, **PROFILES[profile]}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def offline_uuid(username: str) -> str:
    """Match Java's UUID.nameUUIDFromBytes for an offline-mode player."""
    digest = hashlib.md5(("OfflinePlayer:" + username).encode("utf-8")).digest()
    return str(uuid.UUID(bytes=digest, version=3))


def _consent_path() -> Path:
    return ROOT / ".local" / "minecraft-eula-consent.json"


def eula_accepted() -> bool:
    try:
        consent = json.loads(_consent_path().read_text(encoding="utf-8"))
        return (isinstance(consent, dict) and consent.get("schema") == 1
                and consent.get("accepted") is True and consent.get("url") == EULA_URL)
    except (OSError, ValueError):
        return False


def server_memory_mb() -> int:
    """Return this checkout's optional machine-specific Java maximum heap size."""
    path = ROOT / ".local" / "machine.json"
    if not path.exists():
        return 2048
    try:
        configuration = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ServerError(f"Cannot read machine settings from {path}: {exc}") from exc
    if not isinstance(configuration, dict):
        raise ServerError(f"Machine settings must be a JSON object: {path}")
    memory = configuration.get("server_memory_mb", 2048)
    if isinstance(memory, bool) or not isinstance(memory, int) or not 1024 <= memory <= 32768:
        raise ServerError(f"server_memory_mb must be an integer from 1024 to 32768 in {path}")
    return memory


def _properties(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith(("#", "!")):
            key, separator, value = line.partition("=")
            if separator:
                result[key.strip()] = value.strip()
    return result


def _expected_properties(profile: str) -> dict[str, str]:
    details = _profile(profile)
    return {
        "server-ip": "127.0.0.1",
        "server-port": str(details["port"]),
        "online-mode": "false",
        "enforce-secure-profile": "false",
        "level-name": details["level_name"],
        "level-seed": "",
        "gamemode": "survival",
        "force-gamemode": "true",
        "difficulty": "normal",
        "hardcore": "false",
        "enable-command-block": "false",
        "enable-rcon": "false",
        "enable-query": "false",
        "spawn-protection": "0",
        "allow-flight": "true" if profile == "arena" else "false",
        "max-players": "4",
        "view-distance": "8",
        "simulation-distance": "5",
        "generate-structures": "true",
    }


def _expected_ops(profile: str) -> list[dict[str, Any]]:
    if profile == "survival":
        return []
    return [{"uuid": offline_uuid("RLCourseBot"), "name": "RLCourseBot",
             "level": 4, "bypassesPlayerLimit": True}]


def prepare_server(profile: str = "arena") -> dict[str, Any]:
    """Create a marked profile or verify its settings, preserving its world.

    Refuse unmarked, nonempty directories and incompatible managed settings.
    Minecraft may add extra server.properties keys; those are preserved.
    """
    details = _profile(profile)
    directory = ROOT / ".local" / "servers" / profile
    marker = directory / MARKER_NAME
    expected_marker = {"schema": 1, "minecraft_version": MINECRAFT_VERSION, **details}
    if directory.is_symlink():
        raise ServerError(f"Refusing a linked server directory: {directory}")
    if directory.exists():
        if not directory.is_dir():
            raise ServerError(f"Server path is not a directory: {directory}")
        if marker.exists():
            try:
                existing_marker = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ServerError(f"Invalid managed-server marker: {marker}") from exc
            if existing_marker != expected_marker:
                raise ServerError(f"Server marker does not match the {profile} profile: {marker}")
        elif any(directory.iterdir()):
            raise ServerError(f"Refusing an unmarked, nonempty server folder: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        _write_json(marker, expected_marker)

    properties_file = directory / "server.properties"
    expected_properties = _expected_properties(profile)
    if properties_file.exists():
        actual_properties = _properties(properties_file)
        differences = [key for key, value in expected_properties.items()
                       if actual_properties.get(key) != value]
        if differences:
            raise ServerError("Managed server settings changed; restore these keys in "
                              f"{properties_file}: {', '.join(differences)}. "
                              "The existing world and settings were left untouched.")
    else:
        properties_file.write_text(
            "# Managed RL Minecraft project server; keep bound to localhost.\n"
            + "\n".join(f"{key}={value}" for key, value in expected_properties.items())
            + "\n", encoding="utf-8")

    ops_file = directory / "ops.json"
    expected_ops = _expected_ops(profile)
    if ops_file.exists():
        try:
            actual_ops = json.loads(ops_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ServerError(f"Invalid server operator list: {ops_file}") from exc
        if actual_ops != expected_ops:
            raise ServerError(f"Unexpected operator list in {ops_file}; restore the "
                              f"{profile} profile's operators before starting. No files were removed.")
    else:
        _write_json(ops_file, expected_ops)

    accepted = eula_accepted()
    eula_file = directory / "eula.txt"
    desired_eula = "true" if accepted else "false"
    if not eula_file.exists() or _properties(eula_file).get("eula") != desired_eula:
        eula_file.write_text(
            f"# Read the Minecraft EULA: {EULA_URL}\n"
            "# Acceptance is recorded only by the explicit accept-eula command.\n"
            f"eula={desired_eula}\n", encoding="utf-8")
    return {**details, "directory": directory, "eula_accepted": accepted,
            "host": "127.0.0.1", "version": MINECRAFT_VERSION}


def accept_eula() -> dict[str, Any]:
    """Record explicit consent. Call only after the human agrees in the CLI."""
    # Validate existing managed profiles before recording consent or editing them.
    for profile in PROFILES:
        directory = ROOT / ".local" / "servers" / profile
        if directory.exists():
            prepare_server(profile)
    consent = {"schema": 1, "accepted": True, "url": EULA_URL,
               "accepted_at": datetime.now(timezone.utc).isoformat()}
    path = _consent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, consent)
    for profile in PROFILES:
        if (ROOT / ".local" / "servers" / profile).exists():
            prepare_server(profile)
    return consent


def server_command(profile: str = "arena") -> list[str]:
    _profile(profile)
    java = ROOT / ".local" / "runtime" / "java" / "bin" / "java.exe"
    jar = ROOT / ".local" / "server" / f"server-{MINECRAFT_VERSION}.jar"
    for path in (java, jar):
        if not path.is_file():
            raise ServerError(f"Missing server runtime: {path}. Run run.cmd setup first.")
    return [str(java), "-Xms512M", f"-Xmx{server_memory_mb()}M", "-jar", str(jar), "nogui"]


def _check_port(port: int) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(("127.0.0.1", port))
    except OSError as exc:
        raise ServerError(f"Local port {port} is already in use. Stop its server first; "
                          "this command will not stop an existing process.") from exc


def _stop_server(process: subprocess.Popen[str], log_file: Path) -> None:
    if process.poll() is None:
        try:
            if process.stdin:
                process.stdin.write("stop\n")
                process.stdin.flush()
            process.wait(timeout=60)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            warnings.warn(f"Minecraft could not stop through its console or did not stop "
                          f"within 60 seconds; terminating "
                          f"this command's server process. Check {log_file} before reusing "
                          "the world.", RuntimeWarning, stacklevel=2)
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    if process.stdin:
        try:
            process.stdin.close()
        except OSError:
            pass  # The child may have closed its input before this final flush.


@contextmanager
def running_server(profile: str = "arena", *, timeout: float = 120,
                   mirror_output: bool = False) -> Iterator[dict[str, Any]]:
    """Own one server process, yield only after genuine readiness, always stop it."""
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("Server startup timeout must be positive and finite")
    info = prepare_server(profile)
    if not info["eula_accepted"]:
        raise ServerError(f"Minecraft EULA has not been accepted. Read {EULA_URL}, "
                          "then run run.cmd accept-eula if you agree.")
    command = server_command(profile)
    _check_port(info["port"])
    log_directory = info["directory"] / "launcher-logs"
    log_directory.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    log_file = log_directory / f"server-{stamp}.log"
    print(f"Starting Minecraft {profile} server (Java maximum heap: {server_memory_mb()} MiB); "
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
                command, cwd=info["directory"], stdin=subprocess.PIPE,
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
                                      name=f"minecraft-{profile}-log")
            reader.start()
            deadline = time.monotonic() + timeout
            while not ready.is_set():
                if process.poll() is not None or reader_finished.is_set():
                    raise ServerError(f"Minecraft exited before becoming ready. Read {log_file}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ServerError(f"Minecraft did not become ready within {timeout:g} "
                                      f"seconds. Read {log_file}")
                ready.wait(min(remaining, 0.2))
            if process.poll() is not None:
                raise ServerError(f"Minecraft exited immediately after startup. Read {log_file}")
            yield {**info, "log_file": log_file, "process": process}
        finally:
            if process is not None:
                _stop_server(process, log_file)
            if reader is not None:
                reader.join(timeout=10)
            if process is not None and process.stdout is not None:
                process.stdout.close()


def run_server(profile: str = "arena") -> int:
    """Run in the current terminal until Ctrl+C, then save and stop gracefully."""
    try:
        with running_server(profile, mirror_output=True) as info:
            print(f"Minecraft {info['version']} {profile} server ready at "
                  f"127.0.0.1:{info['port']}. Press Ctrl+C to save and stop.", flush=True)
            process = info["process"]
            while process.poll() is None:
                time.sleep(0.25)
            if process.returncode:
                raise ServerError(f"Minecraft exited with code {process.returncode}. "
                                  f"Read {info['log_file']}")
            return 0
    except KeyboardInterrupt:
        print("\nMinecraft server stopped; the world was saved unless a shutdown warning appeared.",
              file=sys.stderr, flush=True)
        return 0
