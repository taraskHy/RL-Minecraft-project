"""Install pinned, verified runtimes inside this checkout, without administrator access.

Release metadata was checked against the official sources on 2026-10-23.
This module only downloads the Minecraft server. It never accepts its EULA or
starts it. Windows x64 is the supported game runtime platform.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import subprocess
import tarfile
import tempfile
from typing import Callable
import urllib.error
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".local"
NODE_VERSION = "24.22.1"
JAVA_VERSION = "21.0.12.1"
PNPM_VERSION = "11.25.0"
MINECRAFT_VERSION = "1.21.4"


class RuntimeSetupError(RuntimeError):
    """A setup problem with an actionable, user-facing explanation."""


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    algorithm: str
    digest: str
    size: int | None = None


# https://nodejs.org/dist/v24.22.1/SHASUMS256.txt
NODE = Asset(
    "node-v24.22.1-win-x64.zip",
    "https://nodejs.org/dist/v24.22.1/node-v24.22.1-win-x64.zip",
    "sha256", "d2046d28c2c3dcd211ac5145c0bf30e2047ef9dfb306693c8c6e2eccd8b95fa5",
)
# https://api.github.com/repos/adoptium/temurin21-binaries/releases/tags/jdk-21.0.12.1%2B1
# The newer 21.0.13 release did not yet contain a Windows x64 JRE when checked.
JAVA = Asset(
    "OpenJDK21U-jre_x64_windows_hotspot_21.0.12.1_1.zip",
    "https://github.com/adoptium/temurin21-binaries/releases/download/"
    "jdk-21.0.12.1%2B1/OpenJDK21U-jre_x64_windows_hotspot_21.0.12.1_1.zip",
    "sha256", "d35f31e712f0fcf6ac5a093edc90204fbff22f720ba3950bd09d331d5e621636",
)
# https://registry.npmjs.org/pnpm/11.25.0 (dist.integrity, sha512)
PNPM = Asset(
    "pnpm-11.25.0.tgz", "https://registry.npmjs.org/pnpm/-/pnpm-11.25.0.tgz",
    "sha512", base64.b64decode(
        "XN6SW08HX3Jetx+64YpC/+eEUkeJ8ZthxzHLhyHsKKruFg4BqNWvT+2ypCzb8wDv4j2zVrDUoXtNY+EfirfJVg=="
    ).hex(),
)
# https://piston-meta.mojang.com/v1/packages/ee8ab0f6ec89b3127436d35e36aa766c1e85c3a9/1.21.4.json
SERVER = Asset(
    "server-1.21.4.jar",
    "https://piston-data.mojang.com/v1/objects/4707d00eb834b446575d89a61a11b5d548d8c001/server.jar",
    "sha1", "4707d00eb834b446575d89a61a11b5d548d8c001", 56880250,
)


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(path: Path, asset: Asset) -> bool:
    return (
        path.is_file()
        and (asset.size is None or path.stat().st_size == asset.size)
        and _hash_file(path, asset.algorithm) == asset.digest
    )


def _inside_local(path: Path) -> Path:
    """Reject links/junctions that redirect managed output outside this checkout."""
    candidate = path.resolve()
    intended = ROOT.resolve() / ".local"
    if LOCAL.resolve() != intended or not candidate.is_relative_to(intended):
        raise RuntimeSetupError(f"Refusing to write outside the project .local folder: {path}")
    return candidate


def download_verified(asset: Asset, directory: Path, progress: Callable[[str], None] = print) -> Path:
    """Stream to a temporary file and publish only a verified complete download."""
    directory = _inside_local(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = _inside_local(directory / asset.name)
    if _matches(destination, asset):
        progress(f"Using verified cached download: {asset.name}")
        return destination
    progress(f"Downloading {asset.name} ...")
    partial: Path | None = None
    try:
        request = urllib.request.Request(asset.url, headers={"User-Agent": "minecraft-rl-course/0.1"})
        with tempfile.NamedTemporaryFile(prefix=asset.name + ".", suffix=".part", dir=directory, delete=False) as output:
            partial = Path(output.name)
            with urllib.request.urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        if not _matches(partial, asset):
            raise RuntimeSetupError(
                f"Checksum or size verification failed for {asset.name}. "
                "The downloaded file was rejected; rerun setup to retry."
            )
        os.replace(partial, destination)
        partial = None
        return destination
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeSetupError(f"Could not download {asset.name}: {exc}. Check your connection and rerun setup.") from exc
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)


def _member_path(directory: Path, name: str) -> Path:
    # Check before normalization: backslashes and colons are unsafe on Windows.
    parts = PurePosixPath(name).parts
    if (
        not parts or PurePosixPath(name).is_absolute()
        or "\\" in name or ":" in name or "\x00" in name
        or any(part in {"..", "."} for part in parts)
        or any(part.endswith((" ", ".")) for part in parts)
        or any(part.split(".")[0].upper() in {
            "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))
        } for part in parts)
    ):
        raise RuntimeSetupError(f"Rejected unsafe archive path: {name!r}")
    result = directory.joinpath(*parts)
    if not result.resolve().is_relative_to(directory.resolve()):
        raise RuntimeSetupError(f"Rejected archive path outside its extraction folder: {name!r}")
    return result


def extract_archive(archive: Path, directory: Path) -> None:
    """Extract only regular files and directories; never follow archive links."""
    directory = _inside_local(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for entry in bundle.infolist():
                # ZipInfo normalizes Windows separators and truncates NULs in
                # filename; validate the original archive spelling instead.
                target = _member_path(directory, entry.orig_filename)
                mode = entry.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
                    raise RuntimeSetupError(f"Rejected archive link or special file: {entry.filename}")
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(entry) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
    else:
        with tarfile.open(archive, "r:gz") as bundle:
            for entry in bundle:
                target = _member_path(directory, entry.name)
                if entry.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif entry.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = bundle.extractfile(entry)
                    if source is None:
                        raise RuntimeSetupError(f"Unreadable archive member: {entry.name}")
                    with source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
                else:
                    raise RuntimeSetupError(f"Rejected archive link or special file: {entry.name}")


def _install_archive(asset: Asset, destination: Path, expected: str, progress: Callable[[str], None]) -> None:
    destination = _inside_local(destination)
    marker = destination / ".runtime-manifest.json"
    expected_marker = {"archive": asset.name, "algorithm": asset.algorithm, "digest": asset.digest}
    if destination.exists():
        try:
            installed = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            installed = None
        if installed == expected_marker and (destination / expected).is_file():
            progress(f"Already installed: {asset.name}")
            return
        raise RuntimeSetupError(
            f"The existing runtime folder is incomplete or belongs to a different release: {destination}. "
            "Move that folder aside and rerun setup; setup will not erase it."
        )
    archive = download_verified(asset, LOCAL / "downloads", progress)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="extract-", dir=destination.parent) as scratch:
        unpacked = Path(scratch) / "unpacked"
        extract_archive(archive, unpacked)
        roots = list(unpacked.iterdir())
        if len(roots) != 1 or not roots[0].is_dir() or not (roots[0] / expected).is_file():
            raise RuntimeSetupError(f"Unexpected contents in {asset.name}; required file {expected} is missing.")
        (roots[0] / ".runtime-manifest.json").write_text(json.dumps(expected_marker, indent=2) + "\n", encoding="utf-8")
        os.replace(roots[0], destination)


def _expected_paths() -> dict[str, Path]:
    return {
        "node": LOCAL / "runtime" / "node" / "node.exe",
        "java": LOCAL / "runtime" / "java" / "bin" / "java.exe",
        "pnpm": LOCAL / "runtime" / "pnpm" / "bin" / "pnpm.cjs",
        "server": LOCAL / "server" / SERVER.name,
    }


def runtime_paths() -> dict[str, Path | None]:
    """Return project-local runtime files; missing components have value None."""
    return {name: path if path.is_file() else None for name, path in _expected_paths().items()}


def _version(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeSetupError(f"Cannot run {Path(command[0]).name}: {exc}") from exc
    return (result.stdout + result.stderr).strip()


def describe_runtime() -> dict[str, dict[str, str | bool | None]]:
    """Check runtime versions and server checksum without downloading or launching Minecraft."""
    paths = _expected_paths()
    expected = {"node": NODE_VERSION, "java": JAVA_VERSION, "pnpm": PNPM_VERSION, "server": MINECRAFT_VERSION}
    result: dict[str, dict[str, str | bool | None]] = {}
    for name, path in paths.items():
        entry: dict[str, str | bool | None] = {
            "path": str(path), "exists": path.is_file(), "expected_version": expected[name], "version": None, "error": None,
        }
        if path.is_file():
            try:
                if name == "server":
                    if not _matches(path, SERVER):
                        raise RuntimeSetupError("Minecraft server checksum does not match the pinned official release. Rerun setup.")
                    entry["version"] = MINECRAFT_VERSION
                else:
                    command = [str(path), "-version" if name == "java" else "--version"]
                    if name == "pnpm":
                        if not paths["node"].is_file():
                            raise RuntimeSetupError("The project-local Node.js runtime is missing. Rerun setup.")
                        command = [str(paths["node"]), str(path), "--version"]
                    output = _version(command)
                    entry["version"] = output.splitlines()[0] if output else ""
                    if expected[name] not in output:
                        raise RuntimeSetupError(f"Expected {name} {expected[name]}, received: {output}")
            except RuntimeSetupError as exc:
                entry["error"] = str(exc)
        else:
            entry["error"] = "Missing. Run setup."
        result[name] = entry
    return result


def setup_runtime(*, install_bridge: bool = True, progress: Callable[[str], None] = print) -> dict[str, Path | None]:
    """Install runtimes, download the server, and install the frozen bridge lockfile."""
    if platform.system() != "Windows" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeSetupError("This portable game setup supports 64-bit x86 Windows. The simulator can still run with Python.")
    _inside_local(LOCAL).mkdir(parents=True, exist_ok=True)
    _install_archive(NODE, LOCAL / "runtime" / "node", "node.exe", progress)
    _install_archive(JAVA, LOCAL / "runtime" / "java", "bin/java.exe", progress)
    _install_archive(PNPM, LOCAL / "runtime" / "pnpm", "bin/pnpm.cjs", progress)
    source = download_verified(SERVER, LOCAL / "downloads", progress)
    destination = _inside_local(LOCAL / "server" / SERVER.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not _matches(destination, SERVER):
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix="server-", suffix=".part", dir=destination.parent, delete=False) as output:
                temporary = Path(output.name)
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, output)
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    description = describe_runtime()
    for name, entry in description.items():
        if entry["error"]:
            raise RuntimeSetupError(f"{name}: {entry['error']}")
        progress(f"Verified {name}: {entry['version']}")
    paths = _expected_paths()
    if install_bridge:
        progress("Installing the bridge's pinned packages (package install scripts are disabled) ...")
        environment = os.environ.copy()
        environment["PATH"] = str(paths["node"].parent) + os.pathsep + environment.get("PATH", "")
        environment["CI"] = "true"
        command = [
            str(paths["node"]), str(paths["pnpm"]), "install", "--frozen-lockfile", "--ignore-scripts",
            "--store-dir", str(LOCAL / "pnpm-store"), "--reporter", "append-only",
        ]
        try:
            subprocess.run(command, cwd=ROOT / "minecraft_bridge", env=environment, check=True, timeout=600)
            subprocess.run([str(paths["node"]), "runtime-smoke.js"], cwd=ROOT / "minecraft_bridge", env=environment, check=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeSetupError(f"Bridge package installation or import check failed: {exc}. Rerun setup after resolving the error above.") from exc
    return runtime_paths()
