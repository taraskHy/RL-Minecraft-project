"""JSON-lines transport for real, naturally generated Survival resource tasks."""

from __future__ import annotations

import json
from pathlib import Path
import queue
import subprocess
import threading
from typing import Any

# The bridge's goals: the wood stage as before, or the stone stage (step S3 of the stone-tools milestone).
GOALS = ("wooden_pickaxe", "stone_pickaxe")


class ResourceBridge:
    """Own a bot connection, never a world or a Minecraft server.

    A fresh world supplies the episode reset. This bridge exposes no reset,
    teleport, give, creative-mode, or other privileged bot operation.

    ``goal`` sets the bridge's success rule and the actions it accepts:
    ``wooden_pickaxe`` (the default) is the wood stage exactly as before, and
    ``stone_pickaxe`` succeeds only on a held stone pickaxe. ``stone_survey``
    adds the stone fields and a ``diagnostic_only`` record to observations
    once a wooden pickaxe is held; no state encoder reads them.
    """

    def __init__(self, *, node: str | Path, port: int = 25567, goal: str = "wooden_pickaxe",
                 stone_survey: bool = False):
        if goal not in GOALS:
            raise ValueError(f"goal must be one of {', '.join(GOALS)}")
        if type(stone_survey) is not bool:
            raise ValueError("stone_survey must be True or False")
        self.goal, self.stone_survey = goal, stone_survey
        self._closed = False
        self._sequence = 0
        self._responses: queue.Queue[str | None] = queue.Queue()
        root = Path(__file__).resolve().parents[1]
        self._process = subprocess.Popen(
            [str(node), str(root / "minecraft_bridge" / "resource-bridge.js"),
             "--port", str(port), "--goal", goal, "--stone-survey", "on" if stone_survey else "off"],
            cwd=root, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        try:
            self.connection = self.request("status")
        except BaseException:
            self.close()
            raise

    def _read(self) -> None:
        assert self._process.stdout
        try:
            for line in self._process.stdout:
                self._responses.put(line)
        finally:
            self._responses.put(None)

    def request(self, operation: str, **values: Any) -> dict:
        if self._closed:
            raise RuntimeError("Resource bridge is closed")
        self._sequence += 1
        assert self._process.stdin
        try:
            self._process.stdin.write(json.dumps({"id": self._sequence, "op": operation, **values}) + "\n")
            self._process.stdin.flush()
            line = self._responses.get(timeout=65)
        except (OSError, queue.Empty) as exc:
            self.close()
            raise RuntimeError("Resource bridge failed or timed out; no transition was invented") from exc
        try:
            response = json.loads(line) if line is not None else None
            if not isinstance(response, dict) or response.get("id") != self._sequence:
                raise ValueError("missing response or mismatched request id")
            if response.get("ok") is not True:
                raise ValueError(response.get("error", "bridge operation failed"))
            if not isinstance(response.get("result"), dict):
                raise ValueError("missing result object")
            return response["result"]
        except (ValueError, TypeError) as exc:
            raise RuntimeError(f"Natural-resource game error: {exc}") from exc

    def observe(self) -> dict:
        return self.request("observe")

    def step(self, action: int) -> dict:
        return self.request("step", action=int(action))

    def players(self) -> dict:
        return self.request("status")["players"]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin:
            try:
                self._process.stdin.close()
            except OSError:
                pass
        try:
            self._process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
        self._reader.join(timeout=2)
        if self._process.stdout:
            self._process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
