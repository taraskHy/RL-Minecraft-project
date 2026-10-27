"""Repository paths and the portable runtimes installed by setup.cmd."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".local"
BRIDGE = ROOT / "bridge"
RUNS = ROOT / "runs"
CONFIGS = ROOT / "configs"
NODE = LOCAL / "runtime" / "node" / "node.exe"
JAVA = LOCAL / "runtime" / "java" / "bin" / "java.exe"
SERVER_JAR = LOCAL / "server" / "server-1.21.4.jar"
