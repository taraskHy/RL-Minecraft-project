"""The generated part of HANDOFF.md: repository state, final policy, runs, games, unindexed data.

    python -m parkour.handoff            # rewrite the block between the markers in HANDOFF.md
    python -m parkour.handoff --check    # say what is out of date; exit 1 if anything is

Standard library only, so it runs on the training PC and in any sandbox without
torch. Only the text between the two markers is rewritten; the rest of HANDOFF.md
is written by hand. `python -m parkour.report` calls update() after it refreshes
results/.

The block has two parts. The repository part (commit, document dates) changes
with every commit, including the one that carries the block, so --check does not
compare it; it lists the commits made since HANDOFF.md last changed instead. The
data part is built only from files under results/ and must match exactly.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path

from .paths import ROOT

HANDOFF = ROOT / "HANDOFF.md"
RESULTS = ROOT / "results"
BEGIN = "<!-- BEGIN AUTO-SNAPSHOT: written by `python -m parkour.handoff`; edits inside are overwritten -->"
DATA = "<!-- snapshot data: compared by `python -m parkour.handoff --check` -->"
END = "<!-- END AUTO-SNAPSHOT -->"
DOCS = ["AGENTS.md", "HANDOFF.md", "TEAM_STATUS.md", "README.md", "docs/design.md", "docs/report_draft.md",
        "results/README.md", "results/DATA.md"]
DATE = "--date=format:%Y-%m-%d %H:%M"  # the author's own clock (Israel time on the training PC)


def git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return r.stdout.rstrip()  # keep leading spaces: `git status --porcelain` columns start with one


def pct(v) -> str:
    return "-" if v is None else f"{100 * v:.1f}"


def repo_part() -> list[str]:
    head = git("log", "-1", "--format=%h|%ad|%s", DATE)
    if not head:
        return ["Git is not available here, so there is no repository part.", ""]
    sha, date, subject = head.split("|", 2)
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    status = git("status", "--porcelain", "--", *DOCS) or ""
    dirty = {line[3:].strip() for line in status.splitlines()}
    lines = [f"Generated on top of commit `{sha}` ({date}, branch `{branch}`): {subject}", "",
             "| Document | Last change | Commits since |", "| --- | --- | --- |"]
    for doc in DOCS:
        last = git("log", "-1", "--format=%h|%ad", DATE, "--", doc)
        if doc in dirty:
            when, since = "updated together with this snapshot", "0"
        elif not last:
            when, since = "not in Git yet", "-"
        else:
            h, d = last.split("|")
            when, since = f"`{h}` {d}", git("rev-list", "--count", f"{h}..HEAD") or "?"
        lines.append(f"| `{doc}` | {when} | {since} |")
    return lines + [""]


def read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def obstacles_at(run_dir: Path, iteration) -> float | None:
    p = run_dir / "eval_obstacles_log.csv"
    if iteration is None or not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["iteration"]) == iteration:
                return float(r["success"])
    return None


def policy_lines(results: Path) -> list[str]:
    lines = ["### Shipped policies", "",
             "Simulator, held-out test sets (1,200 courses each), checkpoint picked on the validation family.", "",
             "| File | Checkpoint | Iteration | Validation | Test, standard | Test, obstacles |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name in ("final_policy", "final_policy_v1"):
        meta = (read_json(results / f"{name}.json") or {}).get("meta", {})
        ckpt = meta.get("checkpoint")
        if not ckpt:
            continue
        run = Path(ckpt).parts[1] if ckpt.startswith("runs/") else None
        best = ((read_json(results / "runs" / run / "summary.json") or {}).get("best") or {}) if run else {}
        it = best.get("iteration")
        obst = obstacles_at(results / "runs" / run, it) if run else None
        lines.append(f"| `results/{name}.pt` / `.json` | `{ckpt}` | {it if it is not None else '-'} | "
                     f"{pct(best.get('validation_success'))} | {pct(best.get('test_success'))} | {pct(obst)} |")
    return lines + [""]


def runs_lines(results: Path) -> list[str]:
    lines = ["### Training runs (`results/runs/*/summary.json`)", "",
             "Simulator, held-out test set. \"Picked\" is the validation-selected checkpoint (`best.pt`); runs "
             "without one predate validation selection. Runs trained on levels 0-8 (`advanced-*`, `neo-*`) are "
             "evaluated on levels 0-8, the others on levels 0-5, so the two groups are not comparable.", "",
             "| Run | Last iteration | Game ticks (B) | Picked: iteration | validation | test | Last: test | "
             "Last: obstacles |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for p in sorted((results / "runs").glob("*/summary.json")):
        s = read_json(p) or {}
        best = s.get("best") or {}
        last = s.get("last_eval") or {}
        obst = (last.get("obstacles") or {}).get("success")
        ticks = f"{s['env_steps'] / 1e9:.2f}" if "env_steps" in s else "-"
        it = best.get("iteration")
        lines.append(f"| `{p.parent.name}` | {s.get('iteration', '-')} | {ticks} | {it if it is not None else '-'} | "
                     f"{pct(best.get('validation_success'))} | {pct(best.get('test_success'))} | "
                     f"{pct(last.get('success'))} | {pct(obst)} |")
    return lines + [""]


def short_path(ckpt: str) -> str:
    """A checkpoint path relative to the repository, whichever PC wrote it."""
    ckpt = ckpt.replace("\\", "/")
    return "runs/" + ckpt.split("/runs/", 1)[1] if "/runs/" in ckpt else ckpt


def games_lines(results: Path) -> list[str]:
    lines = ["### Games played by the bot (`results/games/*/summary.json`)", "",
             "| Folder | Where | Checkpoint | Completed | Simulator, same courses | Same outcome |",
             "| --- | --- | --- | --- | --- | --- |"]
    for p in sorted((results / "games").glob("*/summary.json")):
        s = read_json(p) or {}
        n = s.get("courses")
        done = round(s["success"] * n) if n and s.get("success") is not None else None
        lines.append(f"| `{p.parent.name}` | {s.get('mode', '-')} | `{short_path(s.get('checkpoint', '-'))}` | "
                     f"{f'{done}/{n}' if done is not None else '-'} | {pct(s.get('sim_success_same_courses'))} | "
                     f"{pct(s.get('outcome_agreement'))} |")
    return lines + [""]


def unindexed(results: Path) -> list[str]:
    """Entries of results/ that results/DATA.md does not name (by path, name or a `*` pattern)."""
    index = results / "DATA.md"
    if not index.exists():
        return []
    tokens = {t.strip() for t in re.findall(r"`([^`]+)`", index.read_text(encoding="utf-8"))}
    patterns = [t for t in tokens if "*" in t]

    def named(rel: str) -> bool:
        name = rel.rstrip("/").split("/")[-1]
        keys = {rel, rel.rstrip("/"), f"results/{rel}", f"results/{rel.rstrip('/')}", name}
        return bool(keys & tokens) or any(fnmatch.fnmatch(name, pat) for pat in patterns)

    missing = []
    for entry in sorted(results.iterdir()):
        if entry.name in ("runs", "README.md", "DATA.md"):  # runs/ is indexed as results/runs/NAME/
            continue
        if entry.is_file():
            if not named(entry.name):
                missing.append(f"`{entry.name}`")
        elif entry.name == "games":
            missing += [f"`games/{g.name}/`" for g in sorted(entry.iterdir()) if g.is_dir() and not named(g.name)]
        elif not named(entry.name + "/"):
            n = sum(1 for f in entry.rglob("*") if f.is_file())
            missing.append(f"`{entry.name}/` ({n} file{'' if n == 1 else 's'})")
        else:
            missing += [f"`{entry.name}/{f.name}`" for f in sorted(entry.iterdir()) if f.is_file() and not named(f.name)]
    return missing


def data_part(results: Path = RESULTS) -> list[str]:
    lines = policy_lines(results) + runs_lines(results) + games_lines(results)
    missing = unindexed(results)
    lines += ["### In `results/` but not yet in `results/DATA.md`", ""]
    lines += [", ".join(missing) if missing else "Nothing: the data index names everything.", ""]
    return lines


def block(results: Path = RESULTS) -> str:
    return "\n".join([BEGIN, ""] + repo_part() + [DATA, ""] + data_part(results) + [END])


def _split(text: str) -> tuple[int, int]:
    if text.count(BEGIN) != 1 or text.count(END) != 1 or text.index(BEGIN) > text.index(END):
        raise ValueError(f"{HANDOFF.name} needs exactly one pair of snapshot markers")
    return text.index(BEGIN), text.index(END) + len(END)


def stored_data(text: str) -> str | None:
    """The data part of the snapshot block in a handoff text, or None if the block has none."""
    a, b = _split(text)
    current = text[a:b]
    return current.split(DATA, 1)[1].removesuffix(END).strip() if DATA in current else None


def _read(path: Path) -> tuple[str, bool]:
    raw = path.read_bytes().decode("utf-8")
    return raw.replace("\r\n", "\n"), "\r\n" in raw


def update(path: Path = HANDOFF, results: Path = RESULTS) -> bool:
    """Rewrite the snapshot block in place, keeping the file's line endings. True if it changed."""
    text, crlf = _read(path)
    a, b = _split(text)
    new = text[:a] + block(results) + text[b:]
    if new == text:
        return False
    path.write_bytes((new.replace("\n", "\r\n") if crlf else new).encode("utf-8"))
    return True


def check(path: Path = HANDOFF, results: Path = RESULTS) -> int:
    text, _ = _read(path)
    stale = False
    print("\n".join(repo_part()))
    if stored_data(text) != "\n".join(data_part(results)).strip():
        print("The snapshot's data part does not match results/: run `python -m parkour.handoff`.")
        stale = True
    try:
        rel = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:  # a copy outside the repository has no history
        rel = None
    last = git("log", "-1", "--format=%h", "--", rel) if rel else None
    if last:
        since = git("log", "--format=  %h %ad %s", DATE, f"{last}..HEAD")
        if since:
            print(f"Commits since {rel} last changed (`{last}`); update its hand-written parts:\n{since}")
            stale = True
    if not stale:
        print(f"{path.name} is up to date with results/ and the commit history.")
    return 1 if stale else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report what is out of date; exit 1 if anything is")
    a = ap.parse_args(argv)
    if a.check:
        sys.exit(check())
    print(f"{HANDOFF.name} snapshot {'updated' if update() else 'already current'}.")


if __name__ == "__main__":
    main()
