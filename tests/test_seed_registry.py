"""Every world seed a plan, a declaration or a retained run uses is registered; experiments share none by accident."""

import copy
import csv
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import unittest

from infrastructure.parallel_runs import parse_plan

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = "configs/seed_registry.json"
REGISTRY_KEYS = {"schema", "description", "seed_rule", "kinds", "experiments", "not_world_seed_configs", "ranges"}
EXPERIMENT_KEYS = {"description", "sources", "planned_sources"}
RANGE_KEYS = {"first", "last", "kind", "purpose", "experiments", "reuse"}
KINDS = {"development", "validation", "training", "test", "replacement test", "benchmark", "example", "filler"}
HELD_OUT = {"test", "replacement test"}
LIMIT = 2**63
# A declaration writes a range as [first, last], "first-last" or "first to last".
SPAN = re.compile(r"^\s*(-?\d+)\s*(?:-|–|to)\s*(-?\d+)(?!\d)")
# Prose such as "worlds 40100-40119" or "Seeds 20000-20019 are not reused".
MENTION = re.compile(r"\b(?:world\s+seeds?|worlds|seeds?)\s+(\d+)\s*(?:-|–|to)\s*(\d+)(?!\d)", re.IGNORECASE)
NOT_WORLDS = {"agent", "simulator", "episode"}
COMMAND = re.compile(r"\bresource-(train|demo)\b")
DEFAULT_EPISODES = {"train": 5, "demo": 1}
# A declaration field about training: "training" or "train_worlds", but not "constraints".
TRAINING = re.compile(r"(?<![a-z])train", re.IGNORECASE)
# A declaration key that starts a run of world seeds needs an episodes or episodes_per_policy count beside it.
STARTS = ("world_seeds_from", "first_world_seed")
COUNTS = ("episodes", "episodes_per_policy")
# Declaration keys that name a seed but not a world seed; any other key naming a seed must be one the test reads.
NOT_WORLD_SEED_KEYS = {"agent_seed", "agent_seeds", "seed_note", "seed_notes"}
AGENT = re.compile(r"agent|learner", re.IGNORECASE)
# Wider than any real run (a scheduler job has at most 1000 episodes); expanding it would only hang the test.
WIDEST = 100_000
EPISODE = re.compile(r"^episode-(\d+)$")
PART = re.compile(r"^(?P<base>.+)-part(?P<number>\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class Use:
    """The world seeds of one run unit, retained run, or declaration field."""
    path: str
    name: str
    role: str
    seeds: range
    origin: str  # "plan", "declaration" (a field), "command" (in a declaration) or "run"

    @property
    def label(self) -> str:
        return f"{self.path} {self.name}" if self.name else self.path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def is_integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def spans(seeds) -> str:
    """Seeds as inclusive ranges: '1-3, 42, 100-149'."""
    ordered, parts, start = sorted(set(seeds)), [], None
    for index, seed in enumerate(ordered):
        start = seed if start is None else start
        if index + 1 == len(ordered) or ordered[index + 1] != seed + 1:
            parts.append(str(seed) if start == seed else f"{start}-{seed}")
            start = None
    return ", ".join(parts)


def listed(items, limit: int = 3, separator: str = ", ") -> str:
    items = sorted(items)
    more = f" and {len(items) - limit} more" if len(items) > limit else ""
    return separator.join(items[:limit]) + more


def owning_source(path: str, registry: dict) -> tuple[str | None, str | None]:
    """The experiment whose source file or folder holds ``path``, and that source; the longest source wins."""
    parts, best, length = PurePosixPath(path).parts, (None, None), -1
    for name, experiment in registry.get("experiments", {}).items():
        for source in experiment.get("sources", []) + experiment.get("planned_sources", []):
            prefix = PurePosixPath(source).parts
            if parts[:len(prefix)] == prefix and len(prefix) > length:
                best, length = (name, source), len(prefix)
    return best


def owner(path: str, registry: dict) -> str | None:
    """The experiment whose source file or folder holds ``path``; the longest source wins."""
    return owning_source(path, registry)[0]


def source_of(path: str) -> str:
    """What a source names for a used path: a config file itself, or the folder that holds a retained run."""
    return path if path.startswith("configs/") else PurePosixPath(path).parent.as_posix()


def covering(seed: int, registry: dict) -> list[dict]:
    return [entry for entry in registry["ranges"] if entry["first"] <= seed <= entry["last"]]


def describe(entry: dict) -> str:
    return f"{entry['first']}-{entry['last']} ({entry['kind']} of {', '.join(entry['experiments'])})"


def with_rule(entry: dict) -> str:
    """A range with its reuse rule, so that a message never invites a reuse the rule forbids."""
    allowed = "reuse allowed" if entry["reuse"]["allowed"] else "no reuse"
    return (f"{entry['first']}-{entry['last']} ({entry['kind']} of {', '.join(entry['experiments'])}; {allowed}: "
            f"{entry['reuse']['by']})")


def foreign_hint(path: str, experiment: str) -> str:
    """A declaration's world-seed fields are its experiment's own use, even the seeds of another's checkpoint."""
    return (f"If these seeds belong to another experiment's checkpoint or runs (for example a stage-1 policy), write "
            f"them in {path} as prose, such as 'trained on worlds 100-149 at 87673b8', instead of a world-seed field; "
            f"never add '{experiment}' to a range for them")


def plan_uses(path: str, plan: dict) -> tuple[list[Use], list[str]]:
    """Every run unit's seeds, an evaluation's chunks split as parse_plan splits them."""
    jobs = plan.get("jobs")
    if not isinstance(jobs, list):
        return [], [f"{path}: jobs must be a list, or the seed registry test cannot tell which worlds the plan uses"]
    defaults = plan.get("defaults") if isinstance(plan.get("defaults"), dict) else {}
    uses, problems = [], []
    for index, raw in enumerate(jobs, 1):
        job = {**defaults, **raw} if isinstance(raw, dict) else {}
        name = job["name"] if isinstance(job.get("name"), str) else f"job {index}"
        first, episodes, chunks = job.get("first_world_seed"), job.get("episodes"), job.get("chunks", 1)
        if not all(map(is_integer, (first, episodes, chunks))) or episodes < 1 or chunks < 1:
            problems.append(f"{path} {name}: first_world_seed, episodes and chunks must be integers, episodes and "
                            "chunks at least 1, or the seed registry test cannot tell which worlds the job uses")
            continue
        role = "train" if job.get("kind") == "train" else "evaluate"
        size, extra = divmod(episodes, chunks)
        for chunk in range(chunks):
            count = size + (chunk < extra)
            unit = name if chunks == 1 else f"{name}-c{chunk + 1}"
            uses.append(Use(path, f"run {unit}", role, range(first, first + count), "plan"))
            first += count
    return uses, problems


def seed_span(value) -> range | None:
    if isinstance(value, list) and len(value) == 2 and all(map(is_integer, value)) and value[0] <= value[1]:
        return range(value[0], value[1] + 1)
    match = SPAN.match(value) if isinstance(value, str) else None
    if match and int(match.group(1)) <= int(match.group(2)):
        return range(int(match.group(1)), int(match.group(2)) + 1)
    return None


def command_seeds(text: str) -> tuple[list[tuple[str, range]], list[str]]:
    """Seeds of each resource-train or resource-demo command that names --world-seed.

    A command without --world-seed describes a run whose seeds a plan sets, so it is not counted."""
    found, problems = [], []
    matches = list(COMMAND.finditer(text))
    for index, match in enumerate(matches):
        segment = text[match.start():matches[index + 1].start() if index + 1 < len(matches) else len(text)]
        if "--world-seed" not in segment:
            continue
        seed = re.search(r"--world-seed[ =](-?\d+)(?![\d.])", segment)
        episodes = re.search(r"--episodes[ =](\d+)(?![\d.])", segment)
        if seed is None or ("--episodes" in segment and episodes is None):
            problems.append(f"a command's --world-seed or --episodes is not an integer: {segment.strip()[:120]!r}")
            continue
        count = int(episodes.group(1)) if episodes else DEFAULT_EPISODES[match.group(1)]
        found.append(("train" if match.group(1) == "train" else "evaluate",
                      range(int(seed.group(1)), int(seed.group(1)) + count)))
    return found, problems


def counted(node: dict) -> int | None:
    """The largest episodes or episodes_per_policy count beside a first seed (an integer, or an object of integers)."""
    counts = []
    for key in COUNTS:
        if key in node:
            value = node[key]
            counts += list(value.values()) if isinstance(value, dict) and value else [value]
    return max(counts) if counts and all(is_integer(count) and count >= 1 for count in counts) else None


def declaration_uses(path: str, declaration) -> tuple[list[Use], list[str]]:
    """Seeds in a declaration's world-seed fields and in the commands it lists; a seed field it cannot read fails.

    Every such field counts as the declaring experiment's own use, wherever it sits."""
    uses, problems = [], []

    def walk(node, field: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                inner = f"{field}.{key}" if field else key
                role = "train" if TRAINING.search(inner) else "evaluate"
                if key in STARTS:
                    count = counted(node)
                    if is_integer(value) and count is not None:
                        uses.append(Use(path, inner, role, range(value, value + count), "declaration"))
                    else:
                        problems.append(f"{path} {inner}: {key} needs an integer and an episodes or "
                                        "episodes_per_policy count beside it (an integer, or an object of integers)")
                elif key.endswith("world_seeds"):
                    seeds = seed_span(value)
                    if seeds is None:
                        problems.append(f"{path} {inner}: write world seeds as [first, last] or 'first-last'")
                    else:
                        uses.append(Use(path, inner, role, seeds, "declaration"))
                elif ("seed" in key.lower() and key not in NOT_WORLD_SEED_KEYS
                        and not (key == "seed" and AGENT.search(field.rsplit(".", 1)[-1]))):
                    problems.append(f"{path} {inner}: a seed field the seed registry test cannot read; write world "
                                    "seeds as world_seeds [first, last] or 'first-last', or as world_seeds_from or "
                                    "first_world_seed with an episodes count beside it, and name a seed that is not a "
                                    "world seed agent_seed or agent_seeds")
                walk(value, inner)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{field}[{index}]")
        elif isinstance(node, str):
            found, trouble = command_seeds(node)
            uses.extend(Use(path, field, kind, seeds, "command") for kind, seeds in found)
            problems.extend(f"{path} {field}: {message}" for message in trouble)

    walk(declaration, "")
    return uses, problems


def strings(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from strings(item)
    elif isinstance(node, str):
        yield node


def mentioned_seeds(text: str) -> list[range]:
    found = []
    for match in MENTION.finditer(text):
        before = text[:match.start()].split()
        if before and before[-1].lower().strip(",;:(") in NOT_WORLDS:
            continue
        first, last = int(match.group(1)), int(match.group(2))
        if first <= last:
            found.append(range(first, last + 1))
    return found


@dataclass
class Survey:
    """What the configs and retained runs under one root use."""
    uses: list
    mentions: list
    paths: list
    problems: list


def survey(root: Path, registry: dict) -> Survey:
    """Expand every configs/**/*.json plan and declaration and every retained results/**/metadata.json."""
    found = Survey([], [], [], [])
    excluded = registry.get("not_world_seed_configs", {})
    for file in sorted((root / "configs").rglob("*.json")):
        path = file.relative_to(root).as_posix()
        if path == REGISTRY:
            continue
        try:
            data = read_json(file)
        except ValueError as exc:
            found.problems.append(f"{path} is not valid JSON: {exc}")
            continue
        is_plan = isinstance(data, dict) and "jobs" in data
        uses, problems = plan_uses(path, data) if is_plan else declaration_uses(path, data)
        if path in excluded:
            if is_plan or uses:
                found.problems.append(f"{path} is listed in not_world_seed_configs of {REGISTRY} but uses world seeds "
                                      f"{spans(s for use in uses for s in use.seeds) or '(a plan)'}")
            continue
        found.paths.append(path)
        found.uses += uses
        found.problems += problems
        found.mentions += [(path, seeds) for text in strings(data) for seeds in mentioned_seeds(text)]
    results = root / "results"
    for file in sorted(results.rglob("metadata.json")) if results.is_dir() else []:
        run = file.parent.relative_to(root).as_posix()
        try:
            data = read_json(file)
        except ValueError as exc:
            found.problems.append(f"{run}/metadata.json is not valid JSON: {exc}")
            continue
        # Only a live run records world seeds; other metadata files are not runs in Minecraft worlds.
        if not isinstance(data, dict) or "first_world_seed" not in data:
            continue
        first, requested = data.get("first_world_seed"), data.get("episodes_requested")
        if not (is_integer(first) and is_integer(requested)) or requested < 1:
            found.problems.append(f"{run}/metadata.json: first_world_seed and episodes_requested must be integers, "
                                  "episodes_requested at least 1")
            continue
        found.paths.append(run)
        role = "train" if "training" in str(data.get("mode")) else "evaluate"
        found.uses.append(Use(run, "", role, range(first, first + requested), "run"))
    found.problems += [f"{use.label} spans {len(use.seeds)} world seeds from {use.seeds.start}; check the numbers"
                       for use in found.uses if len(use.seeds) > WIDEST]
    found.uses = [use for use in found.uses if len(use.seeds) <= WIDEST]
    return found


def ownership_problems(found: Survey, registry: dict) -> list[str]:
    seeds_of: dict[str, set] = {}
    for use in found.uses:
        seeds_of.setdefault(use.path, set()).update(use.seeds)
    configs, folders = {}, {}
    for path in found.paths:
        if owner(path, registry) is None:
            group = configs if path.startswith("configs/") else folders
            group.setdefault(source_of(path), set()).update(seeds_of.get(path, ()))
    problems = []
    for path, seeds in sorted(configs.items()):
        using = f" (it uses world seeds {spans(seeds)})" if seeds else ""
        problems.append(f"{path} is not in {REGISTRY}{using}: add it to the sources of the experiment it belongs to "
                        "and register the world seeds it uses, or list it in not_world_seed_configs if it uses no "
                        "Minecraft world seeds")
    for folder, seeds in sorted(folders.items()):
        problems.append(f"{folder} holds retained runs on world seeds {spans(seeds)}, but no experiment in {REGISTRY} "
                        f"lists it among its sources: add {folder} itself to the sources of the experiment it belongs "
                        "to and register the seeds; a folder above it would also claim every later run under it")
    return problems


def coverage_problems(uses: list, registry: dict) -> list[str]:
    missing: dict[tuple[str, str], set] = {}
    names: dict[tuple[str, str], set] = {}
    declared: set = set()
    for use in uses:
        experiment = owner(use.path, registry)
        if experiment is None:
            continue
        own = [entry for entry in registry["ranges"] if experiment in entry["experiments"]]
        lost = {seed for seed in use.seeds if not any(e["first"] <= seed <= e["last"] for e in own)}
        if lost:
            missing.setdefault((use.path, experiment), set()).update(lost)
            names.setdefault((use.path, experiment), set()).update([use.name] if use.name else [])
            if use.origin == "declaration":
                declared.add((use.path, experiment))
    problems = []
    for (path, experiment), seeds in sorted(missing.items()):
        holding = [covering(seed, registry) for seed in seeds]
        others = {(entry["first"], entry["last"]): entry for entries in holding for entry in entries}
        where = f"; they fall in {listed(map(with_rule, others.values()), separator=' and ')}" if others else ""
        label = f"{path} ({listed(names[path, experiment])})" if names[path, experiment] else path
        advice = "Choose unused seeds and add a range for them"
        # A source above the used path may claim it for the wrong experiment; the ranges then name the right one.
        source, claimed = source_of(path), owning_source(path, registry)[1]
        candidates = (set.intersection(*(set(entry["experiments"]) for entries in holding for entry in entries))
                      if all(holding) else set())
        if candidates and len(PurePosixPath(claimed).parts) < len(PurePosixPath(source).parts):
            who = " or ".join(sorted(candidates))
            advice += (f", or, if {source} belongs to {who}, list {source} itself in the sources of {who} (the longest "
                       f"source path decides the owner, and {claimed} now gives it to '{experiment}')")
        if any(entry["reuse"]["allowed"] for entry in others.values()):
            advice += (f", or, only if the reuse rule of the range that holds them allows this use by '{experiment}', "
                       f"name '{experiment}' on that range")
        message = (f"{label} uses world seeds {spans(seeds)}, but no range in {REGISTRY} names experiment "
                   f"'{experiment}' for them{where}. {advice}")
        problems.append(message + (f". {foreign_hint(path, experiment)}" if (path, experiment) in declared else ""))
    return problems


def users_by_seed(uses: list, registry: dict) -> dict[int, dict[str, set]]:
    """The experiments using each seed, with their uses; ownership_problems reports a use no experiment owns."""
    users: dict[int, dict[str, set]] = {}
    for use in uses:
        experiment = owner(use.path, registry)
        if experiment is None:
            continue
        for seed in use.seeds:
            users.setdefault(seed, {}).setdefault(experiment, set()).add(use)
    return users


def overlap_problems(uses: list, registry: dict) -> list[str]:
    """Experiments share a seed only where the one range holding it names them all."""
    seeds_of: dict[tuple, set] = {}
    found: dict[tuple, dict[str, set]] = {}
    ranges: dict[tuple, dict] = {}
    for seed, by in users_by_seed(uses, registry).items():
        entries = covering(seed, registry)
        if len(by) < 2 or (entries and set(by) <= set(entries[0]["experiments"])):
            continue
        span = (entries[0]["first"], entries[0]["last"]) if entries else None
        key = (tuple(sorted(by)), span)
        ranges[key] = entries[0] if entries else None
        seeds_of.setdefault(key, set()).add(seed)
        for experiment, where in by.items():
            found.setdefault(key, {}).setdefault(experiment, set()).update(where)
    problems = []
    for key, seeds in sorted(seeds_of.items(), key=lambda item: (item[0][0], min(item[1]))):
        experiments, entry = key[0], ranges[key]
        unnamed = [experiment for experiment in experiments if entry is None or experiment not in entry["experiments"]]
        who = "; ".join(f"{experiment} ({listed({use.label for use in found[key][experiment]})})"
                        for experiment in experiments)
        start = f"World seeds {spans(seeds)} are used by {len(experiments)} experiments"
        if entry is None:
            message = (f"{start}, but no range in {REGISTRY} holds them: {who}. Choose unused seeds, or, if the "
                       "reuse is deliberate, register them in one range that names all of these experiments")
        else:
            absent = ", ".join(unnamed)
            advice = (f", or, only if the reuse rule of that range allows this use, add {absent} to it"
                      if entry["reuse"]["allowed"] else "; that range allows no reuse")
            message = (f"{start}, but the range that holds them, {describe(entry)}, does not name {absent}: {who}. "
                       f"Choose unused seeds{advice}")
        # Only a declaration field of an experiment the range leaves out may hold another experiment's seeds.
        declared = sorted({(use.path, experiment) for experiment in unnamed for use in found[key][experiment]
                           if use.origin == "declaration"})
        problems.append(message + "".join(f". {foreign_hint(path, experiment)}" for path, experiment in declared))
    return problems


def held_out_problems(uses: list, registry: dict) -> list[str]:
    users = users_by_seed(uses, registry)
    problems = []
    for entry in registry["ranges"]:
        if entry["kind"] not in HELD_OUT:
            continue
        seeds = range(entry["first"], entry["last"] + 1)
        experiments = {experiment for seed in seeds for experiment in users.get(seed, {})}
        if len(experiments) > 1:
            problems.append(f"Held-out range {describe(entry)} is used by {len(experiments)} experiments "
                            f"({', '.join(sorted(experiments))}); a held-out range belongs to exactly one experiment")
        for use in uses:
            trained = [seed for seed in use.seeds if seed in seeds]
            if use.role == "train" and trained:
                problems.append(f"{use.label} trains on world seeds {spans(trained)} of held-out range "
                                f"{describe(entry)}; a held-out range is never trained on")
    return problems


def kind_problems(uses: list, registry: dict) -> list[str]:
    """An evaluation never runs on training worlds, whichever experiment owns the range or the run."""
    problems = []
    for entry in registry["ranges"]:
        if entry["kind"] != "training":
            continue
        seeds = range(entry["first"], entry["last"] + 1)
        for use in uses:
            evaluated = [seed for seed in use.seeds if seed in seeds]
            if use.role == "evaluate" and evaluated:
                problems.append(f"{use.label} evaluates on world seeds {spans(evaluated)} of training range "
                                f"{describe(entry)}; an evaluation never runs on training worlds: choose validation, "
                                "development or test seeds")
    return problems


def mention_problems(mentions: list, registry: dict) -> list[str]:
    problems = []
    for path, seeds in mentions:
        lost = [seed for seed in seeds if not covering(seed, registry)]
        if lost:
            problems.append(f"{path} mentions world seeds {spans(lost)}, which no range in {REGISTRY} covers; register "
                            "them, or reword the text if they are not world seeds")
    return problems


def all_problems(root: Path, registry: dict) -> list[str]:
    """Every problem with the configs and retained runs under ``root``; the registry's own form is checked apart."""
    found = survey(root, registry)
    return (found.problems + ownership_problems(found, registry) + coverage_problems(found.uses, registry)
            + overlap_problems(found.uses, registry) + held_out_problems(found.uses, registry)
            + kind_problems(found.uses, registry) + mention_problems(found.mentions, registry))


def registry_problems(registry, root: Path) -> list[str]:
    """The registry's own form: known keys and kinds, sources that exist, and ranges that never overlap."""
    if not isinstance(registry, dict):
        return [f"{REGISTRY} must be a JSON object"]
    problems = []
    if set(registry) != REGISTRY_KEYS:
        problems.append(f"{REGISTRY} keys must be exactly {sorted(REGISTRY_KEYS)}, not {sorted(registry)}")
    if registry.get("schema") != 1:
        problems.append(f"{REGISTRY}: schema must be 1")
    if not isinstance(registry.get("kinds"), dict) or set(registry["kinds"]) != KINDS:
        problems.append(f"{REGISTRY}: kinds must describe exactly {sorted(KINDS)}")
    experiments = registry.get("experiments")
    if not isinstance(experiments, dict) or not experiments:
        return problems + [f"{REGISTRY}: experiments must be a non-empty object"]
    claimed: dict[str, str] = {}
    for name, experiment in experiments.items():
        if not isinstance(experiment, dict) or set(experiment) - EXPERIMENT_KEYS or "sources" not in experiment:
            problems.append(f"{REGISTRY} experiment {name}: keys must be description, sources and optionally "
                            "planned_sources")
            continue
        if not isinstance(experiment.get("description"), str) or not experiment["description"].strip():
            problems.append(f"{REGISTRY} experiment {name}: needs a description")
        for key in ("sources", "planned_sources"):
            paths = experiment.get(key, [])
            if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
                problems.append(f"{REGISTRY} experiment {name}: {key} must be a list of paths")
                continue
            for path in paths:
                pure = PurePosixPath(path)
                if not path or "\\" in path or pure.is_absolute() or ".." in pure.parts or path.endswith("/"):
                    problems.append(f"{REGISTRY} experiment {name}: {path!r} must be a relative path with '/'")
                elif len(pure.parts) < 2 or pure.parts[0] not in ("configs", "results"):
                    problems.append(f"{REGISTRY} experiment {name}: {key} entry {path} must name a file or folder "
                                    "inside configs/ or results/, the only folders the seed registry test reads, and "
                                    "never configs or results itself, which would claim every later file")
                elif ((pure.parts[0] == "configs" and not path.endswith(".json") and not (root / path).is_dir())
                        or (pure.parts[0] == "results" and (path.endswith(".json") or (root / path).is_file()))):
                    problems.append(f"{REGISTRY} experiment {name}: {key} entry {path} is never read: a source under "
                                    "configs/ is a .json file or a folder, and one under results/ is a folder of "
                                    "retained runs")
                elif key == "sources" and not (root / path).exists():
                    problems.append(f"{REGISTRY} experiment {name}: source {path} does not exist; move it to "
                                    "planned_sources if it is not written yet")
                if path in claimed:
                    problems.append(f"{REGISTRY}: {path} is a source of both {claimed[path]} and {name}")
                claimed[path] = name
    excluded = registry.get("not_world_seed_configs")
    if not isinstance(excluded, dict) or not all(isinstance(v, str) and v.strip() for v in excluded.values()):
        problems.append(f"{REGISTRY}: not_world_seed_configs must map config paths to the reason they hold no "
                        "world seeds")
    else:
        for path in excluded:
            if not (path.startswith("configs/") and path.endswith(".json")):
                problems.append(f"{REGISTRY}: not_world_seed_configs names {path}, but the seed registry test reads "
                                "only .json files under configs/")
            elif not (root / path).is_file():
                problems.append(f"{REGISTRY}: not_world_seed_configs names {path}, which does not exist")
            if path in claimed:
                problems.append(f"{REGISTRY}: {path} is both a source of {claimed[path]} and in not_world_seed_configs")
    ranges = registry.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        return problems + [f"{REGISTRY}: ranges must be a non-empty list"]
    good, named = [], set()
    for index, entry in enumerate(ranges, 1):
        label = f"{REGISTRY} range {index}"
        if not isinstance(entry, dict) or not RANGE_KEYS <= set(entry) <= RANGE_KEYS | {"notes"}:
            problems.append(f"{label}: keys must be {sorted(RANGE_KEYS)} and optionally notes")
            continue
        first, last = entry["first"], entry["last"]
        if not (is_integer(first) and is_integer(last) and -LIMIT <= first <= last < LIMIT):
            problems.append(f"{label}: first and last must be signed 64-bit integers with first <= last")
            continue
        label = f"{REGISTRY} range {first}-{last}"
        users, reuse = entry["experiments"], entry["reuse"]
        if entry["kind"] not in KINDS:
            problems.append(f"{label}: kind {entry['kind']!r} is not one of {sorted(KINDS)}")
        if not isinstance(entry["purpose"], str) or not entry["purpose"].strip():
            problems.append(f"{label}: needs a purpose")
        if (not isinstance(users, list) or not users or len(set(map(str, users))) != len(users)
                or not all(isinstance(u, str) and u in experiments for u in users)):
            problems.append(f"{label}: experiments must be a non-empty list of distinct experiments named in "
                            f"{REGISTRY}, not {users!r}")
            continue
        if (not isinstance(reuse, dict) or set(reuse) != {"allowed", "by"} or not isinstance(reuse["allowed"], bool)
                or not isinstance(reuse["by"], str) or not reuse["by"].strip()):
            problems.append(f"{label}: reuse must be {{\"allowed\": true or false, \"by\": who may reuse it}}")
            continue
        if entry["kind"] in HELD_OUT and reuse["allowed"]:
            problems.append(f"{label}: a {entry['kind']} range can never allow reuse")
        if not reuse["allowed"] and len(users) != 1:
            problems.append(f"{label}: a range that allows no reuse belongs to exactly one experiment, not {users}")
        named.update(users)
        good.append(entry)
    if [entry["first"] for entry in good] != sorted(entry["first"] for entry in good):
        problems.append(f"{REGISTRY}: list the ranges in order of their first seed")
    ordered = sorted(good, key=lambda entry: entry["first"])
    for before, after in zip(ordered, ordered[1:]):
        if after["first"] <= before["last"]:
            problems.append(f"{REGISTRY}: ranges {describe(before)} and {describe(after)} overlap; one range declares "
                            "sharing by naming several experiments")
    unused = sorted(set(experiments) - named)
    if unused:
        problems.append(f"{REGISTRY}: experiments {', '.join(unused)} are named by no range")
    return problems


def tracked_json(root: Path) -> list[str] | None:
    """The .json files git tracks under ``root``, or None if ``root`` is no git checkout or git cannot run."""
    try:
        result = subprocess.run(["git", "ls-files", "-z", "--", "*.json"], cwd=root, capture_output=True,
                                encoding="utf-8", check=False)
    except OSError:
        return None
    return [path for path in result.stdout.split("\0") if path] if result.returncode == 0 else None


def stray_plan_problems(root: Path, paths: list[str]) -> list[str]:
    """resource-parallel runs any plan committed in the checkout, but the survey reads plans only under configs/."""
    problems = []
    for path in sorted(paths):
        # A retained schedule keeps a copy of its plan (plan.json) under results/; its runs are read from metadata.json.
        if PurePosixPath(path).parts[0] in ("configs", "results") or not (root / path).is_file():
            continue
        try:
            data = read_json(root / path)
        except ValueError:
            continue  # resource-parallel cannot read it either.
        if isinstance(data, dict) and "jobs" in data:
            problems.append(f"{path} has jobs like a resource-parallel plan but lies outside configs/, so the seed "
                            "registry test never reads its seeds: move it into configs/ and register it")
    return problems


def retained_run_problems(root: Path) -> list[str]:
    """Each retained episode played the world its metadata.json assigns; each -partN starts where the run stopped."""
    runs: dict[Path, tuple[int, int]] = {}
    results = root / "results"
    for file in sorted(results.rglob("metadata.json")) if results.is_dir() else []:
        try:
            data = read_json(file)
        except ValueError:
            continue  # survey() reports it.
        if isinstance(data, dict) and all(is_integer(data.get(k)) for k in ("first_world_seed", "episodes_requested")):
            runs[file.parent] = (data["first_world_seed"], data["episodes_requested"])

    def attempted(folder: Path) -> int:
        return sum(1 for path in folder.glob("episode-*") if path.is_dir())

    problems = []
    for folder, (first, requested) in runs.items():
        label = folder.relative_to(root).as_posix()
        for episode in sorted(path for path in folder.glob("episode-*") if path.is_dir()):
            match = EPISODE.match(episode.name)
            if match is None:
                problems.append(f"{label}/{episode.name} is not named episode-NNNN")
                continue
            number = int(match.group(1))
            if not 1 <= number <= requested:
                problems.append(f"{label}/{episode.name} lies outside the {requested} episodes its metadata.json "
                                "requested")
            outcome = episode / "outcome.json"
            try:
                record = read_json(outcome) if outcome.is_file() else {}
            except ValueError:
                problems.append(f"{label}/{episode.name}/outcome.json is not valid JSON")
                continue
            seed = record.get("world_seed") if isinstance(record, dict) else None
            if seed is not None and seed != first + number - 1:
                problems.append(f"{label}/{episode.name}/outcome.json records world seed {seed}, but metadata.json "
                                f"puts episode {number} on {first + number - 1}")
        table = folder / "episodes.csv"
        if table.is_file():
            with table.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        number, seed = int(row["episode"]), int(row["world_seed"])
                    except (KeyError, TypeError, ValueError):
                        problems.append(f"{label}/episodes.csv has a row without an integer episode and world_seed")
                        continue
                    if seed != first + number - 1:
                        problems.append(f"{label}/episodes.csv puts episode {number} on world seed {seed}, but "
                                        f"metadata.json puts it on {first + number - 1}")
        match = PART.match(folder.name)
        if match is None:
            continue
        base = folder.with_name(match.group("base"))
        chain = [base] + [base.with_name(f"{base.name}-part{k}") for k in range(1, int(match.group("number")))]
        absent = [path.name for path in chain if path not in runs]
        if absent:
            problems.append(f"{label} is a continuation, but {', '.join(absent)} has no metadata.json beside it")
            continue
        done = sum(attempted(path) for path in chain)
        expected = (runs[base][0] + done, runs[base][1] - done)
        if (first, requested) != expected:
            problems.append(f"{label} starts at world seed {first} with {requested} episodes, but after the {done} "
                            f"episodes attempted in {', '.join(path.name for path in chain)} it must start at "
                            f"{expected[0]} with {expected[1]}")
    return problems


class SeedRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = read_json(ROOT / REGISTRY)
        cls.found = survey(ROOT, cls.registry)

    def assertNone(self, problems):
        if problems:
            self.fail(f"{len(problems)} seed registry problem(s):\n" + "\n".join(problems))

    def seeds(self, prefix: str, role: str | None = None) -> set:
        return {seed for use in self.found.uses if PurePosixPath(use.path).parts[:len(PurePosixPath(prefix).parts)]
                == PurePosixPath(prefix).parts and role in (None, use.role) for seed in use.seeds}

    def test_registry_is_well_formed_and_its_ranges_never_overlap(self):
        self.assertNone(registry_problems(self.registry, ROOT))

    def test_every_config_and_retained_run_is_read_and_belongs_to_an_experiment(self):
        self.assertNone(self.found.problems + ownership_problems(self.found, self.registry))

    def test_every_used_seed_is_registered_for_its_experiment(self):
        self.assertNone(coverage_problems(self.found.uses, self.registry))

    def test_experiments_share_seeds_only_where_their_range_names_them_all(self):
        self.assertNone(overlap_problems(self.found.uses, self.registry))

    def test_held_out_ranges_belong_to_one_experiment_and_are_never_trained_on(self):
        self.assertNone(held_out_problems(self.found.uses, self.registry))

    def test_evaluations_never_run_on_training_worlds(self):
        self.assertNone(kind_problems(self.found.uses, self.registry))

    def test_seeds_mentioned_in_configs_are_registered(self):
        self.assertNone(mention_problems(self.found.mentions, self.registry))

    def test_retained_runs_played_the_worlds_their_metadata_assigns(self):
        self.assertNone(retained_run_problems(ROOT))

    def test_no_committed_plan_lies_outside_configs(self):
        paths = tracked_json(ROOT)
        if paths is None:
            self.skipTest(f"{ROOT} is not a git checkout or git cannot run; resource-parallel runs only plans "
                          "committed in a git checkout, so no plan can run from here")
        # Guards against a listing that silently reads nothing.
        self.assertIn("configs/initial_q_plan.json", paths)
        self.assertNone(stray_plan_problems(ROOT, paths))

    def test_the_survey_finds_the_known_uses(self):
        # Guards against a survey that silently reads nothing.
        self.assertGreaterEqual(sum(1 for use in self.found.uses if use.path.startswith("results/")), 54)
        retained = self.seeds("results")
        # docs/stone_tools_design.md section 8.1: 1-3, 42, 100-149, 10000-10009, 20000-20019 and 30000-30019.
        for first, last in ((1, 3), (42, 42), (100, 149), (10000, 10009), (20000, 20019), (30000, 30019)):
            self.assertLessEqual(set(range(first, last + 1)), retained, f"{first}-{last}")
        self.assertEqual(self.seeds("results/pc_resource_runs/87673b8", "train"), set(range(100, 150)))
        self.assertEqual(self.seeds("results/pc_resource_runs/87673b8/mx-history-sarsa-train-part1"),
                         set(range(124, 150)))
        self.assertEqual(self.seeds("results/pc_resource_runs/87673b8", "evaluate"), set(range(20000, 20020)))
        self.assertEqual(self.seeds("results/parallel_benchmark"), set(range(30000, 30020)))
        self.assertEqual(self.seeds("configs/resource_masking_experiment.json", "train"), set(range(100, 150)))
        self.assertEqual(self.seeds("configs/resource_masking_experiment.json", "evaluate"), set(range(20000, 20020)))
        self.assertEqual(self.seeds("configs/initial_q_experiment.json"),
                         set(range(100, 150)) | set(range(40000, 40020)))
        self.assertEqual(self.seeds("configs/initial_q_plan.json", "train"), set(range(100, 150)))
        self.assertEqual(self.seeds("configs/initial_q_plan.json", "evaluate"),
                         set(range(40000, 40020)) | set(range(40100, 40120)))
        self.assertEqual(self.seeds("configs/parallel_contention_benchmark.json"), set(range(30000, 30020)))
        for name in ("1slot", "2slots", "4slots"):
            self.assertEqual(self.seeds(f"configs/parallel_benchmark_{name}.json"), set(range(30000, 30020)))
        self.assertEqual(self.seeds("configs/parallel_example_plan.json"),
                         set(range(50000, 50010)) | set(range(50100, 50108)))

    def test_the_plan_expansion_matches_the_scheduler(self):
        checked = 0
        for file in sorted((ROOT / "configs").rglob("*.json")):
            text = file.read_text(encoding="utf-8")
            data = json.loads(text)
            if not (isinstance(data, dict) and "jobs" in data):
                continue
            try:
                _, units = parse_plan(text)
            except ValueError:
                continue  # The scheduler refuses it; its seeds are still checked as the file states them.
            path = file.relative_to(ROOT).as_posix()
            mine = sorted((use.name, use.seeds.start, use.seeds.stop) for use in plan_uses(path, data)[0])
            theirs = sorted((f"run {u.name}", u.first_world_seed, u.first_world_seed + u.episodes) for u in units)
            self.assertEqual(mine, theirs, path)
            checked += 1
        self.assertGreaterEqual(checked, 5)

    def test_a_chunked_job_splits_its_seeds_like_the_scheduler(self):
        uses, problems = plan_uses("p.json", {"defaults": {"chunks": 1}, "jobs": [
            {"name": "e", "kind": "evaluate", "episodes": 10, "first_world_seed": 500, "chunks": 4}]})
        self.assertEqual(problems, [])
        self.assertEqual([(use.name, use.seeds) for use in uses],
                         [("run e-c1", range(500, 503)), ("run e-c2", range(503, 506)), ("run e-c3", range(506, 508)),
                          ("run e-c4", range(508, 510))])

    def test_declarations_are_read_from_fields_and_commands(self):
        declaration = {
            "training": {"world_seeds_from": 7, "episodes": {"a": 3, "b": 5}},
            "test": {"world_seeds": [900, 904]},
            "constraints": {"world_seeds": "800 to 801 (two worlds)"},
            "commands": ["run.cmd resource-train --episodes 2 --world-seed 300 --seed 0",
                         "run.cmd resource-demo --policy <p> --world-seed 400", "resource-demo --random-baseline"],
        }
        uses, problems = declaration_uses("d.json", declaration)
        self.assertEqual(problems, [])
        self.assertEqual(sorted(((use.role, use.seeds) for use in uses), key=lambda u: (u[0], u[1].start)),
                         [("evaluate", range(400, 401)), ("evaluate", range(800, 802)), ("evaluate", range(900, 905)),
                          ("train", range(7, 12)), ("train", range(300, 302))])
        _, problems = declaration_uses("d.json", {"test": {"world_seeds": [1, 2, 3]}})
        self.assertIn("d.json test.world_seeds: write world seeds as [first, last]", problems[0])
        uses, problems = declaration_uses("d.json", {
            "training": {"first_world_seed": 100, "episodes": 40},
            "test": {"first_world_seed": 20000, "episodes_per_policy": {"learned": 20, "random": 10}},
            "learners": {"seed": 3, "agent_seeds": [0, 1]}, "seed_note": "agent seeds 0 to 1"})
        self.assertEqual(problems, [])
        self.assertEqual([(use.name, use.role, use.seeds, use.origin) for use in uses],
                         [("training.first_world_seed", "train", range(100, 140), "declaration"),
                          ("test.first_world_seed", "evaluate", range(20000, 20020), "declaration")])
        _, problems = declaration_uses("d.json", {"training": {"first_world_seed": 100}})
        self.assertEqual(problems, ["d.json training.first_world_seed: first_world_seed needs an integer and an "
                                    "episodes or episodes_per_policy count beside it (an integer, or an object of "
                                    "integers)"])
        # A seed field the test cannot read fails, so a declaration never passes on seeds nobody checked.
        for declaration, field in (({"test": {"seeds": [20000, 20019]}}, "test.seeds"),
                                   ({"training": {"world_seed_range": "100-149"}}, "training.world_seed_range"),
                                   ({"test": {"world_seed": 20000}}, "test.world_seed"),
                                   ({"test": {"seed": 20000}}, "test.seed")):
            uses, problems = declaration_uses("d.json", declaration)
            self.assertEqual(uses, [], field)
            self.assertEqual(problems, [f"d.json {field}: a seed field the seed registry test cannot read; write world "
                                        "seeds as world_seeds [first, last] or 'first-last', or as world_seeds_from or "
                                        "first_world_seed with an episodes count beside it, and name a seed that is "
                                        "not a world seed agent_seed or agent_seeds"])
        self.assertEqual(mentioned_seeds("on worlds 40100-40119; Seeds 20000-20019 are not reused; agent seeds 0 to 3"),
                         [range(40100, 40120), range(20000, 20020)])


class SyntheticPlanTests(unittest.TestCase):
    """New files in a copy of configs/ must fail until the registry covers them."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        shutil.copytree(ROOT / "configs", self.root / "configs")
        self.registry = read_json(ROOT / REGISTRY)

    def tearDown(self):
        self.directory.cleanup()

    def write(self, relative: str, payload) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def retain(self, run: str, mode: str, first: int, requested: int) -> None:
        self.write(f"{run}/metadata.json", {"mode": mode, "first_world_seed": first, "episodes_requested": requested})

    def plan(self, train_seed: int, test_seed: int) -> dict:
        return {"schema": 1, "name": "new-plan", "slots": 1, "defaults": {"agent_seed": 0, "max_steps": 60, "pause": 0},
                "jobs": [{"name": "new-train", "kind": "train", "condition": "unmasked", "algorithm": "q_learning",
                          "episodes": 5, "first_world_seed": train_seed, "epsilon": 0.6, "epsilon_end": 0.1},
                         {"name": "new-test", "kind": "evaluate", "random_baseline": True, "condition": "unmasked",
                          "episodes": 6, "first_world_seed": test_seed, "chunks": 2}]}

    def register(self, experiment: str, *ranges: tuple[int, int, str]) -> None:
        self.registry["experiments"].setdefault(experiment, {"description": "a synthetic experiment", "sources": []})
        self.registry["experiments"][experiment]["sources"].append("configs/zz_new_plan.json")
        for first, last, kind in ranges:
            self.registry["ranges"].append({"first": first, "last": last, "kind": kind, "purpose": "synthetic",
                                            "experiments": [experiment], "reuse": {"allowed": False, "by": "no one"}})

    def test_the_copied_configs_alone_pass(self):
        self.assertEqual(all_problems(self.root, self.registry), [])

    def test_a_new_plan_on_undeclared_seeds_fails_until_it_is_registered(self):
        self.write("configs/zz_new_plan.json", self.plan(70000, 70100))
        problems = all_problems(self.root, self.registry)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("configs/zz_new_plan.json is not in configs/seed_registry.json (it uses world seeds "
                      "70000-70004, 70100-70105): add it to the sources of the experiment", problems[0])
        # Listed as a source, it still fails on every seed no range of its experiment covers, chunk by chunk.
        self.register("new_experiment", (70000, 70004, "training"))
        problems = all_problems(self.root, self.registry)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("configs/zz_new_plan.json (run new-test-c1, run new-test-c2) uses world seeds 70100-70105, "
                      "but no range in configs/seed_registry.json names experiment 'new_experiment' for them. Choose "
                      "unused seeds and add a range for them", problems[0])
        self.registry["ranges"].append({"first": 70100, "last": 70105, "kind": "test", "purpose": "synthetic",
                                        "experiments": ["new_experiment"], "reuse": {"allowed": False, "by": "no one"}})
        self.assertEqual(all_problems(self.root, self.registry), [])

    def test_reusing_another_experiments_held_out_range_fails(self):
        self.write("configs/zz_new_plan.json", self.plan(70000, 20000))
        self.register("new_experiment", (70000, 70004, "training"))
        problems = all_problems(self.root, self.registry)
        self.assertEqual(len(problems), 3, problems)
        # The messages show the range's rule and invite no reuse it forbids.
        self.assertEqual(problems[0], "configs/zz_new_plan.json (run new-test-c1, run new-test-c2) uses world seeds "
                                      "20000-20005, but no range in configs/seed_registry.json names experiment "
                                      "'new_experiment' for them; they fall in 20000-20019 (test of "
                                      "resource_masking_experiment; no reuse: no one; "
                                      "configs/initial_q_experiment.json declares it does not reuse them, because "
                                      "their failure analysis motivated it). Choose unused seeds and add a range for "
                                      "them")
        self.assertTrue(problems[1].startswith(
            "World seeds 20000-20005 are used by 2 experiments, but the range that holds them, 20000-20019 (test of "
            "resource_masking_experiment), does not name new_experiment: new_experiment (configs/zz_new_plan.json run "
            "new-test-c1, configs/zz_new_plan.json run new-test-c2); resource_masking_experiment ("), problems[1])
        self.assertTrue(problems[1].endswith("). Choose unused seeds; that range allows no reuse"), problems[1])
        self.assertIn("Held-out range 20000-20019 (test of resource_masking_experiment) is used by 2 experiments",
                      problems[2])

    def test_training_on_a_held_out_range_fails_even_inside_its_experiment(self):
        self.write("configs/zz_new_plan.json", self.plan(40000, 40000))
        self.registry["experiments"]["initial_q_experiment"]["sources"].append("configs/zz_new_plan.json")
        problems = all_problems(self.root, self.registry)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("configs/zz_new_plan.json run new-train trains on world seeds 40000-40004 of held-out range "
                      "40000-40019 (test of initial_q_experiment)", problems[0])

    def test_seeds_mentioned_only_in_prose_must_be_registered(self):
        self.write("configs/zz_new_plan.json", {"name": "zz", "declared": "today", "note": "spare worlds 71000-71009"})
        self.registry["experiments"]["initial_q_experiment"]["sources"].append("configs/zz_new_plan.json")
        self.assertEqual(all_problems(self.root, self.registry),
                         ["configs/zz_new_plan.json mentions world seeds 71000-71009, which no range in "
                          "configs/seed_registry.json covers; register them, or reword the text if they are not world "
                          "seeds"])

    def test_an_unregistered_schedule_on_used_seeds_reports_only_its_folder(self):
        # The initial-Q schedule retained before its folder is registered: no check invents an experiment per run.
        _, units = parse_plan((ROOT / "configs/initial_q_plan.json").read_text(encoding="utf-8"))
        self.assertEqual(len(units), 26)
        schedule = "results/pc_resource_runs/abc1234/iq-experiment"
        for unit in units:
            mode = "online_training" if unit.job["kind"] == "train" else "frozen_evaluation"
            self.retain(f"{schedule}/{unit.name}", mode, unit.first_world_seed, unit.episodes)
        train = next(unit for unit in units if unit.job["kind"] == "train")
        self.retain(f"{schedule}/{train.name}-part1", "online_training", train.first_world_seed + 10,
                    train.episodes - 10)
        self.assertEqual(all_problems(self.root, self.registry),
                         [f"{schedule} holds retained runs on world seeds 100-149, 40000-40019, 40100-40119, but no "
                          f"experiment in configs/seed_registry.json lists it among its sources: add {schedule} itself "
                          "to the sources of the experiment it belongs to and register the seeds; a folder above it "
                          "would also claim every later run under it"])
        self.registry["experiments"]["initial_q_experiment"]["sources"].append(schedule)
        self.assertEqual(all_problems(self.root, self.registry), [])

    def test_testing_frozen_policies_on_their_training_worlds_fails(self):
        plan = self.plan(63000, 63000)
        plan["jobs"][1].pop("random_baseline")
        plan["jobs"][1]["policy_from"] = "new-train"
        self.write("configs/stone_pickaxe_plan.json", plan)
        self.assertEqual(all_problems(self.root, self.registry), [
            f"configs/stone_pickaxe_plan.json run new-test-{chunk} evaluates on world seeds {seeds} of training range "
            "63000-63049 (training of stone_pickaxe_experiment); an evaluation never runs on training worlds: choose "
            "validation, development or test seeds" for chunk, seeds in (("c1", "63000-63002"), ("c2", "63003-63005"))])
        plan["jobs"][1]["first_world_seed"] = 64000
        self.write("configs/stone_pickaxe_plan.json", plan)
        self.assertEqual(all_problems(self.root, self.registry), [])

    def test_a_helper_gate_on_training_worlds_fails_even_when_named_on_their_range(self):
        self.write("configs/zz_new_plan.json", self.plan(62000, 63000))
        self.registry["experiments"]["stone_helper_gate"]["sources"].append("configs/zz_new_plan.json")
        problems = all_problems(self.root, self.registry)
        self.assertEqual(len(problems), 3, problems)
        self.assertEqual(problems[0], "configs/zz_new_plan.json (run new-test-c1, run new-test-c2) uses world seeds "
                                      "63000-63005, but no range in configs/seed_registry.json names experiment "
                                      "'stone_helper_gate' for them; they fall in 63000-63049 (training of "
                                      "stone_pickaxe_experiment; reuse allowed: training runs of the stone experiment "
                                      "only, including a retraining after a code change; never validation or test). "
                                      "Choose unused seeds and add a range for them, or, only if the reuse rule of the "
                                      "range that holds them allows this use by 'stone_helper_gate', name "
                                      "'stone_helper_gate' on that range")
        # Naming the gate on the range against its rule does not make the evaluation pass.
        next(entry for entry in self.registry["ranges"] if entry["first"] == 63000)["experiments"].append(
            "stone_helper_gate")
        self.assertEqual(all_problems(self.root, self.registry), [
            f"configs/zz_new_plan.json run new-test-{chunk} evaluates on world seeds {seeds} of training range "
            "63000-63049 (training of stone_pickaxe_experiment, stone_helper_gate); an evaluation never runs on "
            "training worlds: choose validation, development or test seeds"
            for chunk, seeds in (("c1", "63000-63002"), ("c2", "63003-63005"))])

    def test_a_checkpoints_seeds_are_written_as_prose_not_in_a_world_seed_field(self):
        declaration = {"name": "stone", "training": {"world_seeds_from": 63000, "episodes": {"q_learning": 50}},
                       "test": {"world_seeds": [64000, 64019]},
                       "stage1": {"policy": "results/pc_resource_runs/87673b8/mx-unmasked-q-train/policy.json",
                                  "sha256": "0" * 64,
                                  "provenance": "trained on worlds 100-149 at 87673b8, tested on worlds 20000-20019"}}
        self.write("configs/stone_pickaxe_experiment.json", declaration)
        self.assertEqual(all_problems(self.root, self.registry), [])
        # As a world-seed field, the checkpoint's seeds count as the stone experiment's own, and the messages say so.
        declaration["stage1"]["trained_on_world_seeds"] = [100, 149]
        self.write("configs/stone_pickaxe_experiment.json", declaration)
        problems = all_problems(self.root, self.registry)
        self.assertEqual(len(problems), 2, problems)
        hint = ("If these seeds belong to another experiment's checkpoint or runs (for example a stage-1 policy), "
                "write them in configs/stone_pickaxe_experiment.json as prose, such as 'trained on worlds 100-149 at "
                "87673b8', instead of a world-seed field; never add 'stone_pickaxe_experiment' to a range for them")
        self.assertTrue(problems[0].startswith(
            "configs/stone_pickaxe_experiment.json (stage1.trained_on_world_seeds) uses world seeds 100-149, but no "
            "range in configs/seed_registry.json names experiment 'stone_pickaxe_experiment' for them; they fall in "
            "100-149 (training of resource_masking_experiment, initial_q_experiment; reuse allowed: "), problems[0])
        self.assertTrue(problems[0].endswith(". " + hint), problems[0])
        self.assertTrue(problems[1].startswith(
            "World seeds 100-149 are used by 3 experiments, but the range that holds them, 100-149 (training of "
            "resource_masking_experiment, initial_q_experiment), does not name stone_pickaxe_experiment: "),
            problems[1])
        self.assertTrue(problems[1].endswith(", add stone_pickaxe_experiment to it. " + hint), problems[1])

    def test_a_declaration_on_first_world_seed_is_checked(self):
        self.write("configs/zz_new_plan.json", {"name": "zz", "training": {"first_world_seed": 70000, "episodes": 5},
                                                "test": {"first_world_seed": 20000, "episodes_per_policy": 20}})
        self.register("new_experiment", (70000, 70004, "training"))
        text = "\n".join(all_problems(self.root, self.registry))
        self.assertIn("configs/zz_new_plan.json (test.first_world_seed) uses world seeds 20000-20019, but no range",
                      text)
        self.assertIn("Held-out range 20000-20019 (test of resource_masking_experiment) is used by 2 experiments "
                      "(new_experiment, resource_masking_experiment)", text)
        self.write("configs/zz_new_plan.json", {"name": "zz", "test": {"seeds": [20000, 20019]}})
        self.assertEqual(all_problems(self.root, self.registry),
                         ["configs/zz_new_plan.json test.seeds: a seed field the seed registry test cannot read; write "
                          "world seeds as world_seeds [first, last] or 'first-last', or as world_seeds_from or "
                          "first_world_seed with an episodes count beside it, and name a seed that is not a world seed "
                          "agent_seed or agent_seeds"])

    def test_a_range_naming_three_experiments_lets_two_of_them_share_part_of_it(self):
        # A third wood experiment reuses the masked budget, 100-119, by declaration; 120-149 stay with the other two.
        plan = self.plan(100, 70100)
        plan["jobs"][0]["episodes"] = 20
        self.write("configs/zz_new_plan.json", plan)
        self.register("canopy_rerun", (70100, 70105, "test"))
        problems = all_problems(self.root, self.registry)
        self.assertEqual(len(problems), 2, problems)
        self.assertTrue(problems[0].startswith(
            "configs/zz_new_plan.json (run new-train) uses world seeds 100-119, but no range in "
            "configs/seed_registry.json names experiment 'canopy_rerun' for them; they fall in 100-149 (training of "
            "resource_masking_experiment, initial_q_experiment; reuse allowed: training runs of a declared wood-stage "
            "experiment"), problems[0])
        self.assertTrue(problems[0].endswith("Choose unused seeds and add a range for them, or, only if the reuse rule "
                                             "of the range that holds them allows this use by 'canopy_rerun', name "
                                             "'canopy_rerun' on that range"), problems[0])
        self.assertTrue(problems[1].startswith(
            "World seeds 100-119 are used by 3 experiments, but the range that holds them, 100-149 (training of "
            "resource_masking_experiment, initial_q_experiment), does not name canopy_rerun: canopy_rerun "
            "(configs/zz_new_plan.json run new-train); initial_q_experiment ("), problems[1])
        self.assertTrue(problems[1].endswith(
            "Choose unused seeds, or, only if the reuse rule of that range allows this use, add canopy_rerun to it"),
            problems[1])
        next(entry for entry in self.registry["ranges"] if entry["first"] == 100)["experiments"].append("canopy_rerun")
        self.assertEqual(all_problems(self.root, self.registry), [])

    def test_a_plan_in_a_subfolder_of_configs_is_read_whether_registered_or_not(self):
        self.write("configs/sub/zz.json", self.plan(20000, 70100))
        self.assertEqual(all_problems(self.root, self.registry), [
            "configs/sub/zz.json is not in configs/seed_registry.json (it uses world seeds 20000-20004, 70100-70105): "
            "add it to the sources of the experiment it belongs to and register the world seeds it uses, or list it in "
            "not_world_seed_configs if it uses no Minecraft world seeds",
            "configs/sub/zz.json run new-train trains on world seeds 20000-20004 of held-out range 20000-20019 "
            "(test of resource_masking_experiment); a held-out range is never trained on"])
        self.registry["experiments"]["initial_q_experiment"]["sources"].append("configs/sub/zz.json")
        text = "\n".join(all_problems(self.root, self.registry))
        self.assertIn("configs/sub/zz.json (run new-test-c1, run new-test-c2, run new-train) uses world seeds "
                      "20000-20004, 70100-70105, but no range", text)
        self.assertIn("Held-out range 20000-20019 (test of resource_masking_experiment) is used by 2 experiments", text)
        self.assertIn("configs/sub/zz.json run new-train trains on world seeds 20000-20004 of held-out range", text)

    def test_a_committed_plan_outside_configs_fails(self):
        self.write("plans/zz.json", self.plan(20000, 70100))
        self.write("results/x/bench/plan.json", self.plan(20000, 70100))
        self.write("tools/package.json", {"name": "tools"})
        (self.root / "tools" / "broken.json").write_text("{", encoding="utf-8")
        paths = ["configs/initial_q_plan.json", "plans/zz.json", "results/x/bench/plan.json", "tools/broken.json",
                 "tools/deleted.json", "tools/package.json"]
        self.assertEqual(stray_plan_problems(self.root, paths),
                         ["plans/zz.json has jobs like a resource-parallel plan but lies outside configs/, so the seed "
                          "registry test never reads its seeds: move it into configs/ and register it"])

    def test_a_source_the_survey_never_reads_is_refused(self):
        registry = copy.deepcopy(self.registry)
        registry["experiments"]["initial_q_experiment"]["sources"] += ["plans/zz.json", "results", "configs"]
        registry["experiments"]["stone_development"]["planned_sources"] = ["results/stone_runs/plan.json",
                                                                           "configs/notes.md"]
        registry["not_world_seed_configs"]["docs/x.json"] = "not read"
        text = "\n".join(registry_problems(registry, ROOT))
        for path in ("plans/zz.json", "results", "configs"):
            self.assertIn(f"experiment initial_q_experiment: sources entry {path} must name a file or folder inside "
                          "configs/ or results/", text)
        for path in ("results/stone_runs/plan.json", "configs/notes.md"):
            self.assertIn(f"experiment stone_development: planned_sources entry {path} is never read", text)
        self.assertIn("not_world_seed_configs names docs/x.json, but the seed registry test reads only .json files "
                      "under configs/", text)

    def test_a_registry_with_overlapping_or_misused_ranges_fails(self):
        registry = copy.deepcopy(self.registry)
        registry["ranges"].insert(3, {"first": 140, "last": 160, "kind": "training", "purpose": "synthetic",
                                      "experiments": ["initial_q_experiment"], "reuse": {"allowed": True, "by": "x"}})
        registry["ranges"][-1]["experiments"].append("initial_q_experiment")
        registry["ranges"][-1]["reuse"]["allowed"] = True
        text = "\n".join(registry_problems(registry, ROOT))
        self.assertIn("ranges 100-149 (training of resource_masking_experiment, initial_q_experiment) and 140-160 "
                      "(training of initial_q_experiment) overlap", text)
        self.assertIn("range 64100-64119: a replacement test range can never allow reuse", text)


class RetainedRunTests(unittest.TestCase):
    """Synthetic retained runs: their folder must be registered and each continuation must follow the declared rule."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        (self.root / "configs").mkdir()
        self.registry = read_json(ROOT / REGISTRY)

    def tearDown(self):
        self.directory.cleanup()

    def run_folder(self, name: str, first: int, requested: int, played: int, parent: str = "results/new_runs",
                   mode: str = "online_training") -> None:
        run = self.root / parent / name
        run.mkdir(parents=True)
        (run / "metadata.json").write_text(json.dumps({"mode": mode, "first_world_seed": first,
                                                       "episodes_requested": requested}), encoding="utf-8")
        for number in range(1, played + 1):
            (run / f"episode-{number:04d}").mkdir()
            (run / f"episode-{number:04d}" / "outcome.json").write_text(
                json.dumps({"world_seed": first + number - 1}), encoding="utf-8")

    def test_an_unregistered_results_folder_fails(self):
        self.run_folder("x-train", 100, 10, 10)
        self.assertEqual(all_problems(self.root, self.registry),
                         ["results/new_runs holds retained runs on world seeds 100-109, but no experiment in "
                          "configs/seed_registry.json lists it among its sources: add results/new_runs itself to the "
                          "sources of the experiment it belongs to and register the seeds; a folder above it would "
                          "also claim every later run under it"])
        self.registry["experiments"]["initial_q_experiment"]["sources"].append("results/new_runs")
        self.assertEqual(all_problems(self.root, self.registry), [])

    def test_a_folder_claimed_by_a_source_above_it_is_pointed_to_its_own_experiment(self):
        self.run_folder("stone-survey", 61000, 20, 2, parent="results/stone_runs/c2", mode="frozen_evaluation")
        self.assertEqual(all_problems(self.root, self.registry),
                         ["results/stone_runs/c2 holds retained runs on world seeds 61000-61019, but no experiment in "
                          "configs/seed_registry.json lists it among its sources: add results/stone_runs/c2 itself to "
                          "the sources of the experiment it belongs to and register the seeds; a folder above it "
                          "would also claim every later run under it"])
        # Registering the folder above instead silently claims the stone experiment's later runs for development.
        self.registry["experiments"]["stone_development"]["sources"].append("results/stone_runs")
        self.run_folder("st-q-s0-train", 63000, 40, 2, parent="results/stone_runs/c8")
        self.assertEqual(all_problems(self.root, self.registry),
                         ["results/stone_runs/c8/st-q-s0-train uses world seeds 63000-63039, but no range in "
                          "configs/seed_registry.json names experiment 'stone_development' for them; they fall in "
                          "63000-63049 (training of stone_pickaxe_experiment; reuse allowed: training runs of the "
                          "stone experiment only, including a retraining after a code change; never validation or "
                          "test). Choose unused seeds and add a range for them, or, if results/stone_runs/c8 belongs "
                          "to stone_pickaxe_experiment, list results/stone_runs/c8 itself in the sources of "
                          "stone_pickaxe_experiment (the longest source path decides the owner, and results/stone_runs "
                          "now gives it to 'stone_development'), or, only if the reuse rule of the range that holds "
                          "them allows this use by 'stone_development', name 'stone_development' on that range"])
        self.registry["experiments"]["stone_pickaxe_experiment"]["sources"].append("results/stone_runs/c8")
        self.assertEqual(all_problems(self.root, self.registry), [])

    def test_a_continuation_starts_after_the_attempted_episodes(self):
        self.run_folder("x-train", 100, 10, 3)
        self.run_folder("x-train-part1", 103, 7, 2)
        self.run_folder("x-train-part2", 105, 5, 5)
        self.assertEqual(retained_run_problems(self.root), [])
        shutil.rmtree(self.root / "results" / "new_runs" / "x-train-part2")
        self.run_folder("x-train-part2", 106, 4, 4)
        self.assertEqual(retained_run_problems(self.root),
                         ["results/new_runs/x-train-part2 starts at world seed 106 with 4 episodes, but after the 5 "
                          "episodes attempted in x-train, x-train-part1 it must start at 105 with 5"])

    def test_an_episode_on_the_wrong_world_fails(self):
        self.run_folder("x-test", 20000, 2, 2)
        outcome = self.root / "results" / "new_runs" / "x-test" / "episode-0002" / "outcome.json"
        outcome.write_text(json.dumps({"world_seed": 20005}), encoding="utf-8")
        self.assertEqual(retained_run_problems(self.root),
                         ["results/new_runs/x-test/episode-0002/outcome.json records world seed 20005, but "
                          "metadata.json puts episode 2 on 20001"])


if __name__ == "__main__":
    unittest.main()
