"""The course builder's converter (parkour.builder): a scan of a built course gives the course back."""
import io
import json

import numpy as np
import pytest

from parkour import builder as B
from parkour import courses as C
from parkour import neo_assist as NA
from parkour.paths import CONFIGS
from parkour.train import held_out_courses

ORIGIN = (100, 60, -40)  # where the course is "built" in the world
FULL = [[0, 0, 0, 1, 1, 1]]


def ends(c, origin=ORIGIN):
    p0, pl = c.platforms[0], c.platforms[-1]
    return ((int(p0[0]) + origin[0], int(p0[4]) + origin[1], int(p0[2]) + origin[2]),
            (int(pl[0]) + origin[0], int(pl[4]) + origin[1], int(pl[2]) + origin[2]))


def world_platforms(c, origin):
    return [[a + origin[0], b + origin[0], d + origin[2], e + origin[2], y + origin[1]]
            for a, b, d, e, y in np.asarray(c.platforms).tolist()]


def same_course(c, d, o):
    """The converted course d (origin o) is c built at ORIGIN: platforms in order, walls, ice, slime, shapes, ladders."""
    shift = np.array(ORIGIN) - np.array(o)
    return (world_platforms(d, o) == world_platforms(c, ORIGIN)
            and sorted(map(tuple, (np.asarray(d.extra) - shift).tolist())) == sorted(map(tuple, np.asarray(c.extra).reshape(-1, 3).tolist()))
            and d.ice == sorted(c.ice) and d.slime == sorted(c.slime) and d.shapes == c.shapes
            and sorted(map(tuple, (np.asarray(d.ladders)[:, :3] - shift).tolist() if len(d.ladders) else []))
            == sorted(map(tuple, np.asarray(c.ladders).reshape(-1, 4)[:, :3].tolist())))


def course_sets():
    levels = C.load_levels()
    out = {"standard": held_out_courses(levels, 10, 10, family=2),
           "obstacles": held_out_courses(levels, 10, 10, obstacle_prob=0.5, family=5),
           "single neo": NA.single_neo_courses(40, 32), "two neos": NA.two_neo_courses(52, 32)}
    extra = {L["name"]: [C._spec_from(d) for d in L["specs"]]
             for L in json.loads((CONFIGS / "jump_levels_extra.json").read_text())["levels"]}
    for name, specs in extra.items():
        rng = np.random.default_rng([2026, 2, 7])
        out[name] = [C.generate(rng, levels + [specs], 6, 10, seed=i) for i in range(20)]
    return out


@pytest.fixture(scope="module")
def sets():
    return course_sets()


def test_a_scan_of_a_generated_course_gives_it_back(sets):
    for name, cs in sets.items():
        exact = 0
        for c in cs:
            start, goal = ends(c)
            d, o, _ = B.scan_to_course(B.course_blocks(c, ORIGIN), start, goal)  # every course converts
            exact += same_course(c, d, o)
            assert d.n <= C.MAX_PLATFORMS and np.asarray(d.platforms).min() >= 0
            assert tuple(o) == tuple(int(v) for v in o)
        assert exact >= 0.9 * len(cs), (name, exact, len(cs))  # the rest: another chain through the same platforms


def test_via_points_fix_the_route(sets):
    for name in ("standard", "single neo", "two neos", "slime bounce", "ladder climb"):
        for c in sets[name]:
            start, goal = ends(c)
            via = [(int(p[0]) + ORIGIN[0], int(p[4]) + ORIGIN[1], int(p[2]) + ORIGIN[2]) for p in c.platforms[1:-1]]
            d, o, _ = B.scan_to_course(B.course_blocks(c, ORIGIN), start, goal, via=via)
            assert same_course(c, d, o), name


def test_converted_course_keeps_the_neo_skill_s_blocks(sets):
    c = sets["two neos"][0]
    start, goal = ends(c)
    reply = B.convert({"blocks": B.course_blocks(c, ORIGIN), "start": start, "goal": goal})
    from parkour.neo_dagger import neo_takeoffs
    assert reply["about"]["neos"] == neo_takeoffs(c) and reply["course"]["skill_cps"] == NA.skill_cps(c)
    assert reply["about"]["router_ok"] and len(reply["origin"]) == 3


def test_serve_answers_each_line():
    c = NA.single_neo_courses(40, 1)[0]
    start, goal = ends(c)
    inp = io.StringIO(json.dumps({"blocks": B.course_blocks(c, ORIGIN), "start": start, "goal": goal}) + "\n"
                      + "not json\n" + json.dumps({"blocks": [], "start": [0, 0, 0], "goal": [1, 0, 0]}) + "\n")
    out = io.StringIO()
    B.serve(inp, out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert len(replies) == 3 and "course" in replies[0] and "error" in replies[1] and "error" in replies[2]


def build(*cells):
    return [[x, y, z, name, props, boxes] for x, y, z, name, props, boxes in cells]


def plain(x, y, z, name="stone"):
    return (x, y, z, name, {}, FULL)


def line(n=3, gap=2, y=10, name="stone"):
    """n 1x1 platforms along +x, ``gap`` blocks of air apart."""
    return [plain(i * (gap + 1), y, 0, name) for i in range(n)]


def test_refusals_name_the_block_and_where():
    with pytest.raises(B.BuildError) as e:
        B.scan_to_course(build(*line(), plain(1, 12, 5, "dirt")), (0, 10, 0), (6, 10, 0))
    assert e.value.refused == [[1, 12, 5, "dirt", "not a block the simulator models"]] and "dirt" in str(e.value)
    fence = next(s for s in B._shapes() if s["name"] == "fence_post")
    connected = fence["boxes"] + [[0.4375, 0, 0, 0.5625, 1.5, 0.5]]  # a post with a side arm: a connected fence
    with pytest.raises(B.BuildError) as e:
        B.scan_to_course(build(*line(), (3, 10, 4, "oak_fence", {"north": True}, connected)), (0, 10, 0), (6, 10, 0))
    assert e.value.refused[0][:4] == [3, 10, 4, "oak_fence"] and "not modelled" in e.value.refused[0][4]
    with pytest.raises(B.BuildError, match="touch"):
        B.scan_to_course(build(*line(), plain(1, 10, 1)), (0, 10, 0), (6, 10, 0))  # diagonal to the start
    with pytest.raises(B.BuildError, match="no chain"):
        B.scan_to_course(build(*line(gap=7)), (0, 10, 0), (16, 10, 0))
    with pytest.raises(B.BuildError, match="no platform at the start"):
        B.scan_to_course(build(*line()), (0, 11, 0), (6, 10, 0))
    with pytest.raises(B.BuildError, match="walls and roofs must be stone"):
        B.scan_to_course(build(*line(), plain(4, 11, 3, "ice"), plain(4, 12, 3, "ice")), (0, 10, 0), (6, 10, 0))
    with pytest.raises(B.BuildError, match="no block behind it"):
        ladder = next(s for s in B._shapes() if s["name"] == "ladder_north")
        B.scan_to_course(build(*line(), (3, 13, 6, "ladder", {"facing": "north"}, ladder["boxes"])), (0, 10, 0), (6, 10, 0))
    many = line(n=17, gap=1)
    with pytest.raises(B.BuildError, match="at most 16"):
        B.scan_to_course(build(*many), (0, 10, 0), (32, 10, 0))


def test_walls_shapes_and_kinds():
    # Start, a neo-like wall (two stone blocks on top of each other), a slab, an ice platform, the goal.
    slab = next(s for s in B._shapes() if s["name"] == "bottom_slab")
    cells = build(plain(0, 10, 0), plain(2, 11, 0), plain(2, 12, 0), (3, 10, 0, "stone_slab", dict(slab["properties"]),
                                                                       slab["boxes"]),
                  plain(5, 10, 0, "ice"), plain(8, 10, 0))
    d, o, w = B.scan_to_course(cells, (0, 10, 0), (8, 10, 0))
    assert d.n == 4 and d.shapes == {1: "bottom_slab"} and d.ice == [2]
    assert sorted(map(tuple, (np.asarray(d.extra) + o).tolist())) == [(2, 11, 0), (2, 12, 0)]
    # A platform off the route is a wall; a slab that is not a rectangle is a wall too.
    cells = build(*line(), plain(3, 10, 8), plain(20, 10, 20), plain(21, 10, 20), plain(20, 10, 21))
    d, o, w = B.scan_to_course(cells, (0, 10, 0), (6, 10, 0))
    assert d.n == 3 and len(d.extra) == 4 and any("not a rectangle" in x for x in w) and any("off the route" in x for x in w)


def test_the_bot_s_kit_is_the_converter_s():
    import re
    from parkour.paths import BRIDGE
    src = (BRIDGE / "builder_bot.js").read_text(encoding="utf-8")
    js = re.search(r"const KIT = \[([^\]]*)\]", src, re.S).group(1)
    assert tuple(re.findall(r"'([a-z_]+)'", js)) == B.KIT
    markers = re.search(r"const MARKERS = \{([^}]*)\}", src).group(1)
    assert dict(re.findall(r"([a-z_]+): '([a-z]+)'", markers)) == B.MARKERS
    for name in B.KIT:  # every block of the kit is one the converter takes
        if name in ("stone", "ice", "slime_block", "ladder") or name in B.MARKERS:
            continue
        assert any(s["block"] == name for s in B._shapes()), name


def test_marker_blocks_are_stone_to_the_simulator(sets):
    assert all(B.classify(name, {}, FULL) == ("full", None) for name in B.MARKERS)
    for name in ("standard", "single neo", "two neos"):
        for c in sets[name][:10]:
            start, goal = ends(c)
            mid = np.asarray(c.platforms)[len(c.platforms) // 2]
            via = (int(mid[0]) + ORIGIN[0], int(mid[4]) + ORIGIN[1], int(mid[2]) + ORIGIN[2])
            plain_blocks = B.course_blocks(c, ORIGIN)
            marked = {start: "gold_block", goal: "diamond_block", via: "emerald_block"}
            blocks = [[x, y, z, marked.get((x, y, z), n), p, b] for x, y, z, n, p, b in plain_blocks]
            assert sorted(b[3] for b in blocks if b[3] in B.MARKERS) == sorted(B.MARKERS), name
            d, o, w = B.scan_to_course(blocks, start, goal, via=[via])
            e, q, v = B.scan_to_course(plain_blocks, start, goal, via=[via])
            assert d.to_json() == e.to_json() and tuple(o) == tuple(q) and w == v, name


def test_the_panel_s_commands():
    """bridge/builder_panel.js: the commands that build the panel are valid 1.21.4 commands of at most 256
    characters, each button sends "builder:COMMAND PLAYER" to the bot, and the panel stays off the spawn pad."""
    import re
    import subprocess
    from parkour.paths import BRIDGE, NODE
    if not NODE.exists():
        pytest.skip("the portable Node runtime is missing (setup.cmd)")
    code = ("const P = require('./builder_panel');"
            "console.log(JSON.stringify({commands: P.commands('ParkourBot'), cells: P.cells(), pad: P.PAD, buttons: P.BUTTONS,"
            " parsed: ['builder:run:list Ethan_1', 'builder:help Ethan', 'builder:run ', 'hello', '[builder] x'].map(P.parse)}))")
    out = json.loads(subprocess.run([str(NODE), "-e", code], cwd=BRIDGE, capture_output=True, text=True, check=True).stdout)
    commands, cells, pad = out["commands"], out["cells"], out["pad"]
    assert all(c.endswith(" replace") and len(c) <= 256 for c in commands)
    fills = [c for c in commands if c.startswith("/fill ")]
    assert commands[:len(fills)] == fills and all(c.startswith("/setblock ") for c in commands[len(fills):])
    boxes = [list(map(int, re.match(r"/fill (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) minecraft:air", c).groups())) for c in fills]
    for x, y, z, _ in cells:  # the clearing covers the panel and stays above the pad's floor
        assert any(b[0] <= x <= b[3] and b[1] <= y <= b[4] and b[2] <= z <= b[5] for b in boxes)
    assert min(b[1] for b in boxes) >= pad[1]
    commands = commands[len(fills):]
    kinds = [re.match(r"/setblock -?\d+ \d+ -?\d+ minecraft:([a-z_]+)", c).group(1) for c in commands]
    n = len(out["buttons"])
    assert kinds[:n] == ["command_block"] * n and sorted(kinds[n:]) == ["oak_sign"] * n + ["stone_button"] * n
    sent = []
    for c in commands[:n]:
        text = json.loads(re.fullmatch(r".*\{Command:'tellraw ParkourBot (.*)'\} replace", c).group(1))
        assert text[1] == {"selector": "@p[distance=..8,name=!ParkourBot]"}
        sent.append(text[0])
    assert sent == [f"builder:{b[2]} " for b in out["buttons"]]
    for c in commands[n:]:
        if "oak_sign" in c:  # four lines of JSON text in SNBT strings
            lines = re.fullmatch(r".*messages:\[(.*)\]\},is_waxed:1b\} replace", c).group(1)
            assert [json.loads(s) for s in re.findall(r"'([^']*)'", lines)][2:] == ["", ""]
    where = {tuple(c[:3]): c[3] for c in cells}
    assert len(where) == len(cells) == 3 * n
    faces = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    for (x, y, z), what in where.items():
        assert not (y == pad[1] - 1 and abs(x - pad[0]) <= 2 and abs(z - pad[2]) <= 2)  # off the pad's floor
        assert what == "command_block" or where[(x, y, z - 1) if what == "button" else (x, y - 1, z)] == "command_block"
        # a pressed button powers its command block, which powers any command block touching it: none may
        touching = [where.get((x + a, y + b, z + c)) for a, b, c in faces]
        assert what != "command_block" or "command_block" not in touching
        assert what != "button" or touching.count("command_block") == 1
    assert out["parsed"] == [{"cmd": "!run", "opt": "list", "player": "Ethan_1"}, {"cmd": "!help", "opt": "", "player": "Ethan"},
                             {"cmd": "!run", "opt": "", "player": ""}, None, None]


def test_the_builder_s_server_turns_command_blocks_on(tmp_path, monkeypatch):
    from parkour import build_session
    from parkour import server as S
    for name in ("jar", "java"):
        (tmp_path / name).write_text("")
    monkeypatch.setattr(S, "SERVER_DIR", tmp_path)
    monkeypatch.setattr(S, "SERVER_JAR", tmp_path / "jar")
    monkeypatch.setattr(S, "JAVA", tmp_path / "java")
    monkeypatch.setattr(S, "eula_accepted", lambda: True)
    props = lambda: dict(line.split("=", 1) for line in (tmp_path / "server.properties").read_text().splitlines())  # noqa: E731
    S.prepare(25570)
    assert "enable-command-block" not in props()  # the live runs' server is as before
    S.prepare(25571, build_session.BUILDER_PROPERTIES)
    assert props()["enable-command-block"] == "true" and props()["server-port"] == "25571" and props()["gamemode"] == "adventure"


def test_demo_courses_scan_back_exactly():
    for c in B.demo_courses():
        start, goal = ends(c)
        reply = B.convert({"blocks": B.course_blocks(c, ORIGIN), "start": start, "goal": goal,
                           "expect": {"course": c.to_json(), "origin": list(ORIGIN)}})
        assert reply["matches_expected"], reply.get("warnings")
