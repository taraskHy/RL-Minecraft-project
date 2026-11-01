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
    for name in B.KIT:  # every block of the kit is one the converter takes
        if name in ("stone", "ice", "slime_block", "ladder"):
            continue
        assert any(s["block"] == name for s in B._shapes()), name


def test_demo_courses_scan_back_exactly():
    for c in B.demo_courses():
        start, goal = ends(c)
        reply = B.convert({"blocks": B.course_blocks(c, ORIGIN), "start": start, "goal": goal,
                           "expect": {"course": c.to_json(), "origin": list(ORIGIN)}})
        assert reply["matches_expected"], reply.get("warnings")
