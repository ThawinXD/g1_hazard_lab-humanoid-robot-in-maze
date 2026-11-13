"""pilot_no_goal.py: pilot.py that ignores the goal it is given and always heads for the top-right cell."""

import importlib.util
import os
import re

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PATH = os.path.join(ROOT, "pilot_no_goal.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("pilot_no_goal", PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("n,start,goal", [(9, (1, 2), (6, 8)), (7, (0, 0), (6, 6)), (10, (3, 0), None)])
def test_goal_is_always_top_right(mod, n, start, goal):
    p = mod.Pilot(num_envs=1, num_cells=n, cell_size=2.0, device="cpu", start_cell=start, goal_cell=goal)
    assert p.goal_cell == (n - 1, n - 1)
    assert tuple(p.ap.goal) == (n - 1, n - 1)
    dist = p.weighted_flood_fill(p.ap.walls)[0]
    assert float(dist[n - 1, n - 1]) == 0.0
    if goal is not None and goal != (n - 1, n - 1):
        assert float(dist[goal[0], goal[1]]) > 0.0


def test_passes_server_upload_checks(tmp_path_factory):
    with open(PATH, "rb") as f:
        data = f.read()
    map_re = (
        rb"\b(GENERATED|HAZARDS|HARD_CELLS)\b|g1_maze_lab\.terrains"
        rb"|from\s+g1_maze_lab\s+import\s+[^\n]*\bterrains\b"
    )
    assert re.search(map_re, data) is None
    os.environ["CHALLENGE_DIR"] = str(tmp_path_factory.mktemp("subs"))
    server_py = os.path.join(ROOT, "scripts", "challenge_server.py")
    spec = importlib.util.spec_from_file_location("challenge_server", server_py)
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    assert server.check_python("pilot.py", data) is None  # Uploaded under the name pilot.py
