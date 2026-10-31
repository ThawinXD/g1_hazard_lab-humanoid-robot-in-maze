"""Unit tests for Option B custom training terrains and Hazard-Aware Pilot (no simulator required)."""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pilot import (  # noqa: E402
    CELL_LOW_RAMP,
    CELL_SPIKE_LANE,
    CELL_SPIKE_SLALOM,
    CELL_STEEP_RAMP,
    CELL_TRENCH,
    Pilot,
)
from g1_maze_lab.terrains.hazard_maze import (  # noqa: E402
    HazardMazeTerrainCfg,
    hazard_maze_terrain,
)
from g1_maze_lab.terrains import GENERATED, HAZARDS  # noqa: E402


def _make_scan_grid(cx: float, cy: float) -> torch.Tensor:
    """Create a 33x33 height-scanner xy grid (3.2 m x 3.2 m, 0.1 m spacing) centered at (cx, cy)."""
    xs = torch.linspace(cx - 1.6, cx + 1.6, 33)
    ys = torch.linspace(cy - 1.6, cy + 1.6, 33)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1).unsqueeze(0)


def test_pilot_passes_server_upload_checks():
    """Verify pilot.py compiles, defines class Pilot, and contains no forbidden map references."""
    pilot_path = os.path.join(ROOT, "pilot.py")
    data = open(pilot_path, "rb").read()
    compile(data.decode("utf-8"), "pilot.py", "exec")
    assert re.search(rb"^class\s+Pilot\b", data, re.M) is not None
    map_re = rb"\b(GENERATED|HAZARDS|HARD_CELLS)\b|g1_maze_lab\.terrains|from\s+g1_maze_lab\s+import\s+[^\n]*\bterrains\b"
    assert re.search(map_re, data) is None, "pilot.py must not reference ground-truth map symbols"


def test_pilot_classifies_all_hazard_types_and_safe_lanes():
    """Verify Pilot._classify_cell_from_scan distinguishes trenches, low/steep ramps, spike lanes, and slaloms."""
    p = Pilot(num_envs=1, num_cells=7, cell_size=2.0, device="cpu")
    # Cell (1, 0) centered at (2.0, 0.0), corridor along x (North/South walls closed)
    p.ap.walls[0, 1, 0, 1] = 1
    p.ap.walls[0, 1, 0, 3] = 1
    p.ap.walls[0, 1, 0, 0] = 0
    p.ap.walls[0, 1, 0, 2] = 0

    scan_xy = _make_scan_grid(2.0, 0.0)
    dx = scan_xy[0, :, 0] - 2.0
    dy = scan_xy[0, :, 1] - 0.0

    # 1. Trench (z = -1.0 m across |dx| < 0.10)
    z_trench = torch.where((torch.abs(dx) < 0.12) & (torch.abs(dy) < 0.85), -1.0, 0.0)
    p._classify_cell_from_scan(0, 1, 0, scan_xy[0], z_trench, fallback_axis=0)
    assert int(p.cell_type[0, 1, 0].item()) == CELL_TRENCH

    # 2. Steep Ramp (z = 0.28 m across full corridor width)
    p.cell_type[0, 1, 0] = 0
    z_steep = torch.where((torch.abs(dx) < 0.50) & (torch.abs(dy) < 0.85), 0.28, 0.0)
    p._classify_cell_from_scan(0, 1, 0, scan_xy[0], z_steep, fallback_axis=0)
    assert int(p.cell_type[0, 1, 0].item()) == CELL_STEEP_RAMP

    # 3. Low Ramp (z = 0.15 m across full corridor width)
    p.cell_type[0, 1, 0] = 0
    z_low = torch.where((torch.abs(dx) < 0.50) & (torch.abs(dy) < 0.85), 0.15, 0.0)
    p._classify_cell_from_scan(0, 1, 0, scan_xy[0], z_low, fallback_axis=0)
    assert int(p.cell_type[0, 1, 0].item()) == CELL_LOW_RAMP

    # 4. Single-Lane Spike Bed on +y side (dy in [0.05, 0.85]) -> safe lane must be on -y (-0.48 m)
    p.cell_type[0, 1, 0] = 0
    z_spike_pos = torch.where((torch.abs(dx) < 0.65) & (dy > 0.05) & (dy < 0.85), 0.30, 0.0)
    p._classify_cell_from_scan(0, 1, 0, scan_xy[0], z_spike_pos, fallback_axis=0)
    assert int(p.cell_type[0, 1, 0].item()) == CELL_SPIKE_LANE
    assert float(p.safe_offset[0, 1, 0, 0].item()) < -0.40

    # 5. Slalom Spike Cell: u < -0.3 has spikes on +y, u > +0.3 has spikes on -y, center gap is flat
    p.cell_type[0, 1, 0] = 0
    p.safe_offset[0, 1, 0] = 0.0
    bed1 = (dx > -0.85) & (dx < -0.30) & (dy > 0.05) & (dy < 0.85)
    bed2 = (dx > 0.30) & (dx < 0.85) & (dy < -0.05) & (dy > -0.85)
    z_slalom = torch.where(bed1 | bed2, 0.30, 0.0)
    p._classify_cell_from_scan(0, 1, 0, scan_xy[0], z_slalom, fallback_axis=0)
    assert int(p.cell_type[0, 1, 0].item()) == CELL_SPIKE_SLALOM
    assert float(p.safe_offset[0, 1, 0, 0].item()) < -0.40  # u < 0 safe on -y
    assert float(p.safe_offset[0, 1, 0, 1].item()) > 0.40   # u > 0 safe on +y


def test_pilot_persists_memory_and_reroutes_after_reset():
    """Verify Pilot.reset() keeps wall map and adds +50 penalty to failed cell so route switches."""
    cfg = HazardMazeTerrainCfg(size=(15.0, 15.0))
    cfg.seed = 2026
    hazard_maze_terrain(0.5, cfg)
    walls = GENERATED[-1]["walls"]
    hz = HAZARDS[-1]
    short_first = hz["routes"]["short"][0]
    long_first = hz["routes"]["long"][0]

    p = Pilot(num_envs=1, num_cells=7, cell_size=2.0, device="cpu")
    p.ap.walls[0] = torch.tensor(walls, dtype=torch.int8)

    # Simulate the robot failing in the first cell of the short route
    p.last_cell[0] = torch.tensor(short_first, dtype=torch.long)
    p.reset(torch.tensor([0]))

    # Wall map is preserved and short_first cell cost is penalized by +50
    assert (p.ap.walls[0] == torch.tensor(walls, dtype=torch.int8)).all()
    assert float(p.cell_cost[0, short_first[0], short_first[1]].item()) >= 50.0

    # At start (0, 0), choose_direction now picks the Long Route exit instead of the penalized Short Route exit
    start_cell = torch.tensor([[0, 0]], dtype=torch.long)
    open_sides = p.ap.walls[0:1, 0, 0] == 0
    chosen_d = int(p.choose_direction(start_cell, open_sides, torch.tensor([0])).item())
    next_cell = (int(p.ap._dirs[chosen_d, 0].item()), int(p.ap._dirs[chosen_d, 1].item()))
    assert next_cell == long_first


def test_option_b_custom_subterrains_build_valid_meshes():
    """Verify Option B custom sub-terrains (trenches, trapezoid ramps, rough boxes) generate valid meshes."""
    import importlib.util

    # Load only the trimesh builder functions from hazard_starter/env_cfg.py without launching Isaac Sim
    src_path = os.path.join(ROOT, "src", "g1_maze_lab", "tasks", "hazard_starter", "env_cfg.py")
    src = open(src_path, encoding="utf-8").read()
    start_idx = src.index("def _box_mesh")
    end_idx = src.index("@configclass\nclass HazardTrenchTerrainCfg")
    ns = {"np": np, "trimesh": __import__("trimesh")}
    exec(src[start_idx:end_idx], ns)

    class _DummyCfg:
        size = (8.0, 8.0)
        gap_width_range = (0.06, 0.24)
        pit_depth = 1.0
        inner_platform_half_width = 1.1
        corridor_ring_width = 1.35
        ramp_height_range = (0.08, 0.30)
        ramp_length = 0.75
        cell_size = 2.0
        inner_half_width = 1.0
        height_range = (0.02, 0.09)
        boxes_per_cell = 10

    cfg = _DummyCfg()
    for diff in (0.0, 0.5, 1.0):
        m_trench, o1 = ns["hazard_trench_terrain"](diff, cfg)
        m_ramp, o2 = ns["hazard_trapezoid_ramp_terrain"](diff, cfg)
        m_boxes, o3 = ns["hazard_rough_boxes_terrain"](diff, cfg)
        assert len(m_trench) == 10 and np.allclose(o1, [4.0, 4.0, 0.0])
        assert len(m_ramp) == 5 and np.allclose(o2, [4.0, 4.0, 0.0])
        assert len(m_boxes) > 20 and np.allclose(o3, [4.0, 4.0, 0.0])
