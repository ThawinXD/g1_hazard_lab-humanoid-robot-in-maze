"""Unit tests for the trench test terrain (scripts/trench_test.py), no simulator required."""

from __future__ import annotations

import numpy as np
import trimesh

from g1_maze_lab.tasks.trench_test.terrain import (
    TRENCH_WIDTHS,
    CorridorTrenchTerrainCfg,
    corridor_trench_terrain,
    trench_width,
)

SIZE = (8.0, 3.0)


def _top_z(meshes, points):
    """Height of the highest surface under each (x, y) point (vertical rays from 5 m)."""
    mesh = trimesh.util.concatenate(meshes)
    origins = np.array([[x, y, 5.0] for x, y in points])
    locs, idx, _ = mesh.ray.intersects_location(origins, np.tile([0.0, 0.0, -1.0], (len(points), 1)))
    top = np.full(len(points), -np.inf)
    for loc, i in zip(locs, idx, strict=True):
        top[i] = max(top[i], loc[2])
    return top


def test_each_row_gets_its_own_width():
    """Row r of a curriculum generator has difficulty in [r, r+1) / rows: it must map to TRENCH_WIDTHS[r]."""
    rows = len(TRENCH_WIDTHS)
    for r, w in enumerate(TRENCH_WIDTHS):
        for frac in (0.0, 0.5, 0.999):
            assert trench_width((r + frac) / rows, TRENCH_WIDTHS) == w
    assert trench_width(1.0, TRENCH_WIDTHS) == TRENCH_WIDTHS[-1]


def test_trench_spans_the_corridor_at_the_tile_centre():
    """Floor at z = 0 on both sides, a gap of exactly the row's width at x = centre, 1 m deep, wall to wall."""
    cfg = CorridorTrenchTerrainCfg(size=SIZE)
    w = TRENCH_WIDTHS[2]
    meshes, origin = corridor_trench_terrain(2.5 / len(TRENCH_WIDTHS), cfg)
    cx, cy = 0.5 * SIZE[0], 0.5 * SIZE[1]
    for y in (cy - 0.85, cy, cy + 0.85):  # Across the whole free corridor
        z = _top_z(meshes, [(cx - 0.5 * w - 0.01, y), (cx, y), (cx + 0.5 * w + 0.01, y), (cx - 1.5, y)])
        assert np.allclose(z, [0.0, -1.0, 0.0, 0.0], atol=1e-6), f"surface heights {z} at y = {y:.2f}"
    assert np.allclose(origin, [cx - cfg.start_distance, cy, 0.0])


def test_walls_match_the_maze_corridor():
    """1.0 m walls with their inner faces 0.925 m from the centreline (1.85 m free width), also over the trench."""
    meshes, _ = corridor_trench_terrain(0.0, CorridorTrenchTerrainCfg(size=SIZE))
    cx, cy = 0.5 * SIZE[0], 0.5 * SIZE[1]
    for x in (1.0, cx, 7.0):
        z = _top_z(meshes, [(x, cy + 0.92), (x, cy + 0.93), (x, cy - 0.93), (x, cy + 1.07)])
        assert z[0] < 0.01, f"free corridor narrower than 1.85 m at x = {x}"
        assert np.allclose(z[1:], 1.0, atol=1e-6), f"wall heights {z[1:]} at x = {x}"
