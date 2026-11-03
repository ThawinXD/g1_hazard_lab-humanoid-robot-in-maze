"""Unit tests for the trench / ramp test terrain (scripts/trench_test.py), no simulator required."""

from __future__ import annotations

import numpy as np
import trimesh

from g1_maze_lab.tasks.trench_test.terrain import (
    ICE_CORNER_NEXT,
    RAMP_HEIGHTS,
    TRENCH_WIDTHS,
    CorridorIceCornerTerrainCfg,
    CorridorIceTerrainCfg,
    CorridorRampTerrainCfg,
    CorridorTrenchTerrainCfg,
    corridor_ice_corner_terrain,
    corridor_ice_terrain,
    corridor_ramp_terrain,
    corridor_trench_terrain,
    ramp_floor_height,
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


def test_ramp_matches_the_maze_profile():
    """The maze's ramp across the corridor: flat floor up to 1.0 m before the centre line, a 0.75 m slope, a flat top
    for |u| <= 0.25 at the row's height, walls on both sides; ramp_floor_height gives the same profile."""
    cfg = CorridorRampTerrainCfg(size=SIZE)
    h = RAMP_HEIGHTS[3]
    meshes, origin = corridor_ramp_terrain(3.5 / len(RAMP_HEIGHTS), cfg)
    cx, cy = 0.5 * SIZE[0], 0.5 * SIZE[1]
    us = (-1.5, -0.8, -0.55, -0.25, 0.0, 0.2, 0.6, 1.5)
    expected = [ramp_floor_height(u, h) for u in us]
    assert expected[0] == 0.0 and expected[-1] == 0.0 and np.allclose(expected[3:6], h), f"profile {expected}"
    for y in (cy - 0.85, cy, cy + 0.85):
        z = _top_z(meshes, [(cx + u, y) for u in us])
        assert np.allclose(np.maximum(z, 0.0), expected, atol=2e-3), f"surface heights {z} at y = {y:.2f}"
    z = _top_z(meshes, [(cx, cy + 0.93), (cx, cy - 0.93)])
    assert np.allclose(z, 1.0, atol=1e-6), f"wall heights {z} over the ramp"
    assert np.allclose(origin, [cx - cfg.start_distance, cy, 0.0])


def test_ice_corridor_hands_one_maze_cell_of_ice_to_the_generator():
    """Flat floor and walls like the other corridors, plus one 1.85 m ice tile at the tile centre, given to the hazard
    maze's generator relative to the spawn point (the importer adds it as the low-friction ice mesh)."""
    from g1_maze_lab.terrains.hazard_maze import _PENDING_ICE

    cfg = CorridorIceTerrainCfg(size=SIZE)
    _PENDING_ICE.append((9.0, 9.0, 1.0, 1.0))  # Left over from another terrain: must be replaced
    meshes, origin = corridor_ice_terrain(0.0, cfg)
    cx, cy = 0.5 * SIZE[0], 0.5 * SIZE[1]
    assert [(cfg.start_distance, 0.0, 1.85, 1.85)] == _PENDING_ICE, f"ice tiles {_PENDING_ICE}"
    assert np.allclose(origin + [_PENDING_ICE[0][0], 0.0, 0.0], [cx, cy, 0.0]), "ice tile not at the tile centre"
    z = _top_z(meshes, [(cx, cy), (cx - 1.5, cy), (cx, cy + 0.93), (cx, cy - 0.93)])
    assert np.allclose(z, [0.0, 0.0, 1.0, 1.0], atol=1e-6), f"surface heights {z}"
    _PENDING_ICE.clear()


def test_ice_corner_is_an_l_turning_left_at_an_ice_cell():
    """Entry corridor along +x into an ice cell at the corner, exit corridor north; the cell after the corner holds the
    row's kind (a 0.28 m ramp / a spike bed on the east side / nothing); walls close the outside of the turn."""
    from g1_maze_lab.terrains.hazard_maze import _PENDING_ICE, _PENDING_SPIKES

    size = (8.0, 8.0)
    cfg = CorridorIceCornerTerrainCfg(size=size)
    cx, cy = cfg.corner_x, cfg.corner_y
    rows = len(ICE_CORNER_NEXT)
    for r, kind in enumerate(ICE_CORNER_NEXT):
        meshes, origin = corridor_ice_corner_terrain((r + 0.5) / rows, cfg)
        assert np.allclose(origin, [cx - cfg.start_distance, cy, 0.0])
        assert [(cfg.start_distance, 0.0, 1.85, 1.85)] == _PENDING_ICE, f"{kind}: ice tiles {_PENDING_ICE}"
        z = _top_z(
            meshes, [(cx - 2.5, cy), (cx, cy), (cx, cy + 3.5), (cx + 1.0, cy), (cx, cy - 1.0), (cx - 1.0, cy + 1.0)]
        )
        assert np.allclose(z[:3], 0.0, atol=1e-6) or kind == "rough", f"{kind}: floor heights {z[:3]}"
        assert np.allclose(z[3:], 1.0, atol=1e-6), f"{kind}: wall heights {z[3:]} (east, south, inner corner)"
        z_next = _top_z(meshes, [(cx, cy + 2.0)])[0]
        assert np.isclose(z_next, cfg.ramp_height) if kind == "ramp" else z_next < 0.09, f"{kind}: next cell {z_next}"
        if kind == "spikes":
            tips = np.array([c.bounds[1] for c in _PENDING_SPIKES]) + [origin[0], origin[1], 0.0]
            assert len(tips) > 40 and tips[:, 0].min() > cx and np.all(np.abs(tips[:, 1] - cy - 2.0) < 0.7)
        else:
            assert not _PENDING_SPIKES, f"{kind}: unexpected spike cones"
    _PENDING_ICE.clear()
    _PENDING_SPIKES.clear()
