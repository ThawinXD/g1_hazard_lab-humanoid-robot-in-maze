"""Unit tests for the hazard test terrains (scripts/trench_test.py, ice_turn_test.py, rough_test.py,
strafe_test.py), no simulator required."""

from __future__ import annotations

import numpy as np
import trimesh

from g1_maze_lab.tasks.trench_test.terrain import (
    ICE_CORNER_NEXT,
    RAMP_HEIGHTS,
    ROUGH_KINDS,
    TRENCH_WIDTHS,
    CorridorIceCornerTerrainCfg,
    CorridorIceTerrainCfg,
    CorridorRampTerrainCfg,
    CorridorTrenchTerrainCfg,
    OpenFloorTerrainCfg,
    RoughCorridorTerrainCfg,
    corridor_ice_corner_terrain,
    corridor_ice_terrain,
    corridor_ramp_terrain,
    corridor_trench_terrain,
    open_floor_terrain,
    ramp_floor_height,
    rough_corridor_terrain,
    rough_path,
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


def _rough(kind):
    cfg = RoughCorridorTerrainCfg(size=(8.0, 8.0))
    rows = len(ROUGH_KINDS)
    meshes, origin = rough_corridor_terrain((ROUGH_KINDS.index(kind) + 0.5) / rows, cfg)
    return cfg, meshes, origin


def test_rough_corridor_lanes_are_free_and_walled():
    """Every kind's lane (rough_path) runs between 1.0 m walls through 2-8 cm boxes: away from the corner the walls
    stand 0.93 m to both sides of the lane, the floor along it is at most a box high, and the spawn point is clear."""
    for kind in ROUGH_KINDS:
        cfg, meshes, origin = _rough(kind)
        path = np.array(rough_path(kind, cfg)) + origin[:2]
        assert np.allclose(path[0], origin[:2]), f"{kind}: lane does not start at the spawn point"
        corners = path[1:-1]
        for a, b in zip(path[:-1], path[1:], strict=True):
            length = np.linalg.norm(b - a)
            d = (b - a) / length
            normal = np.array([-d[1], d[0]])
            for t in np.arange(0.3, length - 0.2, 0.5):
                p = a + t * d
                if any(np.linalg.norm(p - c) < 1.0 for c in corners):  # The corner cell is open on one side
                    continue
                z = _top_z(meshes, [p, p + 0.93 * normal, p - 0.93 * normal, p + 0.6 * normal])
                assert z[0] <= 0.08 + 1e-6 and z[3] <= 0.08 + 1e-6, f"{kind}: lane blocked at {p}: {z}"
                assert np.allclose(z[1:3], 1.0, atol=1e-6), f"{kind}: walls {z[1:3]} beside {p}"
        assert _top_z(meshes, [origin[:2]])[0] == 0.0, f"{kind}: box under the spawn point"


def test_rough_corridor_kinds_and_boxes():
    """Straight / lineup are the same straight corridor, right's walls are left's mirrored in y; every cell holds the
    maze's loose boxes (at most 10 per 2 m cell, 2-8 cm high)."""
    cfg, straight, o_s = _rough("straight")
    _, lineup, o_l = _rough("lineup")
    _, left, o_left = _rough("left")
    _, right, o_right = _rough("right")
    assert np.allclose(o_s, [cfg.spawn_x, cfg.corridor_y, 0.0]) and np.allclose(o_s, o_l)
    assert np.allclose(o_right, [cfg.spawn_x, 8.0 - cfg.corridor_y, 0.0]) and np.allclose(o_left, o_s)
    assert rough_path("left", cfg)[-1] == (4.0, 5.0) and rough_path("right", cfg)[-1] == (4.0, -5.0)
    walls = [
        [m.bounds for m in meshes if np.isclose(m.bounds[1][2], 1.0)] for meshes in (straight, lineup, left, right)
    ]
    assert len(walls[0]) == 2 and np.allclose(walls[0], walls[1]) and len(walls[2]) == 4
    mirrored = sorted(
        (round(lo[0], 6), round(8.0 - hi[1], 6), round(hi[0], 6), round(8.0 - lo[1], 6)) for lo, hi in walls[2]
    )
    assert mirrored == sorted(
        (round(lo[0], 6), round(lo[1], 6), round(hi[0], 6), round(hi[1], 6)) for lo, hi in walls[3]
    )
    for meshes, n_cells in ((straight, 4), (left, 5)):
        boxes = [m for m in meshes if m.bounds[1][2] < 0.5 and m.bounds[0][2] >= -1e-9]
        heights = np.array([m.bounds[1][2] for m in boxes])
        assert 8 * n_cells <= len(boxes) <= 10 * n_cells, f"{len(boxes)} boxes in {n_cells} cells"
        assert heights.min() >= 0.02 - 1e-9 and heights.max() <= 0.08 + 1e-9, (
            f"box heights {heights.min()}-{heights.max()}"
        )


def test_open_floor_is_flat_without_walls():
    meshes, origin = open_floor_terrain(0.0, OpenFloorTerrainCfg(size=(8.0, 8.0)))
    z = _top_z(meshes, [(x, y) for x in (0.2, 4.0, 7.8) for y in (0.2, 4.0, 7.8)])
    assert np.allclose(z, 0.0) and np.allclose(origin, [4.0, 4.0, 0.0])
