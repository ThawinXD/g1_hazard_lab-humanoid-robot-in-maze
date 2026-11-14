"""Trench / ramp test terrain: a straight walled corridor with the hazard maze's dimensions and one hazard across it.

Each tile is one corridor along +x: 1.85 m between 1.0 m tall, 0.15 m thick walls (``hazard_maze.py``), one hazard
across the middle of the tile, and the spawn point ``start_distance`` before the hazard's centre line:

* trench: full width, ``pit_depth`` deep (``corridor_trench_terrain``)
* ramp: the maze's trapezoid hump, up ``ramp_length``, a flat top, down again (``corridor_ramp_terrain``)
* ice: one maze cell of ice (``corridor_ice_terrain``); the tile is a separate low-friction mesh added by the hazard
  maze's terrain generator / importer, as in the challenge
* rough floor: a corridor of maze rough cells, straight or an L (``rough_corridor_terrain``)
* open floor: flat floor without walls, for sideways stepping (``open_floor_terrain``)

The trench width / ramp height comes from the tile's row: with a curriculum generator, row ``r`` of ``len(values)``
rows gets ``values[r]``, so every robot's hazard is known from its terrain level.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

TRENCH_WIDTHS = (0.16, 0.18, 0.20, 0.22, 0.24)  # One row per width [m]; the maze uses 0.15-0.22 m
RAMP_HEIGHTS = (0.24, 0.26, 0.28, 0.29, 0.30)  # One row per height [m]; the maze's steep ramps are 0.24-0.30 m
RAMP_LENGTH = 0.75  # Length of each slope [m] (hazard_maze.py ramp_length)
RAMP_HALF = 1.0  # The ramp's footprint along the corridor is one 2 m maze cell: slope feet at +-1.0 m


def trench_width(difficulty: float, widths: tuple[float, ...]) -> float:
    """Trench width for a curriculum difficulty: row r of len(widths) rows has difficulty in [r, r+1) / rows."""
    return float(widths[min(int(difficulty * len(widths)), len(widths) - 1)])


def ramp_floor_height(u, height, ramp_length: float = RAMP_LENGTH, half: float = RAMP_HALF):
    """Floor height at ``u`` [m] from the ramp's centre line (floats or tensors); the slopes start 1 cm below the floor
    at the feet, as in ``hazard_maze.py``."""
    rise = (half - abs(u)) / ramp_length
    if isinstance(rise, float):
        return max(-0.01 + (height + 0.01) * min(max(rise, 0.0), 1.0), 0.0)
    return (-0.01 + (height + 0.01) * rise.clamp(0.0, 1.0)).clamp(min=0.0)


def _box(cx: float, cy: float, cz: float, sx: float, sy: float, sz: float) -> trimesh.Trimesh:
    return trimesh.creation.box((sx, sy, sz), trimesh.transformations.translation_matrix((cx, cy, cz)))


def corridor_trench_terrain(
    difficulty: float, cfg: CorridorTrenchTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Walled corridor along x with a trench across it at the tile centre; spawn ``start_distance`` before it."""
    sx, sy = cfg.size
    cx, cy = 0.5 * sx, 0.5 * sy
    w = trench_width(difficulty, cfg.widths)
    depth, hw, t, h = cfg.pit_depth, cfg.corridor_half_width, cfg.wall_thickness, cfg.wall_height
    part = 0.5 * (sx - w)  # Floor length on each side of the trench
    meshes = [
        _box(cx, cy, -depth - 0.5, sx, sy, 1.0),  # Bottom of the trench (top at z = -depth)
        _box(0.5 * part, cy, -0.5 * depth, part, sy, depth),  # Floor before the trench (top at z = 0)
        _box(sx - 0.5 * part, cy, -0.5 * depth, part, sy, depth),  # Floor after the trench
    ]
    for side in (-1.0, 1.0):  # Walls along the whole tile, bridging the trench like the maze's cell walls
        meshes.append(_box(cx, cy + side * (hw + 0.5 * t), 0.5 * h, sx, t, h))
    return meshes, np.array([cx - cfg.start_distance, cy, 0.0])


def corridor_ramp_terrain(difficulty: float, cfg: CorridorRampTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Walled corridor along x with the maze's ramp across it at the tile centre; spawn ``start_distance`` before it."""
    sx, sy = cfg.size
    cx, cy = 0.5 * sx, 0.5 * sy
    rh = trench_width(difficulty, cfg.heights)
    hw, t, h, L, half = cfg.corridor_half_width, cfg.wall_thickness, cfg.wall_height, cfg.ramp_length, cfg.ramp_half
    meshes = [_box(cx, cy, -0.5, sx, sy, 1.0)]  # Floor, top at z = 0
    prof = [(-half, -0.01), (half, -0.01), (half - L, rh), (-half + L, rh)]  # (along, z), as in hazard_maze.py
    pts = [(cx + u, cy + v, z) for u, z in prof for v in (-hw, hw)]
    meshes.append(trimesh.convex.convex_hull(np.array(pts)))
    for side in (-1.0, 1.0):
        meshes.append(_box(cx, cy + side * (hw + 0.5 * t), 0.5 * h, sx, t, h))
    return meshes, np.array([cx - cfg.start_distance, cy, 0.0])


def corridor_ice_terrain(difficulty: float, cfg: CorridorIceTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Walled corridor along x with one maze cell of ice at the tile centre; spawn ``start_distance`` before it."""
    from g1_maze_lab.terrains.hazard_maze import _PENDING_ICE  # Collected by HazardTerrainGenerator for the importer

    sx, sy = cfg.size
    cx, cy = 0.5 * sx, 0.5 * sy
    hw, t, h = cfg.corridor_half_width, cfg.wall_thickness, cfg.wall_height
    meshes = [_box(cx, cy, -0.5, sx, sy, 1.0)]  # Floor, top at z = 0
    for side in (-1.0, 1.0):
        meshes.append(_box(cx, cy + side * (hw + 0.5 * t), 0.5 * h, sx, t, h))
    _PENDING_ICE.clear()
    _PENDING_ICE.append((cfg.start_distance, 0.0, cfg.ice_size, cfg.ice_size))  # Relative to the returned origin
    return meshes, np.array([cx - cfg.start_distance, cy, 0.0])


@configclass
class CorridorIceTerrainCfg(SubTerrainBaseCfg):
    """One walled corridor with one maze cell of ice (see :func:`corridor_ice_terrain`)."""

    function = corridor_ice_terrain
    ice_size: float = 1.85  # Side of the square ice tile [m]: cell 2.0 - wall 0.15, as in hazard_maze.py
    corridor_half_width: float = 0.925  # Half the free corridor width [m]
    wall_thickness: float = 0.15  # [m]
    wall_height: float = 1.0  # [m]
    start_distance: float = 2.0  # Spawn point before the ice cell's centre [m]


ICE_CORNER_NEXT = ("plain", "rough", "ramp", "spikes")  # One row per kind of cell after the ice corner


def corridor_ice_corner_terrain(
    difficulty: float, cfg: CorridorIceCornerTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """L-shaped walled corridor: east to an ice corner cell at (``corner_x``, ``corner_y``), then north. The cell after
    the corner holds the row's ``next_kinds`` entry, built like hazard_maze.py: plain floor, 10 loose 2-8 cm boxes, a
    ``ramp_height`` ramp, or a spike bed on the east side (wall to the centreline). Spawn ``start_distance`` west of
    the corner cell's centre."""
    from g1_maze_lab.terrains.hazard_maze import _PENDING_ICE, _PENDING_SPIKES

    sx, sy = cfg.size
    cx, cy, c = cfg.corner_x, cfg.corner_y, cfg.cell_size
    hw, t, h = cfg.corridor_half_width, cfg.wall_thickness, cfg.wall_height
    origin = np.array([cx - cfg.start_distance, cy, 0.0])
    meshes = [_box(0.5 * sx, 0.5 * sy, -0.5, sx, sy, 1.0)]  # Floor, top at z = 0
    meshes.append(_box(0.5 * (cx + hw + t), cy - hw - 0.5 * t, 0.5 * h, cx + hw + t, t, h))  # South wall
    meshes.append(_box(0.5 * (cx - hw), cy + hw + 0.5 * t, 0.5 * h, cx - hw, t, h))  # North wall up to the corner
    y0 = cy - hw - t
    meshes.append(_box(cx + hw + 0.5 * t, 0.5 * (y0 + sy), 0.5 * h, t, sy - y0, h))  # East wall (outside of the turn)
    meshes.append(_box(cx - hw - 0.5 * t, 0.5 * (cy + hw + sy), 0.5 * h, t, sy - cy - hw, h))  # West wall after it

    _PENDING_ICE.clear()
    _PENDING_ICE.append((cfg.start_distance, 0.0, 2.0 * hw, 2.0 * hw))  # The corner cell, relative to the origin
    _PENDING_SPIKES.clear()
    kind = cfg.next_kinds[min(int(difficulty * len(cfg.next_kinds)), len(cfg.next_kinds) - 1)]
    nx, ny = cx, cy + c  # Centre of the cell after the corner
    rng = np.random.default_rng()
    if kind == "rough":
        inner = c - t - 0.3
        for _ in range(10):
            bx, by = rng.uniform(0.15, 0.4, size=2)
            bh = rng.uniform(0.02, 0.08)
            px, py = rng.uniform(-0.5 * inner, 0.5 * inner, size=2)
            meshes.append(_box(nx + px, ny + py, 0.5 * bh, bx, by, bh))
    elif kind == "ramp":
        L, half, rh = cfg.ramp_length, 0.5 * c, cfg.ramp_height
        prof = [(-half, -0.01), (half, -0.01), (half - L, rh), (-half + L, rh)]  # (along y, z)
        meshes.append(trimesh.convex.convex_hull(np.array([(nx + v, ny + u, z) for u, z in prof for v in (-hw, hw)])))
    elif kind == "spikes":
        pitch, rad = 0.15, 0.45 * 0.15
        for px in np.arange(nx + rad, nx + hw - rad + 1e-6, pitch):
            for py in np.arange(ny - 0.7 + rad, ny + 0.7 - rad + 1e-6, pitch):
                cone = trimesh.creation.cone(radius=rad, height=rng.uniform(0.25, 0.35), sections=8)
                cone.apply_translation((px - origin[0], py - origin[1], 0.0))  # Relative to the origin
                _PENDING_SPIKES.append(cone)
    return meshes, origin


@configclass
class CorridorIceCornerTerrainCfg(SubTerrainBaseCfg):
    """L-shaped corridor with an ice corner cell (see :func:`corridor_ice_corner_terrain`)."""

    function = corridor_ice_corner_terrain
    next_kinds: tuple[str, ...] = ICE_CORNER_NEXT  # Kind of cell after the corner per terrain row
    corner_x: float = 5.0  # Corner cell centre in the tile [m]
    corner_y: float = 3.0
    cell_size: float = 2.0  # [m]
    ramp_height: float = 0.28  # Steep ramp after the corner [m]
    ramp_length: float = RAMP_LENGTH  # [m]
    corridor_half_width: float = 0.925  # [m]
    wall_thickness: float = 0.15  # [m]
    wall_height: float = 1.0  # [m]
    start_distance: float = 3.0  # Spawn point west of the corner cell's centre [m]


def open_floor_terrain(difficulty: float, cfg: OpenFloorTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Flat floor without walls (sideways stepping and drift test); spawn at the tile centre."""
    sx, sy = cfg.size
    return [_box(0.5 * sx, 0.5 * sy, -0.5, sx, sy, 1.0)], np.array([0.5 * sx, 0.5 * sy, 0.0])


@configclass
class OpenFloorTerrainCfg(SubTerrainBaseCfg):
    """Flat floor without walls (see :func:`open_floor_terrain`)."""

    function = open_floor_terrain


ROUGH_KINDS = ("straight", "lineup", "left", "right")  # One row per kind of rough-floor corridor
ROUGH_LINEUP = (2.0, 0.48)  # lineup: 2.0 m after the spawn the lane moves 0.48 m sideways (a spike-lane line-up)


def rough_path(kind: str, cfg: RoughCorridorTerrainCfg) -> list[tuple[float, float]]:
    """Centreline of a rough corridor as waypoints relative to the spawn point (the returned origin)."""
    if kind in ("straight", "lineup"):
        return [(0.0, 0.0), (cfg.size[0] - cfg.spawn_x, 0.0)]
    leg = cfg.corner_x - cfg.spawn_x  # East leg up to the corner cell's centre
    out = cfg.size[1] - cfg.corridor_y  # Leg after the corner, to the tile edge
    return [(0.0, 0.0), (leg, 0.0), (leg, out if kind == "left" else -out)]


def rough_corridor_terrain(difficulty: float, cfg: RoughCorridorTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Walled corridor of maze rough cells (10 loose 2-8 cm boxes per 2 m cell, as hazard_maze.py), straight east or
    an L turning left / right at (``corner_x``, ``corridor_y``); the row's ``kinds`` entry picks it (``lineup`` is
    the straight corridor, the script moves the lane). A right turn is the left one mirrored in y. Spawn at
    (``spawn_x``, ``corridor_y``), boxes kept ``spawn_clear`` away from it."""
    sx, sy = cfg.size
    kind = cfg.kinds[min(int(difficulty * len(cfg.kinds)), len(cfg.kinds) - 1)]
    c, hw, t, h = cfg.cell_size, cfg.corridor_half_width, cfg.wall_thickness, cfg.wall_height
    cx, cy = cfg.corner_x, cfg.corridor_y
    boxes = [(0.5 * sx, 0.5 * sy, -0.5, sx, sy, 1.0)]  # Floor, top at z = 0: (centre x, y, z, size x, y, z)
    if kind in ("straight", "lineup"):
        for side in (-1.0, 1.0):
            boxes.append((0.5 * sx, cy + side * (hw + 0.5 * t), 0.5 * h, sx, t, h))
        cells = [((i + 0.5) * c, cy) for i in range(int(sx // c))]
    else:  # L turning left (north) at the corner, as G1-Ice-Corner-Test
        boxes.append((0.5 * (cx + hw + t), cy - hw - 0.5 * t, 0.5 * h, cx + hw + t, t, h))  # South wall
        boxes.append((0.5 * (cx - hw), cy + hw + 0.5 * t, 0.5 * h, cx - hw, t, h))  # North wall up to the corner
        y0 = cy - hw - t
        boxes.append((cx + hw + 0.5 * t, 0.5 * (y0 + sy), 0.5 * h, t, sy - y0, h))  # East wall (outside of the turn)
        boxes.append((cx - hw - 0.5 * t, 0.5 * (cy + hw + sy), 0.5 * h, t, sy - cy - hw, h))  # West wall after it
        cells = [((i + 0.5) * c, cy) for i in range(int(cx // c) + 1)]
        cells += [(cx, cy + k * c) for k in range(1, int((sy - cy) // c) + 1)]
        bare = [{"before": (cx - c, cy), "after": (cx, cy + c)}[b] for b in cfg.bare_cells]
        cells = [p for p in cells if all(math.dist(p, q) > 0.5 * c for q in bare)]
    rng = np.random.default_rng(int(round(difficulty * 1e6)))  # Same boxes for the same generator seed
    inner = c - t - 0.3  # Box centres within +-inner/2 of the cell centre (hazard_maze.py)
    for px0, py0 in cells:
        for _ in range(cfg.boxes_per_cell):
            bx, by = rng.uniform(0.15, 0.4, size=2)
            bh = rng.uniform(*cfg.box_height_range)
            px, py = px0 + rng.uniform(-0.5 * inner, 0.5 * inner), py0 + rng.uniform(-0.5 * inner, 0.5 * inner)
            if math.hypot(px - cfg.spawn_x, py - cy) < cfg.spawn_clear:
                continue
            boxes.append((px, py, 0.5 * bh, bx, by, bh))
    if kind == "right":
        boxes = [(x, sy - y, z, a, b, d) for x, y, z, a, b, d in boxes]
        cy = sy - cy
    return [_box(*b) for b in boxes], np.array([cfg.spawn_x, cy, 0.0])


@configclass
class RoughCorridorTerrainCfg(SubTerrainBaseCfg):
    """Walled corridor of maze rough cells (see :func:`rough_corridor_terrain`)."""

    function = rough_corridor_terrain
    kinds: tuple[str, ...] = ROUGH_KINDS  # Corridor kind per terrain row
    spawn_x: float = 1.0  # Spawn point (middle of the first cell) [m]
    corridor_y: float = 3.0  # Corridor centreline (before mirroring) [m]
    corner_x: float = 5.0  # Corner cell centre [m]
    cell_size: float = 2.0  # [m]
    boxes_per_cell: int = 10  # hazard_maze.py rough_boxes_per_cell
    box_height_range: tuple[float, float] = (0.02, 0.08)  # hazard_maze.py rough_height_range [m]
    spawn_clear: float = 0.5  # No box centre this close to the spawn point [m]
    bare_cells: tuple[str, ...] = ()  # L cells left flat, "before" / "after" the corner (G1-Corner-Test's spike cells)
    corridor_half_width: float = 0.925  # [m]
    wall_thickness: float = 0.15  # [m]
    wall_height: float = 1.0  # [m]


@configclass
class CorridorRampTerrainCfg(SubTerrainBaseCfg):
    """One walled corridor with the maze's ramp (see :func:`corridor_ramp_terrain`)."""

    function = corridor_ramp_terrain
    heights: tuple[float, ...] = RAMP_HEIGHTS  # Ramp height per terrain row [m]
    ramp_length: float = RAMP_LENGTH  # Length of each slope [m]
    ramp_half: float = RAMP_HALF  # Slope feet at +-ramp_half from the centre line [m]
    corridor_half_width: float = 0.925  # Half the free corridor width [m]: 0.5 * (cell 2.0 - wall 0.15)
    wall_thickness: float = 0.15  # [m]
    wall_height: float = 1.0  # [m]
    start_distance: float = 2.0  # Spawn point before the ramp's centre line [m]


@configclass
class CorridorTrenchTerrainCfg(SubTerrainBaseCfg):
    """One walled corridor with a trench (see :func:`corridor_trench_terrain`)."""

    function = corridor_trench_terrain
    widths: tuple[float, ...] = TRENCH_WIDTHS  # Trench width per terrain row [m]
    pit_depth: float = 1.0  # Trench depth [m] (hazard_maze.py)
    corridor_half_width: float = 0.925  # Half the free corridor width [m]: 0.5 * (cell 2.0 - wall 0.15)
    wall_thickness: float = 0.15  # [m]
    wall_height: float = 1.0  # [m]
    start_distance: float = 2.0  # Spawn point before the trench line [m]
