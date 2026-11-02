"""Trench test terrain: a straight walled corridor with the hazard maze's dimensions and one trench across it.

Each tile is one corridor along +x: 1.85 m between 1.0 m tall, 0.15 m thick walls (``hazard_maze.py``), a full-width
trench ``pit_depth`` deep across the middle of the tile, and the spawn point ``start_distance`` before the trench line.
The trench width comes from the tile's row: with a curriculum generator, row ``r`` of ``len(widths)`` rows gets
``widths[r]``, so every robot's width is known from its terrain level.
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

TRENCH_WIDTHS = (0.16, 0.18, 0.20, 0.22, 0.24)  # One row per width [m]; the maze uses 0.15-0.22 m


def trench_width(difficulty: float, widths: tuple[float, ...]) -> float:
    """Trench width for a curriculum difficulty: row r of len(widths) rows has difficulty in [r, r+1) / rows."""
    return float(widths[min(int(difficulty * len(widths)), len(widths) - 1)])


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
