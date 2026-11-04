"""Training terrain for ``G1-Hazard-Challenge``: ``G1-Hazard-Pro``'s four tiles, with real ice and harder hazards.

* ``ice_boxes``: the box tile's 1 cm plates (every other 2 m cell) become maze ice: a separate mesh with friction 0.2
  and ``mjc:priority``, handed to ``HazardTerrainGenerator`` / ``HazardTerrainImporter`` like the maze's ice. The
  plates of ``G1-Hazard-Pro`` had the floor's friction, and since Newton's MuJoCo solver takes the larger friction of
  two touching shapes, its foot-friction randomization (0.2-1.0) never made anything slippery: the policy never stood
  on ice in training. The ice cells have no boxes, as in the maze.
* ``trenches`` 0.10-0.26 m (was 0.06-0.24) and ``ramps`` 0.12-0.32 m (was 0.08-0.30): at the curriculum's average
  row (about 6.4) a robot then meets about 0.21 m trenches and 0.26 m ramps instead of 0.18 / 0.23 m. The maze has
  0.15-0.22 m trenches and 0.24-0.30 m steep ramps on the short route.
"""

from __future__ import annotations

import numpy as np
import trimesh

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

from ...terrains import HazardTerrainGeneratorCfg
from ..hazard_starter.env_cfg import HazardTrapezoidRampTerrainCfg, HazardTrenchTerrainCfg, _box_mesh


def hazard_ice_boxes_terrain(
    difficulty: float, cfg: HazardIceBoxesTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Loose maze boxes on half of the 2 m cells and maze ice on the other half (checkerboard), spawn at the centre."""
    from g1_maze_lab.terrains.hazard_maze import _PENDING_ICE  # Collected by HazardTerrainGenerator for the importer

    sx, sy = cfg.size  # Tile size (8 m x 8 m)
    cx, cy = 0.5 * sx, 0.5 * sy  # Tile centre = spawn point
    c = cfg.cell_size  # Cell size (2 m)
    rng = np.random.default_rng(int(round(difficulty * 1e6)) + 2026)  # Deterministic per difficulty
    max_h = cfg.height_range[0] + float(difficulty) * (cfg.height_range[1] - cfg.height_range[0])
    max_h = max(max_h, cfg.height_range[0] + 0.01)  # Valid sampling range at difficulty 0

    meshes = [_box_mesh(cx, cy, -0.5, sx, sy, 1.0)]  # Floor, top at z = 0
    _PENDING_ICE.clear()
    for i in range(int(sx // c)):
        for j in range(int(sy // c)):
            cell_cx, cell_cy = (i + 0.5) * c, (j + 0.5) * c
            if (i + j) % 2 == 1:  # Ice cell: one maze ice tile, relative to the returned origin
                _PENDING_ICE.append((cell_cx - cx, cell_cy - cy, cfg.ice_size, cfg.ice_size))
                continue
            for _ in range(cfg.boxes_per_cell):  # Rough cell: loose boxes, as G1-Hazard-Pro's box tile
                bx, by = rng.uniform(0.15, 0.40, size=2)
                bh = float(rng.uniform(cfg.height_range[0], max_h))
                wx = cell_cx + float(rng.uniform(-0.42 * c, 0.42 * c))
                wy = cell_cy + float(rng.uniform(-0.42 * c, 0.42 * c))
                if abs(wx - cx) < 0.65 and abs(wy - cy) < 0.65:  # Keep the spawn area clear
                    continue
                meshes.append(_box_mesh(wx, wy, 0.5 * bh, float(bx), float(by), bh))
    return meshes, np.array([cx, cy, 0.0])


@configclass
class HazardIceBoxesTerrainCfg(SubTerrainBaseCfg):
    """Checkerboard of maze ice cells and loose-box cells (see :func:`hazard_ice_boxes_terrain`)."""

    function = hazard_ice_boxes_terrain
    height_range: tuple[float, float] = (0.02, 0.09)  # Box height range [m] from difficulty 0 to 1
    boxes_per_cell: int = 10  # Boxes per rough cell (hazard_maze.py)
    cell_size: float = 2.0  # [m]
    ice_size: float = 1.85  # Side of an ice tile [m]: cell 2.0 - wall 0.15, as in hazard_maze.py


# Same grid and proportions as G1-Hazard-Pro (hazard_starter/env_cfg.py OPTION_B_TERRAIN), with the changes above
CHALLENGE_TERRAIN = HazardTerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=10,
    num_cols=12,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "trenches": HazardTrenchTerrainCfg(proportion=0.25, gap_width_range=(0.10, 0.26)),
        "ramps": HazardTrapezoidRampTerrainCfg(proportion=0.30, ramp_height_range=(0.12, 0.32)),
        "ice_boxes": HazardIceBoxesTerrainCfg(proportion=0.25, height_range=(0.02, 0.09), boxes_per_cell=10),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20, noise_range=(0.01, 0.05), noise_step=0.01, border_width=0.25
        ),
    },
)
