"""Route-aware pilot for Challenge 7.5 (G1 Hazard Maze): the hazard pilot of pilot.py plus a route layer.

``pilot.py`` (its ``Pilot``, here ``HazardPilot``) does the perception, the memory, the cell-by-cell planning and the
control. This file subclasses it and only adds a *route layer*. The layer uses how the maze generator builds the two
routes out of the start. Checked offline on 1,123 generated level 7 / 9 mazes, with no exception:

* With the default corners the start has exactly two exits, and both lead to the goal (one loop through the start).
* The SHORT route is at least 6 cells shorter. Its 2nd cell (else its 3rd) is a steep ramp or a spike bed, and it has
  no plain floor apart from the start's neighbours and the goal: every other cell is a hazard, ice or loose boxes.
* Steep ramps, trenches and slaloms are only on the short route; low ramps only on the long route.
* Dead ends never hold a ramp, a spike bed, a trench or a slalom.

What the layer does (``_route_update``, right after every scan):

1. Tags every known cell with the exit of the start it is reached from (BFS over known-open edges).
2. Labels each exit SHORT / LONG / NOT_SHORT / DEAD from the hazard types seen in its branch, using only clues that
   never failed. With exactly two exits, the label of one gives the other. A cell the base pilot calls plain only counts
   once most of it was seen (``PLAIN_SEEN_RAYS``): the near edge of a spike bed, box cell or trench cell looks plain.
3. Estimates the time to the goal [s] for "stay on this branch" and "go back through the start and take the other
   exit": cells x time per cell + the hazards still ahead (time + fall risk) + resets already suffered there. The long
   route also pays for dead-end detours, because its plain dead ends look like the route itself.
4. Enforces the choice with *virtual walls* that only the planner sees (``weighted_flood_fill`` and
   ``choose_direction``). The real wall map is never written. A choice that leaves no way to the goal is dropped.
5. On a SHORT branch, a plain side cell is a likely dead end: it gets extra planner cost (follow the box trail).

``PILOT_ROUTE`` = ``auto`` (default: the estimate decides) | ``short`` / ``long`` (take that route once it is known)
| ``off`` (exactly the base pilot). The base pilot's ``PILOT_MODE`` and ``PILOT_SPEED`` still apply.

Limits: one robot (``num_envs == 1``, as in challenge.py; the layer switches itself off otherwise). With three or more
start exits (random start / goal: 66-74 % of mazes) it never goes back on a guess; it only commits to a branch it knows
is SHORT. The priors (route lengths, hazard counts) are for levels 7 and 9; other levels get generic ones. If the layer
ever raises an error it switches itself off and the base pilot carries on.

Run locally:  uv run python scripts/challenge.py --level 9 --seed 1 --pilot pilot_mod.py --policy <model.pt>
Upload:       this file as the pilot (the server saves it as pilot.py) + pilot.py renamed hazard_pilot.py as a helper.
"""

from __future__ import annotations  # Enable postponed evaluation of type annotations

import importlib.util  # Load pilot.py by path when hazard_pilot.py is not next to this file
import os  # PILOT_ROUTE environment variable and file paths
import sys  # Register the base module under one name
from collections import Counter, deque  # Hazard counts and BFS queues

import torch  # Tensors shared with the base pilot

try:  # Upload: the base pilot sits next to this file as hazard_pilot.py (a helper may not be called pilot.py)
    import hazard_pilot as _base  # Base hazard-aware pilot
except ImportError:  # Local runs: this file is pilot_mod.py next to the base pilot.py
    if os.path.basename(__file__) == "pilot.py":  # Uploaded as pilot.py without its helper file
        raise ImportError("pilot_mod: upload the base pilot.py as the helper file hazard_pilot.py") from None
    _spec = importlib.util.spec_from_file_location(  # Base pilot.py in the same folder as this file
        "hazard_pilot", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pilot.py")
    )
    _base = importlib.util.module_from_spec(_spec)  # Empty module object for it
    sys.modules["hazard_pilot"] = _base  # Any later import gets the same module
    _spec.loader.exec_module(_base)  # Run pilot.py

HazardPilot = _base.Pilot  # Base pilot class
DIRS = _base.DIRS  # (di, dj) per direction: 0 = E, 1 = N, 2 = W, 3 = S
CELL_UNKNOWN = _base.CELL_UNKNOWN  # Hazard types of the base pilot's cell_type map
CELL_PLAIN = _base.CELL_PLAIN
CELL_ICE = _base.CELL_ICE
CELL_ROUGH = _base.CELL_ROUGH
CELL_LOW_RAMP = _base.CELL_LOW_RAMP
CELL_SPIKE_LANE = _base.CELL_SPIKE_LANE
CELL_SPIKE_SLALOM = _base.CELL_SPIKE_SLALOM
CELL_TRENCH = _base.CELL_TRENCH
CELL_STEEP_RAMP = _base.CELL_STEEP_RAMP

DIR_NAMES = ("E", "N", "W", "S")  # Direction names for the log
TYPE_NAMES = {  # Hazard type names for the log
    CELL_UNKNOWN: "unknown",
    CELL_PLAIN: "plain floor",
    CELL_ICE: "ice",
    CELL_ROUGH: "boxes",
    CELL_LOW_RAMP: "low ramp",
    CELL_SPIKE_LANE: "spike bed",
    CELL_SPIKE_SLALOM: "slalom",
    CELL_TRENCH: "trench",
    CELL_STEEP_RAMP: "steep ramp",
}

# Route labels of an exit of the start
UNKNOWN = "UNKNOWN"  # Not enough seen yet
SHORT = "SHORT"  # The short route (hard hazards, boxes everywhere)
LONG = "LONG"  # The long route (a low ramp was seen)
NOT_SHORT = "NOT_SHORT"  # Plain floor where the short route never has any: the long route or a dead end
DEAD = "DEAD"  # Branch walled in all round without the goal
LONGISH = (LONG, NOT_SHORT)  # Labels that rule out the short route
MULTI = -1  # Branch tag of a cell reached from two exits (the branches are known to meet)
BIG = 10**6  # "No path" distance [cells]

# Generator facts per level (tasks/maze/env_cfg.py hazard levels + hazard_maze.py defaults); other levels use level 9's
ROUTE_HAZARD_COUNTS = {
    7: {
        "short": {CELL_TRENCH: 1, CELL_STEEP_RAMP: 2, CELL_SPIKE_LANE: 2, CELL_ICE: 1},  # Rest of the route: boxes
        "long": {CELL_LOW_RAMP: 2, CELL_SPIKE_LANE: 1, CELL_ROUGH: 3, CELL_ICE: 2},  # Rest of the long route: plain
    },
    9: {
        "short": {CELL_TRENCH: 2, CELL_STEEP_RAMP: 2, CELL_SPIKE_LANE: 2, CELL_SPIKE_SLALOM: 1, CELL_ICE: 1},
        "long": {CELL_LOW_RAMP: 3, CELL_SPIKE_LANE: 2, CELL_ROUGH: 4, CELL_ICE: 3},
    },
}
# Route priors from replaying the generator offline (150-300 mazes per level and start/goal setting)
SHORT_CELLS_PRIOR = {7: 14, 9: 24}  # Mean short-route length [cells], default corners
SHORT_PER_MANHATTAN = {7: 1.8, 9: 2.2}  # Random start / goal: mean short-route length / Manhattan distance
GAP_PRIOR = {7: 10, 9: 11}  # Mean "long minus short" [cells], default corners (medians 8 / 10, minimum 6)
GAP_PRIOR_RANDOM = {7: 8, 9: 10}  # Same with random start / goal
GAP_PRIOR_OTHER = 6  # Other levels: the generator's minimum
EXTRA_TURNS_PRIOR = 9  # Mean "long minus short" number of turns (8-10 on both levels)

# Time model [s]. Base pilot speed caps / SPEED_SHARE (the policy reaches ~83 % of the commanded speed, handoff.md)
CELL_TIME = {  # Seconds to walk one 2 m cell of each type
    CELL_UNKNOWN: 2.4,  # Priced like plain floor
    CELL_PLAIN: 2.4,  # 1.0 m/s cruise
    CELL_ICE: 4.4,  # 0.55 m/s cap
    CELL_ROUGH: 3.5,  # 0.68 m/s cap on loose boxes
    CELL_LOW_RAMP: 3.5,  # 0.68 m/s cap
    CELL_SPIKE_LANE: 7.0,  # 0.48 m/s in the safe lane + about 2 s lining up
    CELL_SPIKE_SLALOM: 14.0,  # Stop, strafe about 0.9 m at about 0.18 m/s, walk out
    CELL_TRENCH: 2.7,  # 0.9 m/s crossing
    CELL_STEEP_RAMP: 2.7,  # 0.9 m/s climb after the run-up
}
ROUTE_CELL_TIME = {"short": CELL_TIME[CELL_ROUGH], "long": CELL_TIME[CELL_PLAIN]}  # An ordinary cell of each route
LONG_DETOUR_S_PER_CELL = 0.45  # Long route's dead-end detours: level 9 long routes took ~135 s, the model said 119 s
TURN_S = 0.7  # Extra time per corner, walked as a curve at reduced speed [s]
FAIL_CHANCE = {  # Chance that one crossing ends in a reset, for model_8497 (handoff.md); re-measure for a new policy
    CELL_TRENCH: 0.04,  # scripts/trench_test.py: 0.20 m fails 1 %, 0.22 m 10 % at 0.9 m/s
    CELL_STEEP_RAMP: 0.01,  # scripts/trench_test.py --task G1-Ramp-Test: 99.5-100 % crossed at 0.9 m/s
    CELL_SPIKE_LANE: 0.02,  # Rare spike hits in the benchmark
    CELL_SPIKE_SLALOM: 0.10,  # Lining up and strafing between two beds
    CELL_ICE: 0.02,  # Turns on ice (scripts/ice_turn_test.py)
    CELL_ROUGH: 0.01,  # Feet snagging on loose boxes
    CELL_LOW_RAMP: 0.0,  # Never failed
}
FALL_LOSS_S = 40.0  # Average time a reset costs: walking back to where it happened [s]
DEATH_S = 60.0  # Extra estimate per reset already suffered on a branch [s]
PIVOT_S = 4.0  # Brake and turn back on the spot [s]
SWITCH_MARGIN_S = 2.0  # Go back only if that is estimated at least this much faster [s]
MAX_SWITCH_CELLS = 4  # Never go back from deeper than this into a branch [cells] (pays off 46-62 % of the time at 4)
DEAD_END_HINT = 6.0  # Extra planner cost for a plain side cell of a SHORT branch (the base cost of a plain cell is 1.0)
# Scan rays that must land on a cell's inner floor (+-0.8 m, ~256 rays in full view) in one frame before "plain" counts.
# The base classifier still called real hazard shapes plain at up to 32 rays (spike bed), 96 (boxes), 128 (trench).
PLAIN_SEEN_RAYS = 140


class Pilot(HazardPilot):
    """Hazard pilot + route layer: tells short route, long route and dead ends apart; stays or goes back."""

    def __init__(
        self,
        num_envs: int,  # Number of robots (1 in the challenge)
        num_cells: int,  # Maze size N (7, 9 or 10)
        cell_size: float,  # Cell width [m] (2.0)
        device: str,  # Simulator device
        start_cell: tuple[int, int] = (0, 0),  # Start cell (i, j)
        goal_cell: tuple[int, int] | None = None,  # Goal cell (i, j), default (N-1, N-1)
    ):
        super().__init__(num_envs, num_cells, cell_size, device, start_cell, goal_cell)  # Base pilot as it is
        mode = os.environ.get("PILOT_ROUTE", "auto").lower()  # Route layer mode
        self.route_mode = mode if mode in ("auto", "short", "long", "off") else "auto"  # Unknown values mean auto
        self.route_on = self.route_mode != "off" and num_envs == 1  # The layer plans for one robot only
        n = num_cells  # Shorthand
        self.vwalls = torch.zeros((num_envs, n, n, 4), dtype=torch.bool, device=self.device)  # Virtual walls (planning)
        self.cell_rays = torch.zeros((n, n), dtype=torch.long, device=self.device)  # Most scan rays per cell in a frame
        self.route_label = {}  # Exit direction -> label, recomputed from the map (which persists across resets)
        self.route_choice = None  # Exit to take this attempt (the start's other exits are blocked)
        self.route_closed = set()  # Exits given up this attempt (their branch frontier is blocked)
        self.route_switched = False  # Already went back once this attempt
        self.fresh_attempt = True  # In the start cell with nothing decided yet this attempt
        self.dead_end_hinted = set()  # Cells that got DEAD_END_HINT
        self.go_backs = 0  # Times the layer went back (whole run, for the log)
        self._route_key = None  # Inputs of the last update (the update is skipped while they do not change)
        self._last_msg = None  # Last decision line printed
        self.corners = self.start_cell == (0, 0) and self.goal_cell == (n - 1, n - 1)  # Default start and goal
        self.level_hazards = ROUTE_HAZARD_COUNTS.get(n, ROUTE_HAZARD_COUNTS[9])  # Expected hazards per route
        manhattan = abs(self.start_cell[0] - self.goal_cell[0]) + abs(self.start_cell[1] - self.goal_cell[1])
        if self.corners:
            self.short_prior = SHORT_CELLS_PRIOR.get(n, manhattan)  # Short-route length prior [cells]
            self.gap_prior = GAP_PRIOR.get(n, GAP_PRIOR_OTHER)  # Long-minus-short prior [cells]
        else:
            self.short_prior = round(SHORT_PER_MANHATTAN.get(n, 1.0) * manhattan)
            self.gap_prior = GAP_PRIOR_RANDOM.get(n, GAP_PRIOR_OTHER)
        self.slalom_is_short = n in (7, 9)  # On the final level the long route may have a slalom too

    # ------------------------------------------------------------------
    # Hooks into the base pilot
    # ------------------------------------------------------------------
    def sense_environment(self, obs: dict, cell: torch.Tensor) -> torch.Tensor:
        """Base scan, then the route layer (labels, decision, virtual walls) before the base planner runs."""
        walls_now = super().sense_environment(obs, cell)  # Walls and hazard types of everything in view
        if self.route_on:
            try:
                self._update_seen(obs["scan_xy"])  # How much of each cell the scan has covered
                self._route_update(cell)  # Route labels, stay / go back, virtual walls
            except Exception as err:  # A bug in the layer must never stop the robot: fall back to the base pilot
                print(f"[PILOT_MOD] route layer error, switched off: {err!r}", flush=True)
                self.route_on = False  # Base pilot only from now on
                self.vwalls.zero_()  # No virtual walls left behind
        return walls_now  # Same return value as the base method

    def weighted_flood_fill(self, walls: torch.Tensor) -> torch.Tensor:
        """Base cost-to-goal map, with the route layer's virtual walls added to a copy of ``walls``."""
        if self.route_on and bool(self.vwalls.any()):
            walls = torch.where(self.vwalls, torch.ones_like(walls), walls)  # Virtual walls block like real ones
        return super().weighted_flood_fill(walls)  # Base Bellman-Ford (cached on the walls it is given)

    def choose_direction(self, cell: torch.Tensor, open_sides: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
        """Base choice among the open sides that no virtual wall blocks."""
        if self.route_on and len(cell) == 1:  # One deciding robot (num_envs == 1)
            blocked = self.vwalls[0, int(cell[0, 0].item()), int(cell[0, 1].item())]  # (4,) virtual walls of this cell
            open_sides = open_sides & ~blocked.unsqueeze(0)  # Never leave through a virtual wall
        return super().choose_direction(cell, open_sides, heading)

    def reset(self, env_ids: torch.Tensor):
        """Base reset (keeps the maps), then a fresh route decision for the next attempt."""
        super().reset(env_ids)
        self.route_choice = None  # Decide again in the start cell, now knowing where the robot fell
        self.route_closed = set()  # Every branch is open again
        self.route_switched = False  # May go back once more in this attempt
        self.fresh_attempt = True  # Nothing decided yet this attempt
        self.vwalls.zero_()  # No virtual walls until the next decision
        self._route_key = None  # Force a route update on the next scan

    def trace_planned_route(self, env_id: int = 0, start_cell: tuple[int, int] | None = None) -> list[tuple[int, int]]:
        """Base route trace for the viewer, with the virtual walls applied (the base method reads ap.walls)."""
        if not (self.route_on and bool(self.vwalls.any())):
            return super().trace_planned_route(env_id, start_cell)
        real = self.ap.walls  # Real wall map (the base method only reads it)
        self.ap.walls = torch.where(self.vwalls, torch.ones_like(real), real)  # Show the route the planner takes
        try:
            return super().trace_planned_route(env_id, start_cell)
        finally:
            self.ap.walls = real  # Always put the real map back

    def get_debug_snapshot(self, env_id: int = 0) -> dict:
        """Base snapshot plus the route layer's state."""
        snap = super().get_debug_snapshot(env_id)
        snap["route_layer"] = {
            "mode": self.route_mode if self.route_on else "off",  # Route layer mode
            "labels": {DIR_NAMES[x]: label for x, label in self.route_label.items()},  # Label per exit
            "choice": None if self.route_choice is None else DIR_NAMES[self.route_choice],  # Exit taken
            "closed": [DIR_NAMES[x] for x in sorted(self.route_closed)],  # Exits given up this attempt
            "dead_end_hints": sorted(self.dead_end_hinted),  # Cells priced as likely dead ends
        }
        return snap

    # ------------------------------------------------------------------
    # Route layer
    # ------------------------------------------------------------------
    def _update_seen(self, scan_xy: torch.Tensor):
        """Keep, per cell, the most scan rays that landed on its inner floor in one frame."""
        n = self.num_cells  # Shorthand
        idx = torch.round(scan_xy[0] / self.cell_size).long()  # (R, 2) cell of every ray
        inner = (scan_xy[0] - idx.float() * self.cell_size).abs() < 0.80  # Same inner floor as the base classifier
        keep = inner.all(dim=-1) & ((idx >= 0) & (idx < n)).all(dim=-1)  # On an inner floor, inside the maze
        counts = torch.bincount(idx[keep, 0] * n + idx[keep, 1], minlength=n * n).reshape(n, n)  # Rays per cell
        self.cell_rays = torch.maximum(self.cell_rays, counts)  # Best view so far (kept across resets, like the map)

    def _trusted_types(self) -> torch.Tensor:
        """Hazard type per cell (n, n), with a plain cell that was not seen well yet as unknown."""
        types = self.cell_type[0]  # Base pilot's hazard types
        unseen = (types == CELL_PLAIN) & (self.cell_rays < PLAIN_SEEN_RAYS)  # A hazard may lie beyond what was seen
        return torch.where(unseen, torch.full_like(types, CELL_UNKNOWN), types)

    def _route_update(self, cell: torch.Tensor):
        """Branch tags -> labels -> dead-end hint -> stay / go back -> virtual walls (skipped while nothing changed)."""
        rc = (int(cell[0, 0].item()), int(cell[0, 1].item()))  # Robot cell
        trusted = self._trusted_types()  # Hazard types the layer may rely on
        key = (  # Everything the update depends on
            rc,
            tuple(self.ap.walls[0].flatten().tolist()),
            tuple(trusted.flatten().tolist()),
            float(self.cell_deaths[0].sum().item()),
            self.fresh_attempt,
            self.route_choice,
            tuple(sorted(self.route_closed)),
        )
        if key == self._route_key:
            return  # Same inputs: same result
        walls = self.ap.walls[0].tolist()  # Real wall map [i][j][d]: 1 wall, 0 open, -1 unknown
        types = trusted.tolist()  # Hazard type per cell [i][j]
        exits, tag, dist, parent = self._branches(walls)  # Branch of every known cell
        self._update_labels(walls, types, exits, tag, dist, parent)  # Label per exit
        self._hint_dead_ends(walls, types, exits, tag)  # Plain side cells of a SHORT branch
        if any(t == MULTI for t in tag.values()):  # The branches are known to meet: nothing to choose any more
            if self.route_choice is not None or self.route_closed:
                self._log("branches meet beyond the start: route plan dropped, base planner only")
            self.route_choice, self.route_closed = None, set()
        else:
            self._decide(rc, walls, types, exits, tag, dist, parent)  # Stay or go back
        self._build_vwalls(rc, walls, exits, tag)  # Virtual walls for the decision
        self._route_key = key[:4] + (  # Record the inputs as they are now (the decision may have changed them)
            self.fresh_attempt,
            self.route_choice,
            tuple(sorted(self.route_closed)),
        )

    def _in_grid(self, i: int, j: int) -> bool:
        return 0 <= i < self.num_cells and 0 <= j < self.num_cells

    def _branches(self, walls: list) -> tuple[list, dict, dict, dict]:
        """Exits of the start; for every cell reached from one through known-open edges: its exit, distance, parent."""
        si, sj = self.start_cell  # Start cell
        exits = [  # Known-open sides of the start
            d for d, (di, dj) in enumerate(DIRS) if walls[si][sj][d] == 0 and self._in_grid(si + di, sj + dj)
        ]
        tag, dist, parent = {}, {}, {}  # Cell -> exit (or MULTI), cells from the start, previous cell toward the start
        for x in exits:
            first = (si + DIRS[x][0], sj + DIRS[x][1])  # First cell of this branch
            seen, par = {first: 1}, {first: self.start_cell}  # BFS distance from the start, parent
            queue = deque([first])
            while queue:
                c = queue.popleft()
                for d, (di, dj) in enumerate(DIRS):
                    nb = (c[0] + di, c[1] + dj)  # Neighbour across side d
                    if walls[c[0]][c[1]][d] != 0 or nb == self.start_cell or nb in seen or not self._in_grid(*nb):
                        continue  # Only known-open edges, never back through the start
                    seen[nb], par[nb] = seen[c] + 1, c
                    queue.append(nb)
            for c, r in seen.items():
                if c in tag and tag[c] != x:  # Reached from two exits: the branches meet here
                    tag[c] = MULTI
                else:
                    tag[c], dist[c], parent[c] = x, r, par[c]
        return exits, tag, dist, parent

    def _update_labels(self, walls: list, types: list, exits: list, tag: dict, dist: dict, parent: dict):
        """Label every exit from what its branch shows; print the labels that changed."""
        for x in exits:
            region = [c for c, t in tag.items() if t == x]  # Cells of this branch only
            label, why = self._label_branch(walls, types, x, region, tag, dist, parent)
            if label != self.route_label.get(x, UNKNOWN):
                self.route_label[x] = label
                si, sj = self.start_cell
                first = (si + DIRS[x][0], sj + DIRS[x][1])
                self._log(f"exit {DIR_NAMES[x]} {first}: {label}" + (f" ({why})" if why else ""), force=True)

    def _label_branch(
        self, walls: list, types: list, x: int, region: list, tag: dict, dist: dict, parent: dict
    ) -> tuple[str, str]:
        """(label, reason) of one exit's branch. Only clues that never failed in the generated mazes."""
        short_kinds = (CELL_STEEP_RAMP, CELL_TRENCH) + ((CELL_SPIKE_SLALOM,) if self.slalom_is_short else ())
        short_hit = next((c for c in region if types[c[0]][c[1]] in short_kinds), None)  # Short-route-only hazard
        long_hit = next((c for c in region if types[c[0]][c[1]] == CELL_LOW_RAMP), None)  # Long-route-only hazard
        short_name = TYPE_NAMES[types[short_hit[0]][short_hit[1]]] if short_hit else ""  # For the log
        if short_hit and long_hit:  # Impossible in a real maze: a misread cell, trust neither
            return UNKNOWN, f"conflict: {short_name} at {short_hit}, low ramp at {long_hit}"
        if short_hit:
            return SHORT, f"{short_name} at {short_hit}"
        if long_hit:
            return LONG, f"low ramp at {long_hit}"
        if region and self._walled_in(walls, region, tag):
            return DEAD, "walled in without the goal"
        for c in region:  # A spike bed 2-3 cells in: the cell after it decides (short route: never plain)
            if types[c[0]][c[1]] != CELL_SPIKE_LANE or dist[c] not in (2, 3):
                continue
            p = parent[c]  # Cell before the bed
            step = (c[0] - p[0], c[1] - p[1])  # Direction of travel through the bed (beds are straight cells)
            if step not in DIRS or walls[c[0]][c[1]][DIRS.index(step)] != 0:
                continue  # The far side is not known to be open yet
            f = (c[0] + step[0], c[1] + step[1])  # Cell after the bed
            if not self._in_grid(*f):
                continue
            tf = types[f[0]][f[1]]
            if tf == CELL_UNKNOWN:
                continue
            if tf in (CELL_PLAIN, CELL_LOW_RAMP):
                return NOT_SHORT, f"{TYPE_NAMES[tf]} at {f} after the spike bed at {c}"
            return SHORT, f"spike bed at {c} then {TYPE_NAMES[tf]} at {f} (about 95 % sure)"
        for r in (2, 3):  # Every way on 2 (or 3) cells in is plain: the short route would have a hazard or boxes there
            front = [c for c in region if dist[c] == r - 1]  # Cells in front of that ring
            ring = [c for c in region if dist[c] == r]
            if not front or not ring or any(-1 in walls[c[0]][c[1]] for c in front):
                break  # A side in front is not seen yet: a way on may be missing
            if any(types[c[0]][c[1]] == CELL_UNKNOWN for c in ring):
                break  # Not all classified yet
            if all(types[c[0]][c[1]] in (CELL_PLAIN, CELL_LOW_RAMP) for c in ring):
                return NOT_SHORT, f"only plain floor {r} cells in: {sorted(ring)}"
        return UNKNOWN, ""

    def _walled_in(self, walls: list, region: list, tag: dict) -> bool:
        """True when the branch is fully seen, closed all round and does not hold the goal (a dead end)."""
        if self.goal_cell in region:
            return False
        for c in region:
            for d, (di, dj) in enumerate(DIRS):
                w = walls[c[0]][c[1]][d]
                nb = (c[0] + di, c[1] + dj)
                if w == -1:
                    return False  # A side not seen yet
                if w == 0 and nb != self.start_cell and tag.get(nb) == MULTI:
                    return False  # Opens into the other branch
        return True

    def _hint_dead_ends(self, walls: list, types: list, exits: list, tag: dict):
        """On a SHORT branch, price a plain side cell as a likely dead end (the short route itself is never plain)."""
        si, sj = self.start_cell
        banned = {self.start_cell, self.goal_cell} | {(si + di, sj + dj) for di, dj in DIRS}  # Plain by design
        for x in exits:
            if self.route_label.get(x) != SHORT:
                continue
            first = (si + DIRS[x][0], sj + DIRS[x][1])  # Exit cell (plain, but on the route)
            for c, t in tag.items():
                if t != x or (types[c[0]][c[1]] in (CELL_PLAIN, CELL_UNKNOWN) and c != first):
                    continue  # Only from cells on the box / hazard trail
                for d, (di, dj) in enumerate(DIRS):
                    nb = (c[0] + di, c[1] + dj)
                    if walls[c[0]][c[1]][d] != 0 or nb in banned or nb in self.dead_end_hinted:
                        continue
                    if types[nb[0]][nb[1]] == CELL_PLAIN:
                        self.cell_penalty[0, nb[0], nb[1]] += DEAD_END_HINT  # Kept when the cell is re-classified
                        self.cell_cost[0, nb[0], nb[1]] += DEAD_END_HINT  # Effective now
                        self.dead_end_hinted.add(nb)
                        self._log(f"plain side cell {nb} off the short route: likely dead end (+{DEAD_END_HINT:g})")

    def _decide(self, rc: tuple, walls: list, types: list, exits: list, tag: dict, dist: dict, parent: dict):
        """Choose the exit at the start of an attempt, or stay / go back inside a branch."""
        si, sj = self.start_cell
        alive = [x for x in exits if self.route_label.get(x, UNKNOWN) != DEAD]  # Exits that may lead to the goal
        start_known = all(w != -1 for w in walls[si][sj])  # All four sides of the start seen
        two = start_known and len(alive) == 2  # Exactly two routes out of the start
        label = {x: self.route_label.get(x, UNKNOWN) for x in alive}  # Seen labels
        if two:  # One route's label gives the other's
            a, b = alive
            for x, y in ((a, b), (b, a)):
                if label[x] == UNKNOWN and label[y] in (SHORT, LONG, NOT_SHORT):
                    label[x] = LONG if label[y] == SHORT else SHORT
            if (label[a] == SHORT) == (label[b] == SHORT) and UNKNOWN not in (label[a], label[b]):
                two = False  # Both short or both long: a misread somewhere, do not switch on it
        if rc == self.start_cell:
            if self.fresh_attempt:
                self._choose_exit(alive, label, two, walls, types, tag)  # May leave the choice open
            return
        self.fresh_attempt = False  # Left the start without a decision: decide inside the branch
        d = tag.get(rc)  # Branch the robot is in
        if d is None or d == MULTI or self.route_switched or d in self.route_closed:
            return  # Unknown branch, or already going back / switched once this attempt
        ld = label.get(d, UNKNOWN)
        others = [x for x in alive if x != d]
        if not two or len(others) != 1:  # Three or more exits: never go back on a guess, only commit to SHORT
            if ld == SHORT and self.route_mode in ("auto", "short"):
                self._stay(d, ld, "known short route")
            return
        e, le = others[0], label[others[0]]
        k = dist[rc]  # Cells into the branch
        if self.route_mode == "short":
            go_back = ld in LONGISH and le == SHORT
            why = "PILOT_ROUTE=short"
        elif self.route_mode == "long":
            go_back = ld == SHORT and le in LONGISH
            why = "PILOT_ROUTE=long"
        elif ld in LONGISH and le == SHORT and k <= MAX_SWITCH_CELLS:
            t_stay, t_back, s_est, l_est = self._estimate(rc, d, e, k, walls, types, tag, dist, parent)
            go_back = t_back + SWITCH_MARGIN_S < t_stay
            why = f"stay {t_stay:.0f} s vs go back {t_back:.0f} s (short ~{s_est} cells, long ~{l_est} cells)"
        else:
            go_back = False
            why = "deeper than MAX_SWITCH_CELLS" if ld in LONGISH and le == SHORT else "known route"
        if go_back:
            self._go_back(d, e, ld, le, k, why)
        elif ld != UNKNOWN and le != UNKNOWN:
            self._stay(d, ld, why)

    def _choose_exit(self, alive: list, label: dict, two: bool, walls: list, types: list, tag: dict):
        """In the start cell at the start of an attempt: pick the exit (none while the labels are unknown)."""
        if self.route_mode in ("short", "long") or not two:
            want = SHORT if self.route_mode != "long" else LONG
            cands = [x for x in alive if label[x] == want or (two and want == LONG and label[x] == NOT_SHORT)]
            if len(cands) == 1 and (self.route_mode != "auto" or self._deaths_on(cands[0], tag) == 0):
                self._take(cands[0], label[cands[0]], f"known {label[cands[0]].lower()} route")
            return
        if UNKNOWN in label.values():
            return  # Not known yet: the base planner picks, the layer decides inside the branch
        s = next(x for x in alive if label[x] == SHORT)  # Short exit
        lng = next(x for x in alive if x != s)  # Long exit
        dist_goal = self._dist_to_goal(walls)
        si, sj = self.start_cell
        s_cells = max(1 + dist_goal.get((si + DIRS[s][0], sj + DIRS[s][1]), BIG), self.short_prior)
        l_cells = max(1 + dist_goal.get((si + DIRS[lng][0], sj + DIRS[lng][1]), BIG), s_cells + self.gap_prior)
        t_s = self._route_time(s_cells, "short", Counter()) + DEATH_S * self._deaths_on(s, tag)
        t_l = self._route_time(l_cells, "long", Counter()) + DEATH_S * self._deaths_on(lng, tag)
        x = s if t_s <= t_l else lng
        self._take(x, label[x], f"short {t_s:.0f} s vs long {t_l:.0f} s")

    def _estimate(
        self, rc: tuple, d: int, e: int, k: int, walls: list, types: list, tag: dict, dist: dict, parent: dict
    ) -> tuple[float, float, int, int]:
        """(stay time, go-back time, short cells, long cells) for a robot ``k`` cells into long branch ``d``."""
        dist_goal = self._dist_to_goal(walls)  # Optimistic cells to the goal (unknown = open), not via the start
        si, sj = self.start_cell
        e1 = (si + DIRS[e][0], sj + DIRS[e][1])  # First cell of the other exit
        s_cells = max(1 + dist_goal.get(e1, BIG), self.short_prior)  # Short route from the start
        l_cells = max(k + dist_goal.get(rc, BIG), s_cells + self.gap_prior)  # Long route from the start
        crossed = Counter(types[c[0]][c[1]] for c, t in tag.items() if t == d and dist[c] < k)  # Hazards behind
        share = (l_cells - k) / max(l_cells, 1)  # Part of the long route still ahead
        t_stay = self._route_time(l_cells - k, "long", crossed, share) + DEATH_S * self._deaths_on(d, tag)
        t_walk, c = 0.0, parent[rc]  # Walk back over the known cells to the start
        while True:
            tc = types[c[0]][c[1]]
            t_walk += CELL_TIME.get(tc, CELL_TIME[CELL_PLAIN]) + FAIL_CHANCE.get(tc, 0.0) * FALL_LOSS_S
            if c == self.start_cell:
                break
            c = parent[c]
        t_back = t_walk + PIVOT_S + self._route_time(s_cells, "short", Counter()) + DEATH_S * self._deaths_on(e, tag)
        return t_stay, t_back, s_cells, l_cells

    def _route_time(self, cells: int, kind: str, crossed: Counter, share: float = 1.0) -> float:
        """Estimated time [s] for ``cells`` cells of the ``kind`` route, with the hazards not yet ``crossed``.

        ``share`` = the part of the whole route still ahead (for the long route's extra turns).
        """
        base = ROUTE_CELL_TIME[kind]  # An ordinary cell of this route
        t = max(cells, 0) * base
        for h, count in self.level_hazards[kind].items():  # Hazards still ahead: their extra time and fall risk
            left = max(0, count - crossed.get(h, 0))
            t += left * (CELL_TIME[h] - base + FAIL_CHANCE.get(h, 0.0) * FALL_LOSS_S)
        if kind == "long":  # The long route is also twistier, and its plain dead ends cost detours
            t += TURN_S * EXTRA_TURNS_PRIOR * min(max(share, 0.0), 1.0)
            t += LONG_DETOUR_S_PER_CELL * max(cells, 0)
        return t

    def _dist_to_goal(self, walls: list) -> dict:
        """Cells from every cell to the goal (BFS, unknown edges open, never through the start)."""
        out = {self.goal_cell: 0}
        queue = deque([self.goal_cell])
        while queue:
            c = queue.popleft()
            for d, (di, dj) in enumerate(DIRS):
                nb = (c[0] + di, c[1] + dj)
                if walls[c[0]][c[1]][d] == 1 or nb == self.start_cell or nb in out or not self._in_grid(*nb):
                    continue
                out[nb] = out[c] + 1
                queue.append(nb)
        return out

    def _deaths_on(self, x: int, tag: dict) -> float:
        """Resets already attributed to cells of exit ``x``'s branch."""
        return sum(float(self.cell_deaths[0, c[0], c[1]].item()) for c, t in tag.items() if t == x)

    def _take(self, x: int, label: str, why: str):
        """Commit to exit ``x`` from the start cell."""
        self.route_choice, self.fresh_attempt = x, False
        self._log(f"start: take exit {DIR_NAMES[x]} route={label} ({why})")

    def _stay(self, d: int, label: str, why: str):
        """Commit to the branch the robot is in."""
        if self.route_choice != d:
            self.route_choice = d
            self._log(f"stay on exit {DIR_NAMES[d]} route={label} ({why})")

    def _go_back(self, d: int, e: int, ld: str, le: str, k: int, why: str):
        """Give up branch ``d`` (its frontier gets virtual walls) and head for exit ``e`` through the start."""
        self.route_closed.add(d)
        self.route_choice, self.route_switched = e, True
        self.go_backs += 1
        self._log(f"GO BACK: exit {DIR_NAMES[d]} is {ld}, {k} cells in -> exit {DIR_NAMES[e]} route={le} ({why})")

    def _build_vwalls(self, rc: tuple, walls: list, exits: list, tag: dict):
        """Virtual walls for the decision; dropped if they leave the robot no way to the goal."""
        n = self.num_cells
        v = torch.zeros((n, n, 4), dtype=torch.bool, device=self.device)
        si, sj = self.start_cell

        def block(i: int, j: int, d: int):  # Block an edge from both sides
            v[i, j, d] = True
            ni, nj = i + DIRS[d][0], j + DIRS[d][1]
            if self._in_grid(ni, nj):
                v[ni, nj, (d + 2) % 4] = True

        if self.route_choice is not None:  # Commit: the start's other exits are closed (not a branch being left)
            for x in exits:
                if x != self.route_choice and x not in self.route_closed:
                    block(si, sj, x)
        for x in self.route_closed:  # Going back: the branch can only be left through the start
            for c, t in tag.items():
                if t != x:
                    continue
                for d, (di, dj) in enumerate(DIRS):
                    nb = (c[0] + di, c[1] + dj)
                    if walls[c[0]][c[1]][d] == 1 or nb == self.start_cell or not self._in_grid(*nb):
                        continue
                    if tag.get(nb) != x:  # Unknown edge or a way out of the branch
                        block(c[0], c[1], d)
        self.vwalls[0] = v
        if not bool(v.any()):
            return
        merged = torch.where(self.vwalls, torch.ones_like(self.ap.walls), self.ap.walls)  # Walls the planner will see
        dist = HazardPilot.weighted_flood_fill(self, merged)  # Base cost-to-goal map (also warms its cache)
        if float(dist[0, rc[0], rc[1]].item()) >= 1e5:  # The plan leaves no way to the goal: drop it
            self._log("route plan leaves no way to the goal: dropped, base planner only")
            self.route_choice, self.route_closed = None, set()
            self.vwalls.zero_()

    def _log(self, msg: str, force: bool = False):
        """One line per new decision (labels always)."""
        if force or msg != self._last_msg:
            self._last_msg = msg
            print(f"[PILOT_MOD] t={self.sim_time:5.1f}s {msg}", flush=True)
