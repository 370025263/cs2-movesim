"""Trajectory-tracking controller: turns a planned future path into per-tick button presses.

Every tick the controller tries all candidate button sets (no movement, the 8 WASD directions, each with and without
walk) on a copy of the current state with the simulator's own one-tick physics, and keeps the set whose resulting
velocity is closest to the desired velocity. The desired velocity is the plan's velocity for the next tick plus a
proportional correction towards the planned position.

Aim needs no controller: view angles are an exact function of mouse input, so the mouse delta for a tick is simply
the planned angle for the next tick minus the current angle.
"""
import numpy as np

from .params import MovementParams
from .physics import DUCK, FORWARD, BACK, LEFT, RIGHT, WALK, step

DIRECTIONS = [(), ("f",), ("f", "l"), ("l",), ("b", "l"), ("b",), ("b", "r"), ("r",), ("f", "r")]
_INDEX = {"f": FORWARD, "b": BACK, "l": LEFT, "r": RIGHT}


def candidate_buttons(allow_walk=True):
    """[k, 7] bool: every direction, with and without walk (no-movement only once)."""
    out = []
    for walk in ((False, True) if allow_walk else (False,)):
        for direction in DIRECTIONS:
            if walk and not direction:
                continue
            row = np.zeros(7, bool)
            for key in direction:
                row[_INDEX[key]] = True
            row[WALK] = walk
            out.append(row)
    return np.array(out)


class TrackingController:
    """Greedy one-tick inverse-dynamics controller over the simulator.

    gain: per-second proportional gain on the position error (0 disables position feedback).
    """

    def __init__(self, params=None, gain=8.0, allow_walk=True):
        self.params = params or MovementParams()
        self.gain = gain
        self.candidates = candidate_buttons(allow_walk)

    def choose(self, state, yaw, base_speed, desired_velocity, duck=None):
        """state: PlayerState of n players; yaw [n]; base_speed [n]; desired_velocity [n, 2] for the end of this tick;
        duck [n] bool (copied into the chosen buttons). Returns buttons [n, 7]."""
        n = len(state.pos)
        k = len(self.candidates)
        trial = state.copy()
        trial.pos = np.repeat(trial.pos, k, 0)
        trial.vel = np.repeat(trial.vel, k, 0)
        trial.on_ground = np.repeat(trial.on_ground, k, 0)
        trial.duck = np.repeat(trial.duck, k, 0)
        trial.jump_held = np.repeat(trial.jump_held, k, 0)
        trial.ground_z = np.repeat(trial.ground_z, k, 0)
        buttons = np.tile(self.candidates, (n, 1))
        if duck is not None:
            buttons[:, DUCK] = np.repeat(np.asarray(duck, bool), k)
        step(trial, buttons, np.repeat(np.broadcast_to(yaw, (n,)), k), np.repeat(np.broadcast_to(base_speed, (n,)), k), self.params)
        miss = np.linalg.norm(trial.vel[:, :2].reshape(n, k, 2) - np.asarray(desired_velocity)[:, None], axis=2)
        best = miss.argmin(1)
        chosen = self.candidates[best].copy()
        if duck is not None:
            chosen[:, DUCK] = np.asarray(duck, bool)
        return chosen

    def desired_velocity(self, state, planned_pos_now, planned_pos_next):
        """Velocity that follows the plan for one tick and pulls the player back towards the planned position."""
        rate = self.params.tick_rate
        return (planned_pos_next[:, :2] - planned_pos_now[:, :2]) * rate + self.gain * (planned_pos_now[:, :2] - state.pos[:, :2])
