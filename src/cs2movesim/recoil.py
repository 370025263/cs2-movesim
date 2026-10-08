"""Spray compensation as a feed-forward term, measured from human sprays.

Pros pull the mouse down in a very consistent way while spraying (AK-47 over 7 shots: about 8 degrees, of which a
per-tick median curve explains roughly 70%). Letting a controller add that median curve while the trigger is held,
and letting a learned model predict only the residual, makes aiming far less sensitive to model under-prediction.

Curves are cumulative view-angle changes (degrees; pitch positive = looking down) as a function of ticks since the
trigger was pressed, per weapon, estimated by :func:`estimate_spray_curves` from OpenCS2 rounds.
"""
import json

import numpy as np

NOT_GUNS = {"flashbang", "smokegrenade", "highexplosivegrenade", "hegrenade", "molotov", "incgrenade", "decoy",
            "c4explosive", "taser"}


def is_gun(name):
    """Firearms only: grenades (held attack = preparing a throw), C4 (planting), knives and the Zeus are excluded."""
    name = str(name)
    return name not in NOT_GUNS and "knife" not in name and "bayonet" not in name and "dagger" not in name


def spray_ticks(attack, weapon, alive=None):
    """Ticks since the current spray started (0 on the press tick), -1 when the trigger is not held.
    A spray ends when the trigger is released, the weapon changes or the player dies."""
    attack = np.asarray(attack, bool)
    held = attack if alive is None else attack & np.asarray(alive, bool)
    tau = np.full(len(held), -1, dtype=np.int64)
    for t in range(len(held)):
        if not held[t]:
            continue
        if t > 0 and tau[t - 1] >= 0 and weapon[t] == weapon[t - 1]:
            tau[t] = tau[t - 1] + 1
        else:
            tau[t] = 0
    return tau


def estimate_spray_curves(rounds, max_ticks=192, min_sprays=30, min_length=13):
    """rounds: dicts from :func:`cs2movesim.opencs2.load_ticks`. For every firearm with at least ``min_sprays`` sprays
    longer than ``min_length`` ticks (about 3 shots for rifles), the median cumulative pitch and yaw change at each tick
    since the press, over the sprays still going at that tick; the curve stops where fewer than ``min_sprays`` remain.
    Returns {weapon: {"pitch": [...], "yaw": [...], "sprays": n}}."""
    pieces = {}
    for d in rounds:
        tau = spray_ticks(d["attack"], d["weapon"], d["alive"])
        starts = np.nonzero(tau == 0)[0]
        for s in starts:
            end = s
            while end + 1 < len(tau) and tau[end + 1] == tau[end] + 1:
                end += 1
            length = end - s + 1
            if length < min_length:
                continue
            n = min(length, max_ticks)
            if not is_gun(d["weapon"][s]):
                continue
            pieces.setdefault(str(d["weapon"][s]), []).append(
                (d["pitch"][s:s + n] - d["pitch"][s], d["yaw"][s:s + n] - d["yaw"][s]))
    curves = {}
    for weapon, items in pieces.items():
        if len(items) < min_sprays:
            continue
        pitch, yaw = [], []
        for t in range(max_ticks):
            alive = [(p[t], y[t]) for p, y in items if len(p) > t]
            if len(alive) < min_sprays:
                break
            pitch.append(float(np.median([a[0] for a in alive])))
            yaw.append(float(np.median([a[1] for a in alive])))
        curves[weapon] = {"pitch": pitch, "yaw": yaw, "sprays": len(items)}
    return curves


def increments(curve, tau):
    """Per-tick feed-forward increments (pitch, yaw) for spray ticks ``tau`` (from :func:`spray_ticks`)."""
    pitch = np.asarray(curve["pitch"]) if curve else np.zeros(1)
    yaw = np.asarray(curve["yaw"]) if curve else np.zeros(1)
    d_pitch = np.diff(pitch, prepend=0.0)
    d_yaw = np.diff(yaw, prepend=0.0)
    tau = np.asarray(tau)
    ok = (tau >= 0) & (tau < len(d_pitch))
    out = np.zeros((len(tau), 2))
    out[ok, 0] = d_pitch[tau[ok]]
    out[ok, 1] = d_yaw[tau[ok]]
    return out


def feedforward(attack, weapon, curves, alive=None):
    """Per-tick feed-forward (pitch, yaw) increments for a whole round [T, 2], and their running sum [T, 2]."""
    tau = spray_ticks(attack, weapon, alive)
    out = np.zeros((len(tau), 2))
    for name in set(map(str, np.asarray(weapon)[tau >= 0])):
        curve = curves.get(name)
        if curve:
            mask = np.asarray(weapon).astype(str) == name
            out[mask] = increments(curve, np.where(mask, tau, -1))[mask]
    return out, np.cumsum(out, axis=0)


class SprayCompensator:
    """Deployment side: call :meth:`step` once per tick with the trigger state and weapon; returns the (pitch, yaw)
    degrees to add to the mouse movement of that tick."""

    def __init__(self, curves):
        self.curves = curves
        self.tau = -1
        self.weapon = None

    def step(self, attack, weapon):
        if attack and self.tau >= 0 and weapon == self.weapon:
            self.tau += 1
        elif attack:
            self.tau = 0
        else:
            self.tau = -1
        self.weapon = weapon
        if self.tau < 0 or weapon not in self.curves:
            return 0.0, 0.0
        d = increments(self.curves[weapon], np.array([self.tau]))[0]
        return float(d[0]), float(d[1])


def save_curves(curves, path):
    with open(path, "w") as f:
        json.dump(curves, f)


def load_curves(path):
    with open(path) as f:
        return json.load(f)
