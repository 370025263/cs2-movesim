"""Calibration against OpenCS2 pro demos: window selection, weapon speed estimation, open-loop replay of human inputs."""
from collections import defaultdict

import numpy as np

from .params import MovementParams, weapon_speed
from .physics import DUCK, JUMP, PlayerState, simulate


def usable_ticks(d, blocked_ticks=8):
    """Ticks that the simulator can model: alive, real row, flat ground (|vz| < 0.5), no jump, not defusing (use),
    not planting (C4 + attack), and not pushing into an obstacle (movement keys held with speed < 5 for ``blocked_ticks``)."""
    b = d["buttons"]
    speed = np.hypot(d["vel"][:, 0], d["vel"][:, 1])
    bad = ~d["alive"] | ~d["present"] | (np.abs(d["vel"][:, 2]) > 0.5) | b[:, JUMP] | d["use"] \
        | ((d["weapon"] == "c4explosive") & d["attack"])
    blocked = b[:, :4].any(1) & (speed < 5)
    run = np.zeros(len(speed), int)
    for i in range(1, len(speed)):
        run[i] = run[i - 1] + 1 if blocked[i] else 0
    return ~(bad | (run >= blocked_ticks))


def estimate_weapon_speeds(rounds, min_samples=50):
    """Max running speed per weapon: the most frequent horizontal speed (rounded to 1 u/s, among speeds >= 100) over ticks
    where only movement keys (no walk / duck / jump) have been held unchanged for 32 ticks on flat ground. The mode is used
    instead of a high percentile because right after a weapon switch the previous weapon's speed lingers for a moment."""
    speeds = defaultdict(list)
    for d in rounds:
        b = d["buttons"]
        ok = usable_ticks(d) & b[:, :4].any(1) & ~b[:, 4] & ~b[:, 5]
        steady = ok.copy()
        for lag in range(1, 33):
            steady[lag:] &= (b[lag:] == b[:-lag]).all(1) & ok[:-lag]
        steady[:32] = False
        speed = np.hypot(d["vel"][:, 0], d["vel"][:, 1])
        steady &= speed > 30
        for weapon, value in zip(d["weapon"][steady], speed[steady]):
            speeds[str(weapon)].append(value)
    table = {}
    for weapon, values in speeds.items():
        values = np.round(np.asarray(values))
        values = values[values >= 100]
        if len(values) >= min_samples:
            uniq, counts = np.unique(values, return_counts=True)
            table[weapon] = (float(uniq[counts.argmax()]), int(len(values)))
    return table


def pack_windows(rounds, horizon, stride=None, min_speed=30.0):
    """Cut usable windows of ``horizon`` ticks. Returns arrays shaped [horizon + 1, n, ...]:
    pos, vel, yaw, buttons, attack, base (weapon max speed per tick)."""
    stride = stride or horizon
    cols = {k: [] for k in ("pos", "vel", "yaw", "buttons", "attack", "base")}
    for d in rounds:
        ok = usable_ticks(d)
        cum = np.concatenate([[0], np.cumsum(~ok)])
        speed = np.hypot(d["vel"][:, 0], d["vel"][:, 1])
        t0 = 8
        while t0 < len(ok) - horizon - 2:
            if cum[t0 + horizon + 1] - cum[t0] == 0 and speed[t0:t0 + horizon].max() > min_speed:
                idx = np.arange(t0, t0 + horizon + 1)
                for key in ("pos", "vel", "yaw", "buttons", "attack"):
                    cols[key].append(d[key][idx])
                cols["base"].append([weapon_speed(w) for w in d["weapon"][idx]])
                t0 += stride
            else:
                t0 += 16
    if not cols["pos"]:
        raise ValueError("no usable windows")
    return {k: np.stack(v, axis=1) for k, v in cols.items()}


def replay_errors(windows, params=None, lag=1):
    """Open-loop replay: start from the recorded velocity, feed the recorded buttons and yaw (inputs of tick t act on the
    velocity recorded at tick t + lag), return |simulated - recorded| horizontal velocity [horizon, n]."""
    params = params or MovementParams()
    horizon = windows["pos"].shape[0] - 1
    start_duck = windows["buttons"][0, :, DUCK].astype(float)
    state = PlayerState.create(windows["pos"][0], windows["vel"][0, :, :2], duck=start_duck)
    rows = np.arange(1, horizon + 1) - lag
    _, vel = simulate(state, windows["buttons"][rows], windows["yaw"][rows], windows["base"][rows], params)
    return np.linalg.norm(vel[:, :, :2] - windows["vel"][1:, :, :2], axis=2)
