"""Closed-loop tracking benchmark: can a controller reproduce a human's movement from a plan of their future path?

For each human window (positions, yaw, duck button, weapon speed per tick) the simulator starts from the human's
state. Every ``replan`` ticks (4 = one 16 Hz model step) the controller gets a new plan anchored at the *simulated*
position: planned position at tick t+j = simulated position at t + (human position at t+j - human position at t),
optionally distorted (shrink towards zero, Gaussian noise growing with the horizon) to mimic model error.

Plan formats:
- ``dense``: one point per tick for the next ``dense_ticks`` ticks (the near part of a non-uniform trajectory);
- ``sparse``: points only every ``replan`` ticks, linearly interpolated in between.

Reported against the human, per tick: horizontal velocity error; speed while the human is shooting
(attack held); time to stop (speed from above 150 to below 50) after the human starts a stop; final position drift.
"""
import numpy as np

from .controller import TrackingController
from .params import MovementParams
from .physics import DUCK, PlayerState, step


def plan_offsets(human_pos, t, horizon, mode, replan, shrink, noise, rng):
    """Planned displacement for ticks t+1..t+horizon relative to tick t: [n, horizon, 2]."""
    future = human_pos[t + 1:t + horizon + 1, :, :2] - human_pos[t][None, :, :2]  # [h, n, 2]
    future = np.transpose(future, (1, 0, 2))
    if mode == "sparse":
        knots = np.arange(replan, horizon + 1, replan)
        full = np.zeros_like(future)
        prev_j, prev = 0, np.zeros_like(future[:, 0])
        for k in knots:
            for j in range(prev_j + 1, k + 1):
                w = (j - prev_j) / (k - prev_j)
                full[:, j - 1] = prev + w * (future[:, k - 1] - prev)
            prev_j, prev = k, future[:, k - 1]
        future = full
    if shrink:
        future = future * (1.0 - shrink)
    if noise:
        steps = np.arange(1, horizon + 1)[None, :, None] / replan
        future = future + rng.normal(0, noise, size=(future.shape[0], 1, 2)) * steps
    return future


def run(windows, mode="dense", replan=4, gain=8.0, shrink=0.0, noise=0.0, params=None, seed=0, allow_walk=True):
    """windows: dict with pos [T+1, n, 3], yaw [T+1, n], buttons [T+1, n, 7], base [T+1, n], attack [T+1, n]
    (as saved by ``cs2movesim-pack``). Returns a dict of metrics and the simulated velocities [T, n, 2]."""
    params = params or MovementParams()
    rng = np.random.default_rng(seed)
    pos, yaw, buttons, base = windows["pos"], windows["yaw"], windows["buttons"], windows["base"]
    vel_h = windows["vel"]
    ticks, n = pos.shape[0] - 1, pos.shape[1]
    state = PlayerState.create(pos[0], vel_h[0, :, :2], duck=buttons[0, :, DUCK].astype(float))
    controller = TrackingController(params, gain=gain, allow_walk=allow_walk)
    sim_vel = np.zeros((ticks, n, 2))
    plan, anchor, plan_t = None, None, 0
    for t in range(ticks):
        if t % replan == 0:
            horizon = min(max(replan * 2, 8), ticks - t)
            plan = plan_offsets(pos, t, horizon, mode, replan, shrink, noise, rng)
            anchor, plan_t = state.pos[:, :2].copy(), t
        j = t - plan_t
        now = anchor + (plan[:, j - 1] if j > 0 else 0.0)
        nxt = anchor + plan[:, j]
        desired = controller.desired_velocity(state, np.concatenate([now, np.zeros((n, 1))], 1), np.concatenate([nxt, np.zeros((n, 1))], 1))
        chosen = controller.choose(state, yaw[t], base[t], desired, duck=buttons[t, :, DUCK])
        step(state, chosen, yaw[t], base[t], params)
        sim_vel[t] = state.vel[:, :2]
    human_vel = vel_h[1:, :, :2]
    err = np.linalg.norm(sim_vel - human_vel, axis=2)
    sim_speed = np.linalg.norm(sim_vel, axis=2)
    human_speed = np.linalg.norm(human_vel, axis=2)
    shooting = windows["attack"][1:].astype(bool)
    metrics = {
        "vel_err_median": float(np.median(err)), "vel_err_p90": float(np.percentile(err, 90)),
        "drift_end": float(np.median(np.linalg.norm(state.pos[:, :2] - pos[-1, :, :2], axis=1))),
    }
    if shooting.any():
        metrics["shoot_speed_human"] = float(np.median(human_speed[shooting]))
        metrics["shoot_speed_sim"] = float(np.median(sim_speed[shooting]))
        metrics["shoot_faster_by_20"] = float(np.mean(sim_speed[shooting] > human_speed[shooting] + 20))
    stops_h, stops_s = [], []
    for i in range(n):
        s_h, s_s = human_speed[:, i], sim_speed[:, i]
        for t in range(1, ticks - 16):
            if s_h[t - 1] > 150 and s_h[t] <= 150:
                h = np.nonzero(s_h[t:t + 16] < 50)[0]
                if len(h):
                    s = np.nonzero(s_s[t:t + 32] < 50)[0] if t + 32 <= ticks else np.nonzero(s_s[t:] < 50)[0]
                    stops_h.append(h[0])
                    stops_s.append(s[0] if len(s) else 32)
    if stops_h:
        metrics["stop_ticks_human"] = float(np.median(stops_h))
        metrics["stop_ticks_sim"] = float(np.median(stops_s))
        metrics["stops"] = len(stops_h)
    return metrics, sim_vel
