"""Tick-level CS2 player movement (Source-style ground/air movement), vectorised over many players with numpy.

The update follows the classic Source ``CGameMovement`` order for one tick:

1. duck amount moves towards the duck button state;
2. on the ground: friction, then optional jump (needs a fresh press unless ``auto_bhop``);
3. wish direction from WASD and yaw, wish speed clamped to the effective max speed
   (weapon speed x walk / duck modifiers);
4. ground ``Accelerate`` (or ``AirAccelerate`` in the air), then the ground speed clamp;
5. gravity (half before and half after the position update, like Source) and the position update.

There is no map collision by default. Pass ``ground_height`` (a callable ``(x, y, z) -> ground z``)
to let players land on terrain; otherwise the ground is the plane ``z = ground_z`` given at reset.
All parameters live in :class:`cs2movesim.params.MovementParams` and were calibrated against
OpenCS2 professional demos (see ``cs2movesim.bench``).
"""
from dataclasses import dataclass, field

import numpy as np

from .params import MovementParams

FORWARD, BACK, LEFT, RIGHT, WALK, DUCK, JUMP = range(7)
BUTTONS = ("forward", "back", "left", "right", "walk", "duck", "jump")


@dataclass
class PlayerState:
    """State of ``n`` players. Arrays are float64 unless noted."""

    pos: np.ndarray  # [n, 3]
    vel: np.ndarray  # [n, 3]
    on_ground: np.ndarray  # [n] bool
    duck: np.ndarray  # [n] duck amount 0..1
    jump_held: np.ndarray  # [n] bool, jump was down last tick
    ground_z: np.ndarray = field(default=None)  # [n] flat ground height used when no ground_height callable is given

    @classmethod
    def create(cls, pos, vel=None, on_ground=True, duck=0.0):
        pos = np.atleast_2d(np.asarray(pos, dtype=np.float64)).copy()
        n = len(pos)
        vel = np.zeros((n, 3)) if vel is None else np.atleast_2d(np.asarray(vel, dtype=np.float64)).copy()
        if vel.shape[1] == 2:
            vel = np.concatenate([vel, np.zeros((n, 1))], axis=1)
        return cls(pos=pos, vel=vel, on_ground=np.broadcast_to(np.asarray(on_ground, bool), (n,)).copy(),
                   duck=np.broadcast_to(np.asarray(duck, np.float64), (n,)).copy(), jump_held=np.zeros(n, bool),
                   ground_z=pos[:, 2].copy())

    def copy(self):
        return PlayerState(self.pos.copy(), self.vel.copy(), self.on_ground.copy(), self.duck.copy(), self.jump_held.copy(),
                           None if self.ground_z is None else self.ground_z.copy())

    @property
    def speed2d(self):
        return np.hypot(self.vel[:, 0], self.vel[:, 1])


def wish_velocity(buttons, yaw, max_speed, params):
    """buttons [n, 7] bool, yaw [n] degrees, max_speed [n] effective max speed -> (wishdir [n, 2], wishspeed [n])."""
    rad = np.radians(yaw)
    forward = np.stack([np.cos(rad), np.sin(rad)], axis=1)
    right = np.stack([np.sin(rad), -np.cos(rad)], axis=1)
    fmove = params.forward_speed * (buttons[:, FORWARD].astype(np.float64) - buttons[:, BACK])
    smove = params.side_speed * (buttons[:, RIGHT].astype(np.float64) - buttons[:, LEFT])
    wish = forward * fmove[:, None] + right * smove[:, None]
    speed = np.linalg.norm(wish, axis=1)
    direction = np.where(speed[:, None] > 1e-9, wish / np.maximum(speed, 1e-9)[:, None], 0.0)
    return direction, np.minimum(speed, max_speed)


def effective_max_speed(base_speed, buttons, duck, params):
    """Weapon base speed with walk and duck modifiers. Ducking slows down in proportion to the duck amount."""
    speed = np.asarray(base_speed, dtype=np.float64) * np.where(buttons[:, WALK], params.walk_modifier, 1.0)
    duck_factor = 1.0 - (1.0 - params.duck_modifier) * duck
    return np.minimum(speed * duck_factor, params.max_speed)


def friction(vel, dt, params):
    """Ground friction on the horizontal velocity (in place)."""
    speed = np.hypot(vel[:, 0], vel[:, 1])
    control = np.maximum(speed, params.stop_speed)
    new = np.maximum(speed - control * params.friction * dt, 0.0)
    scale = np.where(speed > 0.1, new / np.maximum(speed, 1e-9), 0.0)
    vel[:, 0] *= scale
    vel[:, 1] *= scale


def accelerate(vel, wishdir, wishspeed, accel, scale_speed, dt):
    """Source ``Accelerate`` on the horizontal velocity (in place). ``scale_speed`` multiplies accel x dt."""
    current = vel[:, 0] * wishdir[:, 0] + vel[:, 1] * wishdir[:, 1]
    add = wishspeed - current
    step = np.minimum(accel * dt * scale_speed, np.maximum(add, 0.0))
    step = np.where(add > 0, step, 0.0)
    vel[:, 0] += step * wishdir[:, 0]
    vel[:, 1] += step * wishdir[:, 1]


def step(state, buttons, yaw, base_speed, params=None, ground_height=None):
    """Advance ``state`` by one tick in place and return it.

    buttons: [n, 7] bool in the order of :data:`BUTTONS`; yaw: [n] degrees (0 = +x, 90 = +y);
    base_speed: [n] or scalar weapon max speed (see :data:`cs2movesim.params.WEAPON_SPEED`).
    """
    params = params or MovementParams()
    buttons = np.atleast_2d(np.asarray(buttons, dtype=bool))
    n = len(state.pos)
    yaw = np.broadcast_to(np.asarray(yaw, dtype=np.float64), (n,))
    base_speed = np.broadcast_to(np.asarray(base_speed, dtype=np.float64), (n,))
    dt = 1.0 / params.tick_rate

    target = buttons[:, DUCK].astype(np.float64)
    state.duck += np.clip(target - state.duck, -params.duck_rate * dt, params.duck_rate * dt)

    ground = state.on_ground
    vel = state.vel
    friction_vel = vel[ground]
    friction(friction_vel, dt, params)
    vel[ground] = friction_vel

    jump = ground & buttons[:, JUMP] & (params.auto_bhop | ~state.jump_held)
    vel[jump, 2] = params.jump_impulse
    state.on_ground = ground & ~jump
    state.jump_held = buttons[:, JUMP].copy()

    max_speed = effective_max_speed(base_speed, buttons, state.duck, params)
    wishdir, wishspeed = wish_velocity(buttons, yaw, max_speed, params)
    ground = state.on_ground
    if ground.any():
        g = vel[ground]
        scale = wishspeed[ground] if params.accel_scale == "wish" else np.maximum(base_speed[ground], params.accel_floor)
        accelerate(g, wishdir[ground], wishspeed[ground], params.accelerate, scale * params.surface_friction, dt)
        speed = np.hypot(g[:, 0], g[:, 1])
        over = speed > max_speed[ground]
        g[over, :2] *= (max_speed[ground][over] / speed[over])[:, None]
        g[:, 2] = 0.0
        vel[ground] = g
    air = ~ground
    if air.any():
        a = vel[air]
        capped = np.minimum(wishspeed[air], params.air_max_wishspeed)
        current = a[:, 0] * wishdir[air, 0] + a[:, 1] * wishdir[air, 1]
        add = capped - current
        amount = np.where(add > 0, np.minimum(params.air_accelerate * wishspeed[air] * dt, add), 0.0)
        a[:, 0] += amount * wishdir[air, 0]
        a[:, 1] += amount * wishdir[air, 1]
        a[:, 2] -= 0.5 * params.gravity * dt
        vel[air] = a

    state.pos += vel * dt

    if air.any():
        vel[air, 2] -= 0.5 * params.gravity * dt
        floor = state.ground_z[air] if ground_height is None else ground_height(state.pos[air, 0], state.pos[air, 1], state.pos[air, 2])
        landed = (state.pos[air, 2] <= floor) & (vel[air, 2] <= 0)
        idx = np.nonzero(air)[0][landed]
        state.pos[idx, 2] = floor[landed]
        vel[idx, 2] = 0.0
        state.on_ground[idx] = True
    return state


def simulate(state, buttons, yaw, base_speed, params=None, ground_height=None):
    """Run ``T`` ticks. buttons [T, n, 7], yaw [T, n], base_speed [T, n] or scalar.
    Returns (positions [T, n, 3], velocities [T, n, 3]) recorded after each tick. ``state`` is modified in place."""
    buttons = np.asarray(buttons, dtype=bool)
    ticks = buttons.shape[0]
    base_speed = np.broadcast_to(np.asarray(base_speed, dtype=np.float64), buttons.shape[:2])
    yaw = np.broadcast_to(np.asarray(yaw, dtype=np.float64), buttons.shape[:2])
    positions = np.zeros((ticks,) + state.pos.shape)
    velocities = np.zeros((ticks,) + state.vel.shape)
    for t in range(ticks):
        step(state, buttons[t], yaw[t], base_speed[t], params, ground_height)
        positions[t] = state.pos
        velocities[t] = state.vel
    return positions, velocities
