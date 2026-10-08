"""Movement parameters. Defaults are the CS2 server defaults (sv_accelerate 5.5, sv_friction 5.2, sv_stopspeed 80, ...);
values marked "calibrated" were fitted on OpenCS2 professional demos with ``cs2movesim-replay --fit``."""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class MovementParams:
    tick_rate: float = 64.0
    forward_speed: float = 450.0  # cl_forwardspeed
    side_speed: float = 450.0  # cl_sidespeed
    max_speed: float = 320.0  # sv_maxspeed
    accelerate: float = 5.5  # sv_accelerate
    accel_scale: str = "wish"  # "wish": accel x dt x wishspeed (classic Source); "weapon": x max(weapon speed, accel_floor)
    accel_floor: float = 250.0
    friction: float = 5.2  # sv_friction
    stop_speed: float = 80.0  # sv_stopspeed
    surface_friction: float = 1.0
    walk_modifier: float = 0.52  # shift-walk speed fraction
    duck_modifier: float = 0.34  # fully ducked speed fraction
    duck_rate: float = 8.0  # duck amount change per second (0 -> 1 in 1 / duck_rate s)
    jump_impulse: float = 301.993378  # sqrt(2 x 800 x 57)
    gravity: float = 800.0  # sv_gravity
    air_accelerate: float = 12.0  # sv_airaccelerate
    air_max_wishspeed: float = 30.0  # sv_air_max_wishspeed
    auto_bhop: bool = False  # sv_autobunnyhopping

    def with_(self, **changes):
        return replace(self, **changes)


# Calibrated preset: best fit of open-loop replays of human inputs on OpenCS2 dust2 demos (32-tick windows).
# Lower than the server defaults because CS2 applies inputs sub-tick; at tick resolution that looks like weaker
# acceleration. Median velocity error after 32 ticks: 7.4 u/s (server defaults: 9.7 u/s) on 11648 windows from 700 rounds.
OPENCS2_FIT = MovementParams(accelerate=5.0, friction=4.4, stop_speed=70.0)

# Weapon max speeds (game units / s) measured on OpenCS2 pro demos (``cs2movesim speeds``): the most frequent steady running
# speed per weapon. Only weapons seen in the sample are listed; unknown weapons fall back to DEFAULT_SPEED, knives to 250.
# Scoping is not simulated: the AWP is listed unscoped (200); scoped it moves at 100.
DEFAULT_SPEED = 250.0
WEAPON_SPEED = {
    "ak47": 215.0, "galilar": 215.0, "m4a1": 225.0, "m4a1_silencer": 225.0, "awp": 200.0, "ssg08": 230.0,
    "deagle": 230.0, "usp_silencer": 240.0, "glock": 240.0, "fiveseven": 240.0, "mp9": 240.0,
    "flashbang": 245.0, "smokegrenade": 245.0, "highexplosivegrenade": 245.0, "incgrenade": 245.0, "molotov": 245.0,
    "c4explosive": 250.0,
}


def weapon_speed(name):
    name = str(name)
    if name.startswith("knife") or "bayonet" in name:
        return 250.0
    return WEAPON_SPEED.get(name, DEFAULT_SPEED)
