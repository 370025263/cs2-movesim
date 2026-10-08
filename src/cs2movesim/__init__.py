"""cs2-movesim: tick-level CS2 movement simulator, a trajectory-tracking controller and OpenCS2 benchmarks."""
from .controller import TrackingController, candidate_buttons
from .params import DEFAULT_SPEED, OPENCS2_FIT, WEAPON_SPEED, MovementParams, weapon_speed
from .physics import BUTTONS, PlayerState, simulate, step

__version__ = "0.1.0"
__all__ = ["BUTTONS", "DEFAULT_SPEED", "OPENCS2_FIT", "WEAPON_SPEED", "MovementParams", "PlayerState", "TrackingController",
           "candidate_buttons", "simulate", "step", "weapon_speed", "__version__"]
