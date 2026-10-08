import numpy as np
import pytest

from cs2movesim import MovementParams, PlayerState, TrackingController, simulate, step
from cs2movesim.physics import BUTTONS

P = MovementParams()


def hold(*names, ticks=128, n=1):
    row = np.zeros(len(BUTTONS), bool)
    for name in names:
        row[BUTTONS.index(name)] = True
    return np.broadcast_to(row, (ticks, n, len(BUTTONS))).copy()


def run(buttons, vel=(0.0, 0.0), base=215.0, yaw=0.0, duck=0.0):
    state = PlayerState.create([[0.0, 0.0, 0.0]], [vel], duck=duck)
    pos, v = simulate(state, buttons, yaw, base, P)
    return state, pos, v


def test_accelerates_to_weapon_speed_and_not_beyond():
    _, _, v = run(hold("forward"))
    speed = np.hypot(v[:, 0, 0], v[:, 0, 1])
    assert speed.max() <= 215.0 + 1e-6
    assert speed[-1] == pytest.approx(215.0, abs=0.5)
    assert np.all(np.diff(speed) >= -1e-9)


def test_forward_is_plus_x_at_yaw_zero_and_left_is_plus_y():
    _, _, v = run(hold("forward"), base=250.0)
    assert v[-1, 0, 0] > 249 and abs(v[-1, 0, 1]) < 1e-6
    _, _, v = run(hold("left"), base=250.0)
    assert v[-1, 0, 1] > 249 and abs(v[-1, 0, 0]) < 1e-6


def test_diagonal_is_not_faster():
    _, _, v = run(hold("forward", "left"), base=250.0)
    assert np.hypot(v[-1, 0, 0], v[-1, 0, 1]) == pytest.approx(250.0, abs=0.5)


def test_release_stops_by_friction():
    _, _, v = run(hold(ticks=64), vel=(250.0, 0.0), base=250.0)
    speed = np.hypot(v[:, 0, 0], v[:, 0, 1])
    assert np.all(np.diff(speed) <= 1e-9)
    assert speed[-1] == 0.0


def test_counter_strafe_stops_faster_than_release():
    def ticks_to_stop(buttons):
        state = PlayerState.create([[0.0, 0.0, 0.0]], [(215.0, 0.0)])
        for t in range(64):
            b = buttons if np.hypot(*state.vel[0, :2]) > 50 else hold(ticks=1)[0]
            step(state, b, 0.0, 215.0, P)
            if np.hypot(*state.vel[0, :2]) < 50:
                return t + 1
        return 64
    assert ticks_to_stop(hold("back", ticks=1)[0]) < ticks_to_stop(hold(ticks=1)[0])


def test_walk_and_full_duck_modifiers():
    _, _, v = run(hold("forward", "walk"), base=250.0)
    assert np.hypot(*v[-1, 0, :2]) == pytest.approx(250.0 * P.walk_modifier, abs=0.5)
    _, _, v = run(hold("forward", "duck"), base=250.0, duck=1.0)
    assert np.hypot(*v[-1, 0, :2]) == pytest.approx(250.0 * P.duck_modifier, abs=0.5)


def test_jump_apex_and_landing():
    buttons = hold(ticks=80)
    buttons[0, 0, BUTTONS.index("jump")] = True
    state, pos, v = run(buttons)
    assert pos[:, 0, 2].max() == pytest.approx(P.jump_impulse ** 2 / (2 * P.gravity), abs=3.0)
    assert state.on_ground[0] and pos[-1, 0, 2] == 0.0


def test_holding_jump_does_not_rejump_without_autobhop():
    state, pos, _ = run(hold("jump", ticks=120))
    landings = np.sum((pos[1:, 0, 2] == 0) & (pos[:-1, 0, 2] > 0))
    assert landings == 1


def test_controller_reproduces_a_feasible_run_and_stop():
    """Reference made by the simulator itself (hold W for 96 ticks, then release): the controller must follow it closely."""
    buttons = hold("forward", ticks=160)
    buttons[96:] = False
    _, reference, _ = run(buttons, yaw=30.0)
    controller = TrackingController(P)
    state = PlayerState.create([[0.0, 0.0, 0.0]])
    start = np.zeros((1, 3))
    errors = []
    for t in range(159):
        now = start if t == 0 else reference[t - 1]
        desired = controller.desired_velocity(state, now, reference[t])
        chosen = controller.choose(state, np.full(1, 30.0), np.full(1, 215.0), desired)
        step(state, chosen, 30.0, 215.0, P)
        errors.append(np.linalg.norm(state.pos[0, :2] - reference[t, 0, :2]))
    assert np.max(errors) < 2.0
    assert np.hypot(*state.vel[0, :2]) < 1.0
