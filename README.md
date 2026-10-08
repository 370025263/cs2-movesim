# cs2-movesim

Tick-level **Counter-Strike 2 player movement simulator**, a **trajectory-tracking controller** that turns a planned path
into per-tick key presses, and benchmarks that calibrate and test both against professional demos from
[OpenCS2](https://huggingface.co/datasets/blanchon/opencs2_dataset).

It was written to answer one question before training an agent: *if a model predicts the player's future path
(instead of predicting WASD directly), can a controller actually execute that path with the precision of a pro —
counter-strafing before the first shot, holding still while spraying?*

```
pip install cs2-movesim              # simulator + controller (numpy only)
pip install "cs2-movesim[opencs2]"   # + pandas/pyarrow to read OpenCS2 ticks.parquet
```

## Simulator

`cs2movesim.step` advances any number of players by one 64 Hz tick, vectorised with numpy, following the Source
movement order: duck amount, ground friction, jump, wish direction from WASD and yaw, `Accelerate`/`AirAccelerate`,
ground speed clamp, gravity and position update. Walk (shift) and duck slow-downs, weapon max speeds and
`sv_autobunnyhopping` are supported. There is no map collision by default; pass `ground_height(x, y, z)` to land on terrain.

```python
import numpy as np
from cs2movesim import PlayerState, simulate, BUTTONS

state = PlayerState.create([[0, 0, 0]])                       # one player at the origin, standing still
buttons = np.zeros((64, 1, len(BUTTONS)), bool)
buttons[:, 0, BUTTONS.index("forward")] = True                 # hold W for one second
pos, vel = simulate(state, buttons, yaw=0.0, base_speed=215.0) # AK-47 max speed
print(np.hypot(*vel[-1, 0, :2]))                               # -> 215.0
```

Parameters (`MovementParams`) default to the CS2 server values (`sv_accelerate 5.5`, `sv_friction 5.2`,
`sv_stopspeed 80`, walk 0.52, duck 0.34, jump impulse 301.99, gravity 800, ...). `OPENCS2_FIT` is a preset fitted on
pro demos (see below). `WEAPON_SPEED` lists max speeds measured from the demos (AK-47 215, M4A1-S 225, AWP 200
unscoped, USP-S 240, grenades 245, knives 250, ...).

## Controller

`TrackingController` picks, every tick, the button set (no movement, 8 directions, each with or without walk) whose
one-tick simulated velocity is closest to the desired velocity: the plan's velocity for the next tick plus a
proportional pull towards the planned position. Aim needs no controller: view angles are an exact function of mouse
input, so the mouse delta is just the planned angle change.

## Benchmarks on OpenCS2

```
cs2movesim sample --n 400 --weapon ak47 --out ticks/   # download POV ticks.parquet files (about 180 KB each)
cs2movesim sample --n 300 --seed 1 --out ticks/
cs2movesim speeds ticks/                               # weapon max speeds
cs2movesim replay ticks/ [--preset opencs2]            # open-loop replay of human inputs
cs2movesim track  ticks/ [--mode sparse] [--noise 1]   # closed-loop tracking of human paths
```

Windows are cut where the simulator applies: alive, flat ground, no jump, not defusing or planting, not pushing into
an obstacle. Numbers below are from 700 de_dust2 POV rounds (400 AK-47 rounds and 300 others).

**Open-loop replay** (11648 windows): start from the recorded velocity, feed the human's recorded buttons and yaw
tick by tick (inputs of tick *t* show up in the velocity recorded at *t + 1*), compare horizontal velocity (u/s):

| after | 4 ticks | 8 ticks | 16 ticks | 32 ticks |
| --- | --- | --- | --- | --- |
| constant velocity, median | 16.6 | 29.8 | 41.3 | 57.6 |
| simulator, server defaults, median | 3.4 | 6.8 | 8.9 | 9.7 |
| simulator, `OPENCS2_FIT`, median | 3.8 | 6.0 | 6.7 | 7.4 |

The remaining tail (90th percentile about 40 u/s) is mostly wall contact, which is not simulated.

**Closed-loop tracking** (1219 windows of 2 s): the simulator starts from the human's state; every 4 ticks (one 16 Hz
model step) the controller receives a plan anchored at the *simulated* position (human's future displacement), and
chooses keys tick by tick. Compared with the human:

| plan | velocity error median / p90 (u/s) | drift after 2 s | speed while shooting (human 15.7) | shots fired >20 u/s faster than human | ticks to stop (human 4) |
| --- | --- | --- | --- | --- | --- |
| exact, one point per tick | 3.9 / 18.0 | 0.95 | 9.0 | 0.1% | 3 |
| exact, one point per 4 ticks, interpolated | 2.7 / 15.1 | 1.7 | 9.4 | 0.3% | 4 |
| shrunk 20% | 8.4 / 31.4 | 5.6 | 7.8 | 0% | 2 |
| shrunk 50% | 17.4 / 63.5 | 13.2 | 6.3 | 0% | 1 |
| noise 0.5 u per step | 10.4 / 21.6 | 4.0 | 16.0 | 3% | 3 |
| noise 1 u per step | 16.7 / 32.9 | 7.8 | 24.4 | 18% | 3 |
| noise 3 u per step | 35.4 / 68.1 | 20.2 | 39.3 | 40% | 5 |

Reading: with a correct plan, a 16 Hz path plus this controller reproduces pro movement, including counter-strafes.
Conservative (shrunk) plans are harmless; *jittery* plans are not: the next-step displacement error has to stay below
about 0.5 game units (pros move about 13 units per 16 Hz step at full speed) or shots start being fired while moving.

## Limitations

No map collision (optional `ground_height` only), no sub-tick input timing, no scoped movement speed, no ladders,
water or surf. Calibrated on de_dust2 only.

## License

MIT. OpenCS2 data is CC BY 4.0 (blanchon/opencs2_dataset).
