"""Reading OpenCS2 (https://huggingface.co/datasets/blanchon/opencs2_dataset) per-POV ``ticks.parquet`` files.

Needs the optional dependencies ``pandas`` and ``pyarrow`` (``pip install cs2-movesim[opencs2]``).
"""
import os
import urllib.request

import numpy as np

from .physics import BUTTONS

HF_URL = "https://huggingface.co/datasets/blanchon/opencs2_dataset/resolve/main/"
ACTION_TO_BUTTON = {"forward": "forward", "back": "back", "move_left": "left", "move_right": "right",
                    "speed": "walk", "duck": "duck", "jump": "jump"}
COLUMNS = ["tick", "active", "pitch", "yaw", "input_weapon", "is_alive", "x", "y", "z", "velocity_x", "velocity_y", "velocity_z"]


def download(relative_path, dest_dir, base_url=HF_URL):
    """Download one file of the dataset (e.g. ``rounds/match_id=.../ticks.parquet``) into ``dest_dir``; returns the local path."""
    dest = os.path.join(dest_dir, relative_path)
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(base_url + relative_path, dest + ".part")
        os.replace(dest + ".part", dest)
    return dest


def load_ticks(path):
    """One POV round -> dict of per-tick arrays on a contiguous tick axis (missing ticks forward-filled, ``present`` marks real rows):
    tick [T], buttons [T, 7] (order of :data:`cs2movesim.physics.BUTTONS`), attack [T], use [T], pitch, yaw [T], weapon [T] (str),
    alive [T], pos [T, 3], vel [T, 3], present [T]."""
    import pandas as pd

    table = pd.read_parquet(path, columns=COLUMNS).sort_values("tick")
    ticks = np.arange(int(table.tick.iloc[0]), int(table.tick.iloc[-1]) + 1)
    present = np.isin(ticks, table.tick.values)
    table = table.set_index("tick").reindex(ticks)
    table = table.astype({c: "object" for c in ("active", "input_weapon") if c in table}).ffill().infer_objects()
    buttons = np.zeros((len(ticks), len(BUTTONS)), dtype=bool)
    attack = np.zeros(len(ticks), dtype=bool)
    use = np.zeros(len(ticks), dtype=bool)
    for i, active in enumerate(table.active.values):
        if not isinstance(active, (list, tuple, np.ndarray)):
            continue
        for action in active:
            button = ACTION_TO_BUTTON.get(action)
            if button is not None:
                buttons[i, BUTTONS.index(button)] = True
            elif action == "attack":
                attack[i] = True
            elif action == "use":
                use[i] = True
    return {
        "tick": ticks, "buttons": buttons, "attack": attack, "use": use,
        "pitch": table.pitch.values.astype(np.float64),
        "yaw": np.degrees(np.unwrap(np.radians(table.yaw.values.astype(np.float64)))),
        "weapon": table.input_weapon.astype(str).to_numpy(dtype=str), "alive": table.is_alive.astype(bool).to_numpy(),
        "pos": table[["x", "y", "z"]].values.astype(np.float64),
        "vel": table[["velocity_x", "velocity_y", "velocity_z"]].values.astype(np.float64),
        "present": present,
    }
