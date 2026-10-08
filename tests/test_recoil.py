import numpy as np

from cs2movesim.recoil import SprayCompensator, estimate_spray_curves, feedforward, spray_ticks


def fake_round(n_sprays=40, length=40, weapon="ak47"):
    """Sprays where the human pulls down 0.2 degrees per tick while firing, separated by 30 idle ticks."""
    attack, pitch, yaw, names = [], [], [], []
    angle = 0.0
    for _ in range(n_sprays):
        for t in range(length):
            attack.append(True)
            angle += 0.2
            pitch.append(angle)
            yaw.append(0.0)
            names.append(weapon)
        for _ in range(30):
            attack.append(False)
            pitch.append(angle)
            yaw.append(0.0)
            names.append(weapon)
    n = len(attack)
    return {"attack": np.array(attack), "pitch": np.array(pitch), "yaw": np.array(yaw), "weapon": np.array(names),
            "alive": np.ones(n, bool)}


def test_spray_ticks_restart_on_release_and_weapon_change():
    tau = spray_ticks([1, 1, 1, 0, 1, 1], np.array(["ak47", "ak47", "ak47", "ak47", "ak47", "m4a1"]))
    assert tau.tolist() == [0, 1, 2, -1, 0, 0]


def test_curve_recovers_the_pull_and_feedforward_cancels_it():
    d = fake_round()
    curves = estimate_spray_curves([d], min_sprays=30)
    assert "ak47" in curves
    assert np.allclose(np.diff(curves["ak47"]["pitch"]), 0.2)
    increments, total = feedforward(d["attack"], d["weapon"], curves, d["alive"])
    residual = d["pitch"] - d["pitch"][0] - total[:, 0]
    assert np.abs(np.diff(residual)[d["attack"][1:] & d["attack"][:-1]]).max() < 1e-9


def test_grenades_are_not_compensated():
    assert estimate_spray_curves([fake_round(weapon="flashbang")], min_sprays=30) == {}


def test_compensator_matches_feedforward():
    d = fake_round(n_sprays=31)
    curves = estimate_spray_curves([d], min_sprays=30)
    comp = SprayCompensator(curves)
    online = np.array([comp.step(a, w) for a, w in zip(d["attack"], d["weapon"])])
    offline, _ = feedforward(d["attack"], d["weapon"], curves, d["alive"])
    assert np.allclose(online, offline)
