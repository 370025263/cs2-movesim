"""Command line: ``cs2movesim sample | speeds | replay | track``.

    cs2movesim sample --n 300 --out ticks/            # download POV ticks.parquet files from OpenCS2 (Hugging Face)
    cs2movesim speeds ticks/                          # weapon max speeds measured from the sample
    cs2movesim replay ticks/ [--preset opencs2]       # open-loop replay of human inputs vs recorded velocity
    cs2movesim track ticks/ [--mode sparse --noise 1] # closed-loop tracking of human paths by the controller
    cs2movesim recoil ticks/ --out curves.json        # median spray (recoil compensation) curves per weapon
"""
import argparse
import glob
import os
import time

import numpy as np

from .params import OPENCS2_FIT, MovementParams

INDEX = "index/pov_rounds.parquet"


def load_rounds(directory):
    from .opencs2 import load_ticks
    paths = sorted(glob.glob(os.path.join(directory, "**", "*.parquet"), recursive=True))
    paths = [p for p in paths if not p.endswith("pov_rounds.parquet")]
    if not paths:
        raise SystemExit(f"no ticks.parquet under {directory}")
    return [load_ticks(p) for p in paths]


def cmd_sample(args):
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor

    from .opencs2 import download
    index = pd.read_parquet(download(INDEX, args.out), columns=["map_name", "primary_weapon", "ticks_parquet_path"])
    index = index[index.map_name.astype(str) == args.map]
    if args.weapon:
        index = index[index.primary_weapon.astype(str) == args.weapon]
    paths = index.ticks_parquet_path.sample(min(args.n, len(index)), random_state=args.seed)
    paths = [p.split("@main/")[1] for p in paths]
    started = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda p: download(p, args.out), paths))
    print(f"downloaded {len(paths)} POV rounds of {args.map} into {args.out} in {time.time() - started:.0f}s")


def cmd_speeds(args):
    from .calibrate import estimate_weapon_speeds
    table = estimate_weapon_speeds(load_rounds(args.dir), min_samples=args.min_samples)
    for weapon, (speed, count) in sorted(table.items(), key=lambda kv: -kv[1][1]):
        print(f"{weapon:24s} {speed:6.1f}  ({count} ticks)")


def preset(name):
    return OPENCS2_FIT if name == "opencs2" else MovementParams()


def cmd_replay(args):
    from .calibrate import pack_windows, replay_errors
    windows = pack_windows(load_rounds(args.dir), args.horizon, stride=args.horizon * 3 // 4)
    hold = np.linalg.norm(windows["vel"][1:, :, :2] - windows["vel"][0][None, :, :2], axis=2)
    errors = replay_errors(windows, preset(args.preset), lag=args.lag)
    marks = [t for t in (4, 8, 16, 32, 64) if t <= args.horizon]
    print(f"{errors.shape[1]} windows of {args.horizon} ticks; horizontal velocity error (u/s) at ticks {marks}")
    print("  constant velocity  median", np.median(hold[[m - 1 for m in marks]], 1).round(1))
    print("  simulator          median", np.median(errors[[m - 1 for m in marks]], 1).round(1),
          " p90", np.percentile(errors[[m - 1 for m in marks]], 90, 1).round(1))


def cmd_track(args):
    from .bench import run
    from .calibrate import pack_windows
    windows = pack_windows(load_rounds(args.dir), args.horizon, min_speed=60.0)
    metrics, _ = run(windows, mode=args.mode, replan=args.replan, gain=args.gain, shrink=args.shrink, noise=args.noise,
                     params=preset(args.preset), seed=args.seed, allow_walk=not args.no_walk, estimate=args.estimate,
                     velocity_noise=args.velocity_noise, belief_params=OPENCS2_FIT if args.estimate != "true" else None)
    print(f"{windows['pos'].shape[1]} windows of {args.horizon} ticks, plan={args.mode}, replan every {args.replan} ticks")
    for key, value in metrics.items():
        print(f"  {key:20s} {value:.3f}" if isinstance(value, float) else f"  {key:20s} {value}")


def cmd_recoil(args):
    from .recoil import estimate_spray_curves, save_curves
    curves = estimate_spray_curves(load_rounds(args.dir), max_ticks=args.max_ticks, min_sprays=args.min_sprays)
    save_curves(curves, args.out)
    for weapon, curve in sorted(curves.items(), key=lambda kv: -kv[1]["sprays"]):
        at = [curve["pitch"][t] for t in (13, 26, 38) if t < len(curve["pitch"])]
        print(f"{weapon:16s} {curve['sprays']:5d} sprays, {len(curve['pitch']):3d} ticks, pitch pulled at 0.2/0.4/0.6 s: "
              + ", ".join(f"{v:.2f}" for v in at))
    print(f"saved {len(curves)} curves to {args.out}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cs2movesim", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sample", help="download POV ticks.parquet files from OpenCS2")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--map", default="de_dust2")
    p.add_argument("--weapon", default=None, help="only rounds with this primary weapon, e.g. ak47")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.set_defaults(func=cmd_sample)
    p = sub.add_parser("speeds", help="weapon max speeds measured from the sample")
    p.add_argument("dir")
    p.add_argument("--min-samples", type=int, default=20)
    p.set_defaults(func=cmd_speeds)
    p = sub.add_parser("replay", help="open-loop replay of human inputs")
    p.add_argument("dir")
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--lag", type=int, default=1)
    p.add_argument("--preset", choices=["server", "opencs2"], default="server")
    p.set_defaults(func=cmd_replay)
    p = sub.add_parser("track", help="closed-loop tracking of human paths")
    p.add_argument("dir")
    p.add_argument("--horizon", type=int, default=128)
    p.add_argument("--mode", choices=["dense", "sparse"], default="dense")
    p.add_argument("--replan", type=int, default=4)
    p.add_argument("--gain", type=float, default=8.0)
    p.add_argument("--shrink", type=float, default=0.0)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--no-walk", action="store_true")
    p.add_argument("--preset", choices=["server", "opencs2"], default="server")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--estimate", choices=["true", "noisy", "plan", "dead"], default="true",
                   help="where the controller's velocity comes from (see cs2movesim.bench)")
    p.add_argument("--velocity-noise", type=float, default=0.0)
    p.set_defaults(func=cmd_track)
    p = sub.add_parser("recoil", help="median spray compensation curves per weapon")
    p.add_argument("dir")
    p.add_argument("--out", required=True)
    p.add_argument("--max-ticks", type=int, default=192)
    p.add_argument("--min-sprays", type=int, default=30)
    p.set_defaults(func=cmd_recoil)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
