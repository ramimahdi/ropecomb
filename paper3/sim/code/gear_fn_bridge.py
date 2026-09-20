"""
Bridge between the fitter and your rigid-body simulator.

The fitter outputs span half-widths R_i and engagement offsets s_i.
Your simulator wants get_gear_fn(heavy_dec_dist). This converts one to the other.

    from gear_fn_bridge import make_gear_fn, load_case

    cfg  = load_case("A_1000to1_N8")
    gear = make_gear_fn(cfg["R"], cfg["s"], cfg["k"])
    result = CatabultRigidBodySimulation(
        heavy_weight=cfg["M"], target_weight=cfg["m"],
        heavy_height=cfg["v0"]**2/(2*9.8),      # height giving v0 at engagement
        max_time=0.5, max_target_acc=1e9,
        get_gear_fn=gear, time_unit=1e-5)

Cases in fitted_geometries.json:

    A_1000to1_N8         8 pins, unconstrained. Direct comparison with your existing run.
    B_1000to1_N6         6 pins. Fewer parts, wider spans.
    C_1000to1_N8_narrow  8 pins, spans capped at 0.5 m. Buildable comb (4.0 m) but
                         tracking floors out. Smoothest engagement (worst 2/R = 8.0).
    D_1000to1_N8_long    braking distance 1.5 m. Higher exit speed, 37% less force.
    E_100to1_N3          the gentle regime: 112 m/s at ~36 g, only 3 pins.
    F_10000to1_N20       your 10,000:1 case with enough pins (you used 13, floor is ~12).
"""

import json
import math
from pathlib import Path

HERE = Path(__file__).parent


def make_gear_fn(R, s, k=1.0):
    """
    Return get_gear_fn(d) giving the NET ratio (array x fixed stage) at
    source-mass displacement d.

    R : list of span half-widths (m)
    s : list of engagement offsets (m) -- displacement at which each pin first
        contacts the rope
    k : fixed second-stage ratio (pass 1.0 for the array ratio alone)
    """
    Rs = list(R)
    ss = list(s)

    def get_gear_fn(d):
        total = 0.0
        for Ri, si in zip(Rs, ss):
            Di = d - si
            if Di > 0.0:
                total += 2.0 * Di / math.sqrt(Ri * Ri + Di * Di)
        return k * total

    return get_gear_fn


def load_case(name, path=None):
    path = Path(path) if path else HERE / "fitted_geometries.json"
    with open(path) as f:
        cases = json.load(f)
    if name not in cases:
        raise KeyError(f"{name!r} not found. Available: {list(cases)}")
    return cases[name]


def gear_table(name, n=25):
    """Print the net ratio across the stroke -- sanity check before simulating."""
    c = load_case(name)
    g = make_gear_fn(c["R"], c["s"], c["k"])
    print(f"{name}:  N={c['N']}  k={c['k']}  brake={c['d_max']} m  "
          f"ideal final ratio={c['ideal_final_ratio']:.1f}:1")
    print(f"{'d (m)':>10}{'net ratio':>12}")
    for i in range(n + 1):
        d = c["d_max"] * i / n
        print(f"{d:>10.4f}{g(d):>12.2f}")


if __name__ == "__main__":
    import sys
    gear_table(sys.argv[1] if len(sys.argv) > 1 else "A_1000to1_N8")
