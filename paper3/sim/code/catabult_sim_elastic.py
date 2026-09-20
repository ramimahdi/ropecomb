"""
Elastic variant of the RopeComb simulator.

Identical to catabult_sim.simulate except that the rigid unilateral constraint

    target_acc_dist_delta = MAX(free_flight, rope_limited)

is replaced by a physical spring:

    T = k * max(0, rope_end_dist - target_dist)          k = E*A/L

The rope can stretch, so the payload is free to briefly outrun the rope tip
without the model having to switch branches. The force becomes continuous,
which is both more physical and much better conditioned numerically.

Stiffness for the worked case (Dyneema R3, 2 mm, ~17 m):
    E ~ 100 GPa (fibre), A ~ 2.8e-6 m^2 (from 9.9 kN break at 3.5 GPa)
    k = E*A/L ~ 16,700 N/m
    -> ~0.24 m stretch at 4 kN (1.4% strain), ~20-38 Hz depending on gear

catabult_sim.py is left untouched as the faithful transcription of the
published pseudocode. Report both.
"""

import math
import numpy as np

G = 9.8


def rope_stiffness(E=100e9, A=2.8e-6, L=17.0):
    """k = E*A/L. Defaults are the 2 mm Dyneema R3 of the worked case."""
    return E * A / L


def simulate_elastic(heavy_weight, target_weight, heavy_height, get_gear_fn,
                     k_rope, max_time=0.5, max_target_acc=1e9, max_heavy_dist=None,
                     time_unit=1e-5, rope_mass=0.0, pretension=None,
                     damping=0.0, max_steps=20_000_000):
    """
    Same signature as catabult_sim.simulate plus k_rope (N/m).

    max_heavy_dist models the release mechanism firing at a set carriage
    displacement -- normally the braking distance the array was fitted over.
    Without it the carriage runs on past the end of the array, the gear ratio
    saturates, and the payload coasts on a slack rope, which is not the
    apparatus being modelled.

    rope_mass (kg) optionally lumps the fast rope's inertia onto the payload,
    which is the leading parasitic-mass term.
    """
    g = G
    m_eff = target_weight + rope_mass

    gear = 0.0
    average_gear = 0.0
    heavy_speed = math.sqrt(2.0 * g * heavy_height)
    v0 = heavy_speed

    target_speed = 0.0
    target_dist = 0.0
    heavy_dist = 0.0
    total_time = 0.0

    # Initial rope extension. Default is bare equilibrium (supports the payload
    # weight only), which means the rope must stretch from nothing as the stroke
    # begins -- a step input that rings the spring. A real machine would be
    # pre-tensioned toward the working load; pass pretension in newtons.
    F0 = target_weight * g if pretension is None else float(pretension)
    rope_end_dist = F0 / k_rope
    tension = F0
    target_acc = 0.0
    prev_stretch = rope_end_dist

    H = {key: [] for key in ("t", "gear", "heavy_speed", "target_speed", "heavy_dist",
                             "target_dist", "rope_end_dist", "heavy_acc", "target_acc",
                             "force_heavy", "force_target", "stretch", "slack", "energy")}

    KE0 = 0.5 * heavy_weight * v0 ** 2
    steps, stop = 0, "max_time"

    while total_time < max_time and target_acc < max_target_acc:
        if steps >= max_steps:
            stop = "max_steps"
            break
        if max_heavy_dist is not None and heavy_dist >= max_heavy_dist:
            stop = "release"
            break
        steps += 1
        total_time += time_unit

        # --- heavy mass: reaction is gear x rope tension -----------------
        net_force_on_heavy = g * heavy_weight - average_gear * tension
        heavy_acc = net_force_on_heavy / heavy_weight

        new_heavy_speed = heavy_speed + heavy_acc * time_unit
        d_heavy = time_unit * (new_heavy_speed + heavy_speed) / 2.0
        heavy_dist += d_heavy
        heavy_speed = new_heavy_speed

        # --- gear --------------------------------------------------------
        new_gear = get_gear_fn(heavy_dist)
        average_gear = (gear + new_gear) / 2.0
        gear = new_gear

        rope_end_dist += d_heavy * average_gear

        # --- payload on a spring (continuous, no branch) ------------------
        stretch = rope_end_dist - target_dist
        if stretch > 0.0:
            rate = (stretch - prev_stretch) / time_unit
            tension = k_rope * stretch + damping * rate
            if tension < 0.0:
                tension = 0.0
        else:
            tension = 0.0
        prev_stretch = stretch

        new_target_acc = tension / m_eff - g
        new_target_speed = target_speed + new_target_acc * time_unit
        target_dist += time_unit * (new_target_speed + target_speed) / 2.0
        target_speed = new_target_speed
        target_acc = new_target_acc

        # --- logging ------------------------------------------------------
        ke = 0.5 * heavy_weight * heavy_speed ** 2 + 0.5 * m_eff * target_speed ** 2
        spring = 0.5 * k_rope * max(stretch, 0.0) ** 2
        H["t"].append(total_time)
        H["gear"].append(gear)
        H["heavy_speed"].append(heavy_speed)
        H["target_speed"].append(target_speed)
        H["heavy_dist"].append(heavy_dist)
        H["target_dist"].append(target_dist)
        H["rope_end_dist"].append(rope_end_dist)
        H["heavy_acc"].append(heavy_acc)
        H["target_acc"].append(new_target_acc)
        H["force_heavy"].append(abs(net_force_on_heavy))
        H["force_target"].append(tension)
        H["stretch"].append(stretch)
        H["slack"].append(stretch <= 0.0)
        H["energy"].append(ke + spring - heavy_weight * g * heavy_dist)
    else:
        stop = "max_target_acc" if target_acc >= max_target_acc else "max_time"

    out = {key: np.asarray(v) for key, v in H.items()}
    e = out["energy"]
    n = len(e)

    mean_force = (0.5 * m_eff * target_speed ** 2 / target_dist) if target_dist > 0 else 0.0
    out["summary"] = dict(
        stop_reason=stop, steps=steps, time_unit=time_unit, k_rope=k_rope,
        v0=v0,
        target_final_speed=float(target_speed),
        heavy_final_speed=float(heavy_speed),
        elapsed=float(total_time),
        heavy_travel=float(heavy_dist),
        target_travel=float(target_dist),
        final_gear=float(gear),
        peak_force_target=float(out["force_target"].max()) if n else 0.0,
        mean_force_target=float(mean_force),
        peak_over_mean=float(out["force_target"].max() / mean_force) if (n and mean_force > 0) else float("nan"),
        peak_stretch=float(out["stretch"].max()) if n else 0.0,
        slack_fraction=float(out["slack"].mean()) if n else 0.0,
        first_slack_time=float(out["t"][out["slack"]][0]) if out["slack"].any() else None,
        energy_drift=float(abs(e[-1] - KE0) / KE0) if n else float("nan"),
    )
    return out


def stiffness_sweep(heavy_weight, target_weight, heavy_height, get_gear_fn,
                    ks, time_unit=1e-5, max_time=0.5, max_target_acc=1e9,
                    max_heavy_dist=None):
    """Sweep rope stiffness; report where the instability disappears."""
    print(f"{'k (N/m)':>12} {'stretch':>9} {'exit m/s':>10} {'elapsed ms':>11} "
          f"{'peak/mean':>10} {'slack %':>8} {'E drift':>9}")
    print("-" * 78)
    rows = []
    for k in ks:
        s = simulate_elastic(heavy_weight, target_weight, heavy_height, get_gear_fn,
                             k_rope=k, time_unit=time_unit, max_time=max_time,
                             max_target_acc=max_target_acc,
                             max_heavy_dist=max_heavy_dist)["summary"]
        rows.append(s)
        print(f"{k:>12,.0f} {s['peak_stretch']:>8.3f}m {s['target_final_speed']:>10.1f} "
              f"{s['elapsed']*1e3:>11.2f} {s['peak_over_mean']:>9.2f}x "
              f"{s['slack_fraction']*100:>7.1f}% {s['energy_drift']*100:>8.3f}%")
    return rows
