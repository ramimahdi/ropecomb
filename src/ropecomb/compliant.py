"""Compliant simulator: the output member is an elastic, pre-tensioned spring.

The rigid model's unilateral branch

    step = max(free_flight, rope_limited)

is replaced by a physical spring,

    T = k_rope * max(0, rope_end_dist - target_dist)

so the payload may briefly outrun the rope tip without the model switching
branches. The force becomes continuous, which is both more physical and far
better conditioned numerically.

Pre-tension matters and is not a detail. A member starting at bare equilibrium
must stretch from nothing as the stroke begins, which is a step input that rings
the spring. A real machine would be pre-tensioned toward the working load; pass
``pretension`` in newtons, normally the design force. The payload then sees the
design force from the first instant rather than ramping up from zero.

The compliant result is SENSITIVE to stiffness. Section 5.7 of the paper reports
peak-to-mean rising sharply as the member is stiffened, so a design should be
checked across a stiffness range rather than at a single nominal value. Use
:func:`stiffness_sweep`.
"""

from __future__ import annotations

import numpy as np

from .rigid import G_ACC

__all__ = ["simulate_compliant", "rope_stiffness", "stiffness_sweep"]

_KEYS = ("t", "gear", "heavy_speed", "target_speed", "heavy_dist", "target_dist",
         "rope_end_dist", "heavy_acc", "target_acc", "force_heavy",
         "force_target", "stretch", "slack", "energy")


def rope_stiffness(E=100e9, A=2.8e-6, L=17.0):
    """Axial stiffness k = E*A/L, N/m.

    Defaults describe the 2 mm UHMWPE member of the worked case: fibre modulus
    about 100 GPa, area 2.8e-6 m^2 from a 9.9 kN break at 3.5 GPa, 17 m long.
    That gives roughly 16,500 N/m, the nominal value used throughout the paper.
    """
    return E * A / L


def simulate_compliant(M, m, v0, ratio_fn, k_rope, *, pretension=None,
                       max_heavy_dist=None, damping=0.0, rope_mass=0.0,
                       max_time=0.5, max_target_acc=1e9, dt=2e-5,
                       max_steps=20_000_000):
    """Run a real array on a compliant, optionally pre-tensioned member.

    Parameters
    ----------
    k_rope : float
        Axial stiffness of the output member, N/m. See :func:`rope_stiffness`.
    pretension : float, optional
        Initial tension, N. Defaults to bare payload weight, which rings the
        spring; pass the design force for the pre-tensioned case reported in
        the paper.
    rope_mass : float
        Optionally lump the fast member's inertia onto the payload. This is the
        leading parasitic-mass term after sheave inertia.
    damping : float
        Viscous term on the stretch rate, N/(m/s). The published results are
        insensitive to it.
    """
    g = G_ACC
    m_eff = float(m) + float(rope_mass)

    gear = 0.0
    average_gear = 0.0
    heavy_speed = float(v0)
    target_speed = 0.0
    target_dist = 0.0
    heavy_dist = 0.0
    total_time = 0.0
    target_acc = 0.0

    F0 = m * g if pretension is None else float(pretension)
    rope_end_dist = F0 / k_rope
    tension = F0
    prev_stretch = rope_end_dist

    H = {key: [] for key in _KEYS}
    KE0 = 0.5 * M * heavy_speed ** 2
    steps, stop = 0, "max_time"

    while total_time < max_time and target_acc < max_target_acc:
        if steps >= max_steps:
            stop = "max_steps"
            break
        if max_heavy_dist is not None and heavy_dist >= max_heavy_dist:
            stop = "release"
            break
        steps += 1
        total_time += dt

        # --- source mass: reaction is ratio x member tension ---------------
        net_force_on_heavy = g * M - average_gear * tension
        heavy_acc = net_force_on_heavy / M

        new_heavy_speed = heavy_speed + heavy_acc * dt
        d_heavy = dt * (new_heavy_speed + heavy_speed) / 2.0
        heavy_dist += d_heavy
        heavy_speed = new_heavy_speed

        # --- ratio ----------------------------------------------------------
        new_gear = ratio_fn(heavy_dist)
        average_gear = (gear + new_gear) / 2.0
        gear = new_gear
        rope_end_dist += d_heavy * average_gear

        # --- payload on a spring: continuous, no branch ---------------------
        stretch = rope_end_dist - target_dist
        if stretch > 0.0:
            rate = (stretch - prev_stretch) / dt
            tension = k_rope * stretch + damping * rate
            if tension < 0.0:
                tension = 0.0
        else:
            tension = 0.0
        prev_stretch = stretch

        new_target_acc = tension / m_eff - g
        new_target_speed = target_speed + new_target_acc * dt
        target_dist += dt * (new_target_speed + target_speed) / 2.0
        target_speed = new_target_speed
        target_acc = new_target_acc

        ke = 0.5 * M * heavy_speed ** 2 + 0.5 * m_eff * target_speed ** 2
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
        H["energy"].append(ke + spring - M * g * heavy_dist)
    else:
        stop = "max_target_acc" if target_acc >= max_target_acc else "max_time"

    out = {key: np.asarray(v) for key, v in H.items()}
    n = len(out["t"])
    mean_force = (0.5 * m_eff * target_speed ** 2 / target_dist) if target_dist > 0 else 0.0
    out["summary"] = dict(
        stop_reason=stop, steps=steps, dt=dt, k_rope=float(k_rope), v0=float(v0),
        target_final_speed=float(target_speed),
        heavy_final_speed=float(heavy_speed),
        elapsed=float(total_time),
        heavy_travel=float(heavy_dist),
        target_travel=float(target_dist),
        final_gear=float(gear),
        peak_force_target=float(out["force_target"].max()) if n else 0.0,
        mean_force_target=float(mean_force),
        peak_stretch=float(out["stretch"].max()) if n else 0.0,
        slack_fraction=float(out["slack"].mean()) if n else 0.0,
        first_slack_time=float(out["t"][out["slack"]][0]) if out["slack"].any() else None,
        energy_drift=float(abs(out["energy"][-1] - KE0) / KE0) if n else float("nan"),
    )
    return out


def stiffness_sweep(M, m, v0, ratio_fn, k_values, *, pretension=None,
                    max_heavy_dist=None, dt=1e-5, **kw):
    """Peak-to-mean payload force across a range of member stiffnesses.

    The single most useful robustness check on a candidate design: a geometry
    whose peak-to-mean climbs steeply with stiffness is relying on a modelling
    assumption rather than on its geometry.
    """
    from .metrics import peak_to_mean

    rows = []
    for k in k_values:
        run = simulate_compliant(M, m, v0, ratio_fn, k, pretension=pretension,
                                 max_heavy_dist=max_heavy_dist, dt=dt, **kw)
        rows.append(dict(k_rope=float(k),
                         peak_to_mean=peak_to_mean(run["force_target"]),
                         exit=run["summary"]["target_final_speed"],
                         peak_stretch=run["summary"]["peak_stretch"]))
    return rows
