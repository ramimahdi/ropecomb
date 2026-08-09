"""Rigid-body simulator: the tension member is inextensible and massless.

The model contains no friction, no rotational inertia, no member mass and no
aerodynamic drag. It is the baseline throughout the paper because it has no free
parameters. Integration is explicit and first order in force with trapezoidal
displacement updates; the force on the source at the start of each step uses the
ratio and reaction carried over from the previous step. That one-step lag is
deliberate and is what makes the scheme explicit.

Two entry points share one loop:

    simulate_rigid   - the ratio is read from a geometry, via net_ratio_fn
    simulate_ideal   - the ratio is SOLVED FOR at each step so that the payload
                       force stays uniform

Because they differ only in that one line, the ideal profile and any real array
are produced by identical code and are directly comparable. That removes a class
of error which is otherwise easy to make.

Note on convergence: the terminal traction-loss transient is NOT time-step
converged. Exit velocities and stroke durations are reliable; peak force
magnitudes and the precise divergence time are indicative only.
"""

from __future__ import annotations

import math

import numpy as np

G_ACC = 9.8

__all__ = ["simulate_rigid", "simulate_ideal", "G_ACC"]

_KEYS = ("t", "gear", "heavy_speed", "target_speed", "heavy_dist", "target_dist",
         "rope_end_dist", "heavy_acc", "target_acc", "force_heavy",
         "force_target", "slack", "energy")


def _integrate(M, m, v0, ratio_fn, ideal_force, max_time, max_target_acc,
               max_heavy_dist, min_heavy_speed, dt, max_steps):
    g = G_ACC

    gear = 0.0
    average_gear = 0.0
    heavy_acc = g
    heavy_speed = float(v0)

    net_force_on_heavy = heavy_acc * M
    target_acc = g
    target_speed = 0.0
    target_reaction_force = 0.0

    heavy_dist = 0.0
    target_dist = 0.0
    rope_end_dist = 0.0
    total_time = 0.0

    H = {key: [] for key in _KEYS}
    KE0 = 0.5 * M * heavy_speed ** 2
    steps = 0
    stop = "max_time"

    while total_time < max_time and target_acc < max_target_acc:
        if steps >= max_steps:
            stop = "max_steps"
            break
        if max_heavy_dist is not None and heavy_dist >= max_heavy_dist:
            stop = "release"
            break
        if min_heavy_speed is not None and heavy_speed <= min_heavy_speed:
            stop = "source reached target speed"
            break
        steps += 1
        total_time += dt

        # --- source mass -------------------------------------------------
        # Reaction is (ratio x payload tension), both carried over from the
        # previous step. Net force goes negative once reaction exceeds weight.
        net_force_on_heavy = (g * M) - (average_gear * target_reaction_force)
        heavy_acc = net_force_on_heavy / M

        new_heavy_speed = heavy_speed + heavy_acc * dt
        d_heavy = dt * (new_heavy_speed + heavy_speed) / 2.0
        heavy_dist += d_heavy
        heavy_speed = new_heavy_speed

        # --- ratio ---------------------------------------------------------
        if ideal_force is not None:
            # Invert the trapezoidal relation dy/dd for the ratio that sustains
            # a uniform payload force. This is the only line that differs from
            # a real array.
            new_gear = 2.0 * (0.5 * dt * (ideal_force * dt + 2.0 * target_speed)
                              / d_heavy) - gear
        else:
            new_gear = ratio_fn(heavy_dist)
        average_gear = (gear + new_gear) / 2.0
        gear = new_gear

        rope_end_dist += d_heavy * average_gear

        # --- payload under a unilateral constraint --------------------------
        # The member may pull but not push, so the payload advances by whichever
        # is greater: its free flight, or the advance the rope demands.
        free = (target_speed - 0.5 * g * dt) * dt
        taut = rope_end_dist - target_dist
        step = free if free > taut else taut
        slack = free > taut

        new_target_speed = 2.0 * (step / dt) - target_speed
        new_target_acc = (new_target_speed - target_speed) / dt

        target_speed = new_target_speed
        target_dist += step

        # --- reaction, averaged over the step, positive accelerations only ---
        a1 = (g + target_acc) if target_acc > 0 else 0.0
        a2 = (g + new_target_acc) if new_target_acc > 0 else 0.0
        target_reaction_force = m * (a1 + a2) / 2.0
        target_acc = new_target_acc

        # --- diagnostics (do not feed back into the integration) ------------
        ke = 0.5 * M * heavy_speed ** 2 + 0.5 * m * target_speed ** 2
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
        H["force_target"].append(target_reaction_force)
        H["slack"].append(slack)
        H["energy"].append(ke - M * g * heavy_dist)
    else:
        stop = "max_target_acc" if target_acc >= max_target_acc else "max_time"

    out = {key: np.asarray(v) for key, v in H.items()}
    n = len(out["t"])
    mean_force = (0.5 * m * target_speed ** 2 / target_dist) if target_dist > 0 else 0.0
    out["summary"] = dict(
        stop_reason=stop, steps=steps, dt=dt, v0=float(v0),
        target_final_speed=float(target_speed),
        heavy_final_speed=float(heavy_speed),
        elapsed=float(total_time),
        heavy_travel=float(heavy_dist),
        target_travel=float(target_dist),
        final_gear=float(gear),
        peak_force_target=float(out["force_target"].max()) if n else 0.0,
        mean_force_target=float(mean_force),
        peak_force_heavy=float(out["force_heavy"].max()) if n else 0.0,
        slack_fraction=float(out["slack"].mean()) if n else 0.0,
        first_slack_time=float(out["t"][out["slack"]][0]) if out["slack"].any() else None,
        energy_drift=float(abs(out["energy"][-1] - KE0) / KE0) if n else float("nan"),
    )
    return out


def simulate_rigid(M, m, v0, ratio_fn, *, max_heavy_dist=None,
                   min_heavy_speed=None, max_time=0.5, max_target_acc=3e4,
                   dt=1e-5, max_steps=20_000_000):
    """Run a real array on an inextensible member.

    Parameters
    ----------
    M, m : float
        Source and payload mass, kg.
    v0 : float
        Source speed entering the stroke, m/s.
    ratio_fn : callable
        g(d) -> net ratio at source displacement d. Build with
        :func:`ropecomb.geometry.net_ratio_fn`.
    max_heavy_dist : float, optional
        Release position. Normally the braking distance the array was fitted
        over. Without it the carriage runs past the end of the array, the ratio
        saturates and the payload coasts on a slack rope, which is not the
        apparatus being modelled.
    max_target_acc : float
        Abort if payload acceleration exceeds this, m/s^2. Guards against the
        terminal divergence running away.

    Returns
    -------
    dict of history arrays plus a ``summary`` dict.
    """
    return _integrate(M, m, v0, ratio_fn, None, max_time, max_target_acc,
                      max_heavy_dist, min_heavy_speed, dt, max_steps)


def simulate_ideal(M, m, v0, force, *, min_heavy_speed=None, max_heavy_dist=None,
                   max_time=0.6, max_target_acc=1e9, dt=1e-5,
                   max_steps=20_000_000):
    """Run the theoretical uniform-force profile through the same loop.

    ``force`` is the uniform payload force to be held, in newtons. The ratio
    required to deliver it is solved for at each step. Stop with
    ``min_heavy_speed`` to end the stroke when the source reaches a chosen speed.
    """
    return _integrate(M, m, v0, None, float(force), max_time, max_target_acc,
                      max_heavy_dist, min_heavy_speed, dt, max_steps)
