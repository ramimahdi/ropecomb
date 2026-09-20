"""
RopeComb array geometry fitter.

Fits engagement-member span widths R_i and engagement offsets s_i so that the
array's ratio profile tracks the ideal constant-force profile.

KEY IDEA
--------
Everything here happens in the DISPLACEMENT domain (carriage descent d), not the
time domain. In that domain both sides are smooth and closed-form-ish:

  array   :  G_array(d) = sum_i 2*D_i / sqrt(R_i^2 + D_i^2),  D_i = max(0, d - s_i)
  target  :  dy/dd = sqrt(2*F*y/m) / sqrt(v0^2 + 2*g*d - 2*F*y/M),  G* = dy/dd

No velocities, no traction state, no discrete events. The dynamics (sawtooth,
traction loss, divergence) are NOT modelled here -- run your existing rigid-body
simulator on the fitted geometry to check those. This replaces the SEARCH, not
the SIMULATION.

Usage:
    python ropecomb_fit.py

Requires: numpy, scipy, matplotlib
"""

import numpy as np
from scipy.optimize import least_squares, brentq
import matplotlib.pyplot as plt

G_ACC = 9.8


# ----------------------------------------------------------------------
# 1. Ideal ratio profile in the displacement domain
# ----------------------------------------------------------------------

def ideal_profile(M, m, v0, F, d_max, n=600):
    """
    Integrate payload displacement y(d) under constant payload force F.

    M     : source mass (kg)
    m     : payload mass (kg)
    v0    : source speed at start of stroke (m/s)
    F     : constant force on payload (N)
    d_max : source-mass braking distance (m)

    Returns (d, y, G_star, v_h, v_t).
    """
    a = np.sqrt(2.0 * F / m)          # v_t = a * sqrt(y)
    d = np.linspace(0.0, d_max, n)
    h = d[1] - d[0]
    y = np.zeros(n)

    def slope(dd, yy):
        yy = max(yy, 0.0)
        vh2 = v0**2 + 2.0 * G_ACC * dd - 2.0 * F * yy / M
        if vh2 <= 1e-12:
            return 0.0
        return a * np.sqrt(yy) / np.sqrt(vh2)

    # y = 0 is a degenerate fixed point of the ODE, so seed the first step from
    # the analytic small-d expansion:  dy/dd = a*sqrt(y)/v0  =>  y = a^2 d^2/(4 v0^2)
    y[1] = (a**2 / (4.0 * v0**2)) * h**2

    for i in range(1, n - 1):
        k1 = slope(d[i], y[i])
        k2 = slope(d[i] + h / 2, y[i] + h * k1 / 2)
        k3 = slope(d[i] + h / 2, y[i] + h * k2 / 2)
        k4 = slope(d[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    vh = np.sqrt(np.maximum(v0**2 + 2.0 * G_ACC * d - 2.0 * F * y / M, 0.0))
    vt = a * np.sqrt(np.maximum(y, 0.0))
    G_star = np.divide(vt, vh, out=np.zeros_like(vt), where=vh > 1e-9)
    return d, y, G_star, vh, vt


def solve_force(M, m, v0, d_max, vh_final):
    """
    Find the constant payload force F that leaves the source mass at vh_final
    after braking distance d_max.
    """
    def resid(F):
        _, _, _, vh, _ = ideal_profile(M, m, v0, F, d_max, n=400)
        return vh[-1] - vh_final

    lo, hi = 1.0, 10.0
    while resid(hi) > 0 and hi < 1e9:
        hi *= 2.0
    if resid(hi) > 0:
        raise RuntimeError("No force reaches the requested final source speed.")
    return brentq(resid, lo, hi, xtol=1e-2)


# ----------------------------------------------------------------------
# 2. Array model
# ----------------------------------------------------------------------

def array_ratio(d, R, s):
    """Array ratio G_array(d). d is (n,), R and s are (N,)."""
    D = np.maximum(d[:, None] - s[None, :], 0.0)
    return np.sum(2.0 * D / np.sqrt(R[None, :] ** 2 + D**2), axis=1)


def engagement_slope_jumps(R):
    """Slope discontinuity contributed at each engagement: dG/dd jumps by 2/R."""
    return 2.0 / R


# ----------------------------------------------------------------------
# 3. Fit
# ----------------------------------------------------------------------

def fit_array(d, G_star, N, k, d_max, smooth_weight=0.0, seed=0,
              R_bounds=(1e-3, 5.0), width_budget=None, width_weight=50.0,
              convex_weight=0.0, overshoot_weight=0.0, convex_last=0,
              convex_taper=0.0, resid_weight=None, slope_weight=0.0,
              init_span_range=(0.2, 0.8), init_jitter=(0.6, 1.4)):
    """
    Fit N pins (span half-widths R, engagement offsets s) so that
    k * G_array(d) ~ G_star(d).

    smooth_weight penalises 1/R_i, i.e. abrupt engagement. Raise it to trade
    tracking accuracy for a smoother force profile.

    width_budget constrains the TOTAL array width, 2*sum(R) <= width_budget,
    via a one-sided penalty. Note this is not the same as capping each R at
    width_budget/(2N): the total-width constraint still permits the wide-early,
    narrow-late profile the target curve wants, whereas a uniform per-span cap
    forbids it and degrades sharply as N grows.

    Returns (R, s, result, rms) with pins sorted by engagement order.
    """
    rng = np.random.default_rng(seed)
    scale = np.ptp(G_star) if np.ptp(G_star) > 0 else 1.0

    def unpack(p):
        return np.exp(p[:N]), p[N:]

    def resid(p):
        R, s = unpack(p)
        err = (k * array_ratio(d, R, s) - G_star) / scale
        if resid_weight is not None:
            err = err * resid_weight
        r = err
        if overshoot_weight > 0.0:
            # Exceeding the target means the rope end outruns what the payload
            # needs, which shows up as a tension spike. Undershooting merely
            # lags. Penalise the two asymmetrically.
            r = np.concatenate([r, overshoot_weight * np.maximum(0.0, err)])
        if slope_weight > 0.0:
            # Payload force depends on the SLOPE of the ratio, not its value:
            #   a_t = G' v_h^2 + G a_h
            # so an objective that matches G alone controls force only
            # indirectly. Match dG/dd as well.
            Ga = k * array_ratio(d, R, s)
            g1 = np.gradient(Ga, d)
            t1 = np.gradient(G_star, d)
            sc1 = np.abs(t1).max() + 1e-9
            r = np.concatenate([r, slope_weight * (g1 - t1) / sc1])
        if smooth_weight > 0.0:
            r = np.concatenate([r, smooth_weight * engagement_slope_jumps(R)])
        if width_budget is not None:
            over = max(0.0, 2.0 * float(np.sum(R)) - width_budget) / width_budget
            r = np.concatenate([r, [width_weight * over]])
        if convex_weight > 0.0:
            # Each engaged member is individually concave (f'' < 0), so the array
            # can only approximate the convex target through staggered onsets.
            # Penalise the concave excursions: where the array's curvature falls
            # below the target's, the ratio is decelerating when it should not be.
            Ga = k * array_ratio(d, R, s)
            g2 = np.gradient(np.gradient(Ga, d), d)
            t2 = np.gradient(np.gradient(G_star, d), d)
            # Weight by where the target actually demands curvature. The target
            # is near-straight over its first half (curvature two orders of
            # magnitude lower than the second), so concavity there is harmless;
            # an unweighted penalty is dominated by that region and is useless.
            w = np.maximum(t2, 0.0) / (np.abs(t2).max() + 1e-9)
            if convex_taper > 0.0:
                # Time is scarce late in the stroke: a curvature deficit early
                # can still be recovered, one near the end cannot be. Taper the
                # weight linearly to zero across the stroke.
                ramp = np.clip(1.0 - convex_taper * (d / d_max), 0.0, 1.0)
                w = w * ramp
            if convex_last > 0:
                # Concavity after the last (or n-th from last) engagement is
                # less damaging: no further member is waiting to take up load,
                # so a decelerating ratio there cannot desynchronise anything.
                cut = np.sort(s)[-convex_last]
                w = np.where(d < cut, w, 0.0)
            deficit = np.maximum(0.0, t2 - g2) * w
            r = np.concatenate([r, convex_weight * deficit / (np.abs(t2).max() + 1e-9)])
        return r

    lo = np.concatenate([np.full(N, np.log(R_bounds[0])), np.zeros(N)])
    hi = np.concatenate([np.full(N, np.log(R_bounds[1])), np.full(N, d_max)])

    # Seed inside the R bounds, otherwise least_squares rejects the guess.
    #
    # Span half-widths are drawn in two stages so that the TOTAL comb width
    # varies from seed to seed. Drawing every R_i independently would make the
    # total concentrate near its mean by the central limit theorem -- at N=16
    # the total would sit within a few percent of 0.5N every time, and wide
    # arrays would never be sampled at all. Instead a per-seed mean span w is
    # drawn first, then per-member jitter about it:
    #
    #     w        ~ U(init_span_range)      mean span for this seed, metres
    #     R_i      = 0.5 * w * U(jitter)     half-widths
    #     total    = 2 * sum(R_i)  ~  w * N
    #
    # With init_span_range = (0.2, 0.8) the expected starting comb is 0.5N m
    # and starting widths span roughly [0.2N, 0.8N].
    r_lo, r_hi = R_bounds
    w = rng.uniform(*init_span_range)
    jitter = rng.uniform(*init_jitter, size=N)
    R0 = np.clip(0.5 * w * jitter, r_lo * 1.001, r_hi * 0.999)
    p0 = np.concatenate([
        np.log(R0),
        np.sort(rng.uniform(0.0, 0.85 * d_max, N)),
    ])
    p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)

    sol = least_squares(resid, p0, bounds=(lo, hi), max_nfev=20000)

    R, s = unpack(sol.x)
    order = np.argsort(s)
    R, s = R[order], s[order]
    rms = float(np.sqrt(np.mean((k * array_ratio(d, R, s) - G_star) ** 2)))
    return R, s, sol, rms


def multistart_fit(d, G_star, N, k, d_max, n_starts=12, smooth_weight=0.0):
    """Fit from several random starts; return the best and all RMS values."""
    best, all_rms = None, []
    for seed in range(n_starts):
        R, s, sol, rms = fit_array(d, G_star, N, k, d_max,
                                   smooth_weight=smooth_weight, seed=seed)
        all_rms.append(rms)
        if best is None or rms < best[3]:
            best = (R, s, sol, rms)
    return best, np.array(all_rms)


def min_pins(G_final, k):
    """Hard floor: each pin contributes at most 2 to the array ratio."""
    return int(np.ceil(G_final / (2.0 * k)))


# ----------------------------------------------------------------------
# 4. Driver
# ----------------------------------------------------------------------

def main():
    # ---- case setup: edit these -------------------------------------
    M, m = 1000.0, 1.0      # source and payload mass (kg)
    v0 = 10.0               # source speed entering the stroke (m/s)
    d_max = 0.86            # source braking distance (m)
    vh_final = 5.85         # source speed at end of stroke (m/s)
    k = 7.0                 # fixed second-stage ratio
    smooth_weight = 0.0     # raise (try 0.02) to penalise abrupt engagement
    # -----------------------------------------------------------------

    F = solve_force(M, m, v0, d_max, vh_final)
    d, y, G_star, vh, vt = ideal_profile(M, m, v0, F, d_max)

    print(f"constant payload force      : {F:,.0f} N  ({F/m/G_ACC:,.0f} g)")
    print(f"payload exit velocity       : {vt[-1]:,.1f} m/s")
    print(f"payload travel              : {y[-1]:,.2f} m")
    print(f"source speed {v0:.1f} -> {vh[-1]:.2f} m/s")
    print(f"final ideal net ratio       : {G_star[-1]:,.1f}:1")
    print(f"hard floor on pin count     : N >= {min_pins(G_star[-1], k)}")
    print()

    floor = min_pins(G_star[-1], k)
    N_list = [floor, floor + 2, floor + 4, floor + 6, floor + 9, floor + 13]

    results, rms_curve = {}, []
    for N in N_list:
        (R, s, sol, rms), spread = multistart_fit(
            d, G_star, N, k, d_max, n_starts=10, smooth_weight=smooth_weight)
        results[N] = (R, s, rms, spread)
        rms_curve.append(rms)
        near = int(np.sum(spread < 1.5 * rms))
        print(f"N={N:3d}  rms={rms:8.3f}  "
              f"max slope jump={np.max(2.0/R):8.1f}  "
              f"{near}/{len(spread)} starts within 1.5x best")

    # ---- plots ------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    ax[0].plot(d, G_star, "k--", lw=2, label="ideal $G^*(d)$")
    for N in N_list[:4]:
        R, s, rms, _ = results[N]
        ax[0].plot(d, k * array_ratio(d, R, s), lw=1.2, label=f"N={N}")
    ax[0].set_xlabel("source displacement d (m)")
    ax[0].set_ylabel("net ratio")
    ax[0].legend(fontsize=8)
    ax[0].set_title("fitted vs ideal ratio profile")

    ax[1].plot(N_list, rms_curve, "o-")
    ax[1].set_xlabel("number of pins N")
    ax[1].set_ylabel("RMS ratio error")
    ax[1].set_title("pick N at the knee")
    ax[1].grid(alpha=0.3)

    N_best = N_list[2]
    R, s, rms, _ = results[N_best]
    ax[2].stem(s, R)
    ax[2].set_xlabel("engagement offset $s_i$ (m)")
    ax[2].set_ylabel("span half-width $R_i$ (m)")
    ax[2].set_title(f"fitted geometry, N={N_best}")

    plt.tight_layout()
    plt.savefig("ropecomb_fit.png", dpi=130)
    print("\nwrote ropecomb_fit.png")

    print(f"\ngeometry for N={N_best}:")
    print(f"{'pin':>4} {'s_i (m)':>10} {'R_i (m)':>10} {'2/R_i':>10}")
    for i, (si, Ri) in enumerate(zip(s, R), 1):
        print(f"{i:>4} {si:>10.4f} {Ri:>10.4f} {2.0/Ri:>10.1f}")


if __name__ == "__main__":
    main()
