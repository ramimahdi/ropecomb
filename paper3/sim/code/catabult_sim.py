"""
RopeComb rigid-body simulator — direct transcription of the paper's pseudocode.

The integration loop is deliberately faithful to the original, including:
  - net_force_on_heavy uses the PREVIOUS iteration's average_gear and
    target_reaction_force (one-step lag), before the gear is updated
  - trapezoidal update of heavy_dec_dist
  - the unilateral rope constraint via MAX(free-flight, rope-limited)
  - reaction force averaged over two steps, positive accelerations only

Nothing in the dynamics has been "improved". Diagnostics (energy audit, history)
are recorded alongside and do not feed back into the integration.

    from catabult_sim import simulate, plot_3x3, convergence_check
    from gear_fn_bridge import make_gear_fn, load_case

    cfg = load_case("A_1000to1_N8")
    r = simulate(cfg["M"], cfg["m"], 5.102, get_gear_fn=make_gear_fn(cfg["R"], cfg["s"], cfg["k"]))
    print(r["summary"])
    plot_3x3(r, "run_A.png", title="A_1000to1_N8")
"""

import math
import numpy as np

G = 9.8


# ----------------------------------------------------------------------
# Core loop — faithful to the pseudocode
# ----------------------------------------------------------------------

def simulate(heavy_weight, target_weight, heavy_height,
             get_gear_fn, max_time=0.5, max_target_acc=1e9, max_heavy_dist=None,
             min_heavy_speed=None, time_unit=1e-5, max_steps=20_000_000):
    """
    Returns a dict of history arrays plus a summary.
    heavy_height sets the entry speed: v0 = sqrt(2*g*heavy_height).

    get_gear_fn may be either:
      * a callable -- the ratio produced by a real array, called as
        fn(heavy_dec_dist_delta, heavy_dec_dist), or fn(heavy_dec_dist);
      * a number -- the IDEAL case. The value is the uniform force to be held
        on the payload, and the ratio needed to deliver it is solved for at each
        step instead of being read from a geometry. Everything else in the loop
        is unchanged, so the ideal and a real array are produced by identical
        code and are directly comparable.
    """
    ideal_force = None
    if isinstance(get_gear_fn, (int, float)):
        ideal_force = float(get_gear_fn)
    else:
        try:
            get_gear_fn(0.0, 0.0); _two_arg = True
        except TypeError:
            _two_arg = False
    g = G

    gear = 0.0
    heavy_acc = g
    heavy_speed = math.sqrt(2.0 * heavy_acc * heavy_height)
    v0 = heavy_speed

    net_force_on_heavy = heavy_acc * heavy_weight
    target_acc = g
    target_speed = 0.0
    target_reaction_force = 0.0

    heavy_dec_dist = 0.0
    target_acc_dist = 0.0
    target_rope_end_dist = 0.0
    heavy_dec_dist_delta = 0.0
    total_time = 0.0
    average_gear = 0.0

    H = {k: [] for k in ("t", "gear", "heavy_speed", "target_speed", "heavy_dist",
                         "target_dist", "rope_end_dist", "heavy_acc", "target_acc",
                         "force_heavy", "force_target", "slack", "energy")}

    KE0 = 0.5 * heavy_weight * v0 ** 2
    steps = 0
    stop = "max_time"

    while total_time < max_time and target_acc < max_target_acc:
        if steps >= max_steps:
            stop = "max_steps"
            break
        if max_heavy_dist is not None and heavy_dec_dist >= max_heavy_dist:
            stop = "end of array"
            break
        if min_heavy_speed is not None and heavy_speed <= min_heavy_speed:
            stop = "source at target speed"
            break
        steps += 1
        total_time += time_unit

        # --- heavy mass -------------------------------------------------
        net_force_on_heavy = (g * heavy_weight) - (average_gear * target_reaction_force)
        heavy_acc = net_force_on_heavy / heavy_weight

        new_heavy_speed = heavy_speed + heavy_acc * time_unit
        heavy_dec_dist_delta = time_unit * (new_heavy_speed + heavy_speed) / 2.0
        heavy_dec_dist += heavy_dec_dist_delta
        heavy_speed = new_heavy_speed

        # --- gear -------------------------------------------------------
        if ideal_force is not None:
            # solve for the ratio that sustains a uniform payload force
            new_gear = (2.0 * (0.5 * time_unit
                        * (ideal_force * time_unit + 2.0 * target_speed)
                        / heavy_dec_dist_delta) - gear)
        elif _two_arg:
            new_gear = get_gear_fn(heavy_dec_dist_delta, heavy_dec_dist)
        else:
            new_gear = get_gear_fn(heavy_dec_dist)
        average_gear = (gear + new_gear) / 2.0
        gear = new_gear

        target_rope_end_dist += heavy_dec_dist_delta * average_gear

        # --- target mass ------------------------------------------------
        free = (target_speed - 0.5 * g * time_unit) * time_unit
        taut = target_rope_end_dist - target_acc_dist
        target_acc_dist_delta = free if free > taut else taut
        slack = free > taut                      # rope not pulling this step

        new_target_speed = 2.0 * (target_acc_dist_delta / time_unit) - target_speed
        new_target_acc = (new_target_speed - target_speed) / time_unit

        target_speed = new_target_speed
        target_acc_dist += target_acc_dist_delta

        # --- reaction ---------------------------------------------------
        a1 = (g + target_acc) if target_acc > 0 else 0.0
        a2 = (g + new_target_acc) if new_target_acc > 0 else 0.0
        target_reaction_force = target_weight * (a1 + a2) / 2.0

        target_acc = new_target_acc

        # --- logging (does not affect the integration) -------------------
        ke_h = 0.5 * heavy_weight * heavy_speed ** 2
        ke_t = 0.5 * target_weight * target_speed ** 2
        pe = heavy_weight * g * heavy_dec_dist
        H["t"].append(total_time)
        H["gear"].append(gear)
        H["heavy_speed"].append(heavy_speed)
        H["target_speed"].append(target_speed)
        H["heavy_dist"].append(heavy_dec_dist)
        H["target_dist"].append(target_acc_dist)
        H["rope_end_dist"].append(target_rope_end_dist)
        H["heavy_acc"].append(heavy_acc)
        H["target_acc"].append(new_target_acc)
        H["force_heavy"].append(abs(net_force_on_heavy))
        H["force_target"].append(target_reaction_force)
        H["slack"].append(slack)
        H["energy"].append(ke_h + ke_t - pe)     # invariant if lossless
    else:
        stop = "max_target_acc" if target_acc >= max_target_acc else "max_time"

    out = {k: np.asarray(v) for k, v in H.items()}

    e = out["energy"]
    drift = float(abs(e[-1] - KE0) / KE0) if len(e) else float("nan")

    out["summary"] = dict(
        stop_reason=stop, steps=steps, time_unit=time_unit,
        v0=v0,
        target_final_speed=float(target_speed),
        heavy_final_speed=float(heavy_speed),
        elapsed=float(total_time),
        heavy_travel=float(heavy_dec_dist),
        target_travel=float(target_acc_dist),
        final_gear=float(gear),
        peak_force_target=float(out["force_target"].max()) if len(e) else 0.0,
        mean_force_target=float(
            0.5 * target_weight * target_speed ** 2 / target_acc_dist) if target_acc_dist > 0 else 0.0,
        peak_force_heavy=float(out["force_heavy"].max()) if len(e) else 0.0,
        slack_fraction=float(out["slack"].mean()) if len(e) else 0.0,
        first_slack_time=float(out["t"][out["slack"]][0]) if out["slack"].any() else None,
        energy_drift=drift,
    )
    return out


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------

def convergence_check(dts=(1e-4, 1e-5, 1e-6), **kw):
    """Is the divergence physical or numerical? Re-run at several time steps."""
    print(f"{'dt':>10} {'exit m/s':>10} {'elapsed ms':>11} {'peak F':>10} "
          f"{'1st slack ms':>13} {'E drift':>9} {'stop':>16}")
    print("-" * 84)
    rows = []
    for dt in dts:
        s = simulate(time_unit=dt, **kw)["summary"]
        fs = s["first_slack_time"]
        rows.append(s)
        print(f"{dt:>10.0e} {s['target_final_speed']:>10.1f} {s['elapsed']*1e3:>11.3f} "
              f"{s['peak_force_target']:>10,.0f} "
              f"{(fs*1e3 if fs else float('nan')):>13.3f} "
              f"{s['energy_drift']*100:>8.3f}% {s['stop_reason']:>16}")
    v = [r["target_final_speed"] for r in rows]
    spread = (max(v) - min(v)) / max(abs(min(v)), 1e-9) * 100
    print(f"\nexit-speed spread across time steps: {spread:.2f}%")
    print("  < ~1%  -> converged; the instability is physical")
    print("  > ~5%  -> time-step dependent; the instability is at least partly numerical")
    return rows


def build_ideal(M, m, v0, d_max, vh_final):
    """
    Theoretical optimal (uniform payload force) profile, mapped into the time
    domain so it can be overlaid on the simulation panels.
    """
    from ropecomb_fit import ideal_profile, solve_force

    F = solve_force(M, m, v0, d_max, vh_final)
    d, y, Gs, vh, vt = ideal_profile(M, m, v0, F, d_max, n=2000)

    # t(d) = integral of dd / v_h
    inv = 1.0 / np.maximum(vh, 1e-9)
    t = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(d))])

    return dict(
        t=t, F=F,
        heavy_dist=d, target_dist=y, gear=Gs,
        heavy_speed=vh, target_speed=vt,
        heavy_acc=(M * G - Gs * F) / M,
        target_acc=np.full_like(d, F / m - G),
        force_target=np.full_like(d, F),
        rope_end_dist=y,
    )


def _robust_ylim(ax, series, ref=None, headroom=4.0, pct=99.0):
    """
    Clip the y-axis on BOTH sides so a spike -- positive or negative -- does not
    flatten the rest of the curve. Whichever side is clipped gets a red label
    giving the true extreme. Returns True if either side was clipped.
    """
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return False

    lo_data, hi_data = float(s.min()), float(s.max())
    p_lo = float(np.percentile(s, 100.0 - pct))
    p_hi = float(np.percentile(s, pct))
    span = max(p_hi - p_lo, abs(p_hi), abs(p_lo), 1e-12)
    pad = 0.15 * span

    cap_hi, cap_lo = p_hi + pad, p_lo - pad
    if ref:
        # never clip tighter than a few multiples of the reference magnitude
        cap_hi = max(cap_hi, headroom * abs(ref))
        if lo_data < 0:
            cap_lo = min(cap_lo, -headroom * abs(ref))

    clip_hi = hi_data > cap_hi * (1.0 + 1e-9)
    clip_lo = lo_data < cap_lo * (1.0 + 1e-9) if cap_lo < 0 else lo_data < cap_lo
    if not (clip_hi or clip_lo):
        return False

    ax.set_ylim(cap_lo if clip_lo else lo_data - 0.05 * span,
                cap_hi if clip_hi else hi_data + 0.05 * span)

    if clip_hi:
        ax.text(0.98, 0.94, f"max {hi_data:,.0f}\n(off scale)", transform=ax.transAxes,
                fontsize=6.5, ha="right", va="top", color="crimson")
    if clip_lo:
        ax.text(0.98, 0.06, f"min {lo_data:,.0f}\n(off scale)", transform=ax.transAxes,
                fontsize=6.5, ha="right", va="bottom", color="crimson")
    return True


def plot_3x3(r, path, title="", ideal=None):
    """
    3x3 panel: distances / speeds / forces & accelerations.
    Solid black = simulation, dashed blue = theoretical optimal (if supplied).
    Spiking panels are y-clipped so the body of the curve stays readable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = r["t"] * 1e3
    ti = ideal["t"] * 1e3 if ideal is not None else None
    s = r["summary"]

    fig, ax = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle(title or "RopeComb simulation", fontsize=13)

    #  row, col, key,            title,                            clip?  ref for clipping
    panels = [
        (0, 0, "heavy_dist",    "a) heavy braking distance (m)",   False, None),
        (0, 1, "target_dist",   "b) payload travel (m)",           False, None),
        (0, 2, "gear",          "c) net gear ratio",               True,  s["final_gear"] / 2),
        (1, 0, "heavy_speed",   "d) heavy mass speed (m/s)",       False, None),
        (1, 1, "target_speed",  "e) payload speed (m/s)",          False, None),
        (1, 2, None,            "f) speed ratio vs commanded gear", True, s["final_gear"] / 2),
        (2, 0, "heavy_acc",     "g) heavy acceleration (m/s$^2$)", True,  None),
        (2, 1, "target_acc",    "h) payload acceleration (m/s$^2$)", True,
         s["mean_force_target"] / max(s.get("payload_mass", 1.0), 1e-9)),
        (2, 2, "force_target",  "i) force on payload (N)",         True,  s["mean_force_target"]),
    ]

    for i, j, key, ttl, clip, ref in panels:
        a = ax[i][j]

        if key is None:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(r["heavy_speed"] > 1e-9,
                                 r["target_speed"] / r["heavy_speed"], np.nan)
            a.plot(t, ratio, "k", lw=1, label="actual $v_t/v_h$")
            a.plot(t, r["gear"], color="0.55", lw=1, label="commanded gear")
            if ideal is not None:
                a.plot(ti, ideal["gear"], "b--", lw=1.1, label="ideal gear")
            a.legend(fontsize=6.5)
            if clip:
                _robust_ylim(a, np.nan_to_num(ratio, nan=0.0), ref)
        else:
            a.plot(t, r[key], "k", lw=1, label="simulated")
            if key == "target_dist":
                a.plot(t, r["rope_end_dist"], color="0.6", ls="-", lw=0.7, label="rope end")
            if ideal is not None and key in ideal:
                a.plot(ti, ideal[key], "b--", lw=1.1, label="ideal gear")
            if key in ("target_dist", "force_target"):
                a.legend(fontsize=6.5)
            if clip:
                use_ref = ref
                if use_ref is None:
                    use_ref = float(np.percentile(np.abs(r[key]), 90))
                _robust_ylim(a, r[key], use_ref)

        if r["slack"].any():
            a.axvline(r["t"][r["slack"]][0] * 1e3, color="orange", ls=":", lw=1,
                      label="_first slack")

        a.set_title(ttl, fontsize=9)
        a.set_xlabel("time (ms)", fontsize=8)
        a.tick_params(labelsize=7)
        a.grid(alpha=0.25)

    ax[2][2].axhline(s["mean_force_target"], color="g", ls="-.", lw=0.9)
    pk = s["peak_force_target"] / max(s["mean_force_target"], 1e-9)
    ax[2][2].text(0.02, 0.92, f"peak/mean = {pk:.2f}x", transform=ax[2][2].transAxes,
                  fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(path, dpi=130)
    plt.close(fig)
    return path


def report(r):
    s = r["summary"]
    print(f"  stop            : {s['stop_reason']}  ({s['steps']:,} steps @ dt={s['time_unit']:.0e})")
    print(f"  entry speed     : {s['v0']:.2f} m/s")
    print(f"  payload exit    : {s['target_final_speed']:,.1f} m/s")
    print(f"  heavy final     : {s['heavy_final_speed']:.2f} m/s")
    print(f"  elapsed         : {s['elapsed']*1e3:.2f} ms")
    print(f"  heavy travel    : {s['heavy_travel']:.3f} m")
    print(f"  payload travel  : {s['target_travel']:.2f} m")
    print(f"  final gear      : {s['final_gear']:.1f}:1")
    print(f"  peak/mean force : {s['peak_force_target']:,.0f} / {s['mean_force_target']:,.0f} N"
          f"  = {s['peak_force_target']/max(s['mean_force_target'],1e-9):.2f}x")
    print(f"  peak force heavy: {s['peak_force_heavy']:,.0f} N")
    fs = s["first_slack_time"]
    print(f"  first slack     : {fs*1e3:.2f} ms" if fs else "  first slack     : none")
    print(f"  slack fraction  : {s['slack_fraction']*100:.1f}% of steps")
    print(f"  energy drift    : {s['energy_drift']*100:.4f}%")


def ideal_benchmark(M, m, v0=10.0, vf=2.0, T=0.1, dt=1e-5):
    """
    External performance benchmark, after the convention of the original paper:
    a uniform payload force sustained for T seconds, sized so the source mass
    falls from v0 to vf -- i.e. near-complete energy extraction.

    Unlike a profile derived from a design's own endpoint, this does not depend
    on the apparatus and a design can therefore fall short of it. Integrated in
    the time domain by the recurrence used in the original work.
    """
    from scipy.optimize import brentq

    def run(F):
        n = int(T / dt) + 1
        t = np.zeros(n); vh = np.zeros(n); vt = np.zeros(n)
        Gr = np.zeros(n); d = np.zeros(n); y = np.zeros(n)
        vh[0] = v0; gear = 0.0
        for i in range(1, n):
            t[i] = t[i-1] + dt
            vh[i] = vh[i-1] + ((M*G - gear*F)/M)*dt
            if vh[i] <= 1e-6:                 # source stalled: F is too large
                vh[i:] = -1.0
                return t, vh, vt, Gr, d, y
            dd = dt*(vh[i]+vh[i-1])/2.0; d[i] = d[i-1] + dd
            vt[i] = vt[i-1] + (F/m)*dt
            dy = dt*(vt[i]+vt[i-1])/2.0; y[i] = y[i-1] + dy
            gear = 2.0*(dy/dd if dd > 1e-15 else 0.0) - gear
            Gr[i] = gear
        return t, vh, vt, Gr, d, y

    F = brentq(lambda f: run(f)[1][-1] - vf, 1e2, 5e6, xtol=1.0)
    t, vh, vt, Gr, d, y = run(F)
    return dict(t=t, F=F, heavy_speed=vh, target_speed=vt, gear=Gr,
                heavy_dist=d, target_dist=y, rope_end_dist=y,
                heavy_acc=(M*G - Gr*F)/M,
                target_acc=np.full_like(t, F/m - G),
                force_target=np.full_like(t, F),
                force_heavy=Gr*F)


def ideal_run(M, m, v0, F, vh_stop=2.0, dt=1e-5, max_t=0.6):
    """
    The ideal case, integrated by the reverse recurrence of the original work:
    a uniform payload force F is imposed, and at each step the ratio required to
    deliver it is computed from the trapezoidal relation while the source mass
    responds to the reaction. Halts when the source reaches vh_stop.

    Returns the same channels as simulate(), so the two can be plotted together.
    """
    n = int(max_t / dt) + 1
    T=[0.0]; VH=[v0]; VT=[0.0]; GG=[0.0]; D=[0.0]; Y=[0.0]; FH=[M*G]; FT=[0.0]
    vh, vt, gear, d, y, t = v0, 0.0, 0.0, 0.0, 0.0, 0.0
    for _ in range(n):
        net = M*G - gear*F
        vh_new = vh + (net/M)*dt
        if vh_new <= vh_stop:
            break
        dd = dt*(vh_new+vh)/2.0
        vt_new = vt + (F/m)*dt
        dy = dt*(vt_new+vt)/2.0
        gear = 2.0*(dy/dd) - gear
        vh, vt, d, y, t = vh_new, vt_new, d+dd, y+dy, t+dt
        T.append(t); VH.append(vh); VT.append(vt); GG.append(gear)
        D.append(d); Y.append(y); FH.append(abs(M*G - gear*F)); FT.append(F)
    a = np.asarray
    return dict(t=a(T), F=F, heavy_speed=a(VH), target_speed=a(VT), gear=a(GG),
                heavy_dist=a(D), target_dist=a(Y), rope_end_dist=a(Y),
                heavy_acc=(M*G - a(GG)*F)/M,
                target_acc=np.full(len(T), F/m - G),
                force_heavy=a(FH), force_target=a(FT))
