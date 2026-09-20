"""
Per-contact friction extension of the RopeComb integrators.

The kinematics are unchanged: the rope tip advances by G_net(d) dd exactly as in
catabult_sim.py / catabult_sim_elastic.py. What changes is the REACTION on the
carriage, which is computed from a second function react_fn(d) in place of the
kinematic ratio:

    reaction on the source = react_fn(d) * F_payload(t)

For a lossless machine react_fn == gear_fn and the loops reduce to the originals.

Contact model (per-contact efficiency eta, tension steps by 1/eta across every
sheave in the direction opposite to power flow):

  * Output member, from the payload end: segments s_0, s_1, ..., s_p with
    T(s_r) = F eta^{-r}. Sheaves r = 1..p alternate between the two counter-moving
    blocks starting with block 2 (the block driven by the trailing, high-fall
    array), as drawn in Figure 6(b) of the manuscript; the dead end goes to the
    block that keeps n_1 = floor(k/2), n_2 = ceil(k/2). The pull on block j is
    tau_j F, tau_j = sum over segments touching block j of eta^{-r}.
    For a one-line stage (single block, k falls): tau = sum_{r=0}^{k-1} eta^{-r}.

  * Tension member j, from its block back to its anchor on the guide: the entry
    sheave, then, per engagement member, a fixed support and the member itself.
    Members are numbered in engagement order, member 1 nearest the anchor
    (the design drawings), member N nearest the stage. The block-side segment of
    member i sits 2(N - i) + 2 contacts from the block and its anchor-side
    segment one more, so

        T_i^b = tau_j F eta^{-(2(N-i)+2)},   T_i^a = T_i^b / eta

    and the reaction member i applies to the carriage is cos(theta_i) (T_i^b + T_i^a),
    with 2 cos(theta_i) = 2 D_i / sqrt(R_i^2 + D_i^2) the member's term in G_j.

  Lossless limit: tau_j = n_j, all T = n_j F, reaction = n_j F G_j. Exact.

Lateral load with friction = max_d |rho_1(d) - rho_2(d)| / max_d (rho_1 + rho_2),
rho_j = reaction of array j per unit payload force.
"""
import math
import numpy as np

G = 9.8


# ---------------------------------------------------------------- contact model
def stage_factors(k, eta, two_line=True):
    """Return (tau_1, tau_2) for the two-line counter-moving stage, or (tau,) for one line."""
    if not two_line:
        return (sum(eta ** (-r) for r in range(k)),)
    p = (k - 1) // 2
    n1, n2 = k // 2, (k + 1) // 2
    seg1, seg2 = set(), set()
    for r in range(1, p + 1):                 # sheave r joins segments r-1 and r
        (seg2 if r % 2 == 1 else seg1).update((r - 1, r))
    dead_block = 1 if len(seg1) < n1 else 2   # dead end goes where a segment is missing
    (seg1 if dead_block == 1 else seg2).add(p)
    assert len(seg1) == n1 and len(seg2) == n2, (len(seg1), len(seg2), n1, n2)
    tau1 = sum(eta ** (-r) for r in seg1)
    tau2 = sum(eta ** (-r) for r in seg2)
    return (tau1, tau2)


def line_reaction_fn(R, s, tau, eta):
    """rho(d) = reaction of one array per unit payload force, with per-contact losses."""
    R = np.asarray(R, float); s = np.asarray(s, float); N = len(R)
    i = np.arange(1, N + 1)
    fb = tau * eta ** (-(2.0 * (N - i) + 2.0))
    fa = fb / eta
    w = fb + fa                               # (T_i^b + T_i^a)/F
    def rho(d):
        D = np.maximum(0.0, d - s)
        c = D / np.sqrt(R * R + D * D)        # cos(theta_i)
        return float(np.sum(w * c))
    return rho


def kinematic_fn(lines):
    """lines: list of (R, s, n_j). G_net(d) = sum n_j G_j(d)."""
    L = [(np.asarray(R, float), np.asarray(s, float), float(n)) for R, s, n in lines]
    def g(d):
        t = 0.0
        for R, s, n in L:
            D = np.maximum(0.0, d - s)
            t += n * float(np.sum(2.0 * D / np.sqrt(R * R + D * D)))
        return t
    return g


def make_config(lines, k, eta, two_line):
    """
    lines: for two_line, [(R1, s1), (R2, s2)] (may be the same geometry twice);
           for one line, [(R, s)].
    Returns gear_fn, react_fn, per-line rho functions, fall counts, taus.
    """
    taus = stage_factors(k, eta, two_line)
    if two_line:
        n = (k // 2, (k + 1) // 2)
        rhos = [line_reaction_fn(R, s, t, eta) for (R, s), t in zip(lines, taus)]
        gear = kinematic_fn([(R, s, nj) for (R, s), nj in zip(lines, n)])
    else:
        n = (k,)
        rhos = [line_reaction_fn(lines[0][0], lines[0][1], taus[0], eta)]
        gear = kinematic_fn([(lines[0][0], lines[0][1], k)])
    def react(d):
        return sum(r(d) for r in rhos)
    return gear, react, rhos, n, taus


def lateral_load(rhos, d_max, n=4001):
    d = np.linspace(0.0, d_max, n)
    r1 = np.array([rhos[0](x) for x in d]); r2 = np.array([rhos[1](x) for x in d])
    return float(np.max(np.abs(r1 - r2)) / np.max(r1 + r2)), float(np.max(r1 + r2))


# ------------------------------------------------------------ integrators (with react_fn)
def simulate_rigid(M, m, h, gear_fn, react_fn, max_time=0.5, max_target_acc=1e9,
                   max_heavy_dist=None, time_unit=1e-5, max_steps=20_000_000):
    g = G
    gear = 0.0; react = 0.0
    heavy_speed = math.sqrt(2.0 * g * h); v0 = heavy_speed
    target_acc = g; target_speed = 0.0; target_reaction_force = 0.0
    heavy_dec_dist = 0.0; target_acc_dist = 0.0; target_rope_end_dist = 0.0
    total_time = 0.0; average_gear = 0.0; average_react = 0.0
    F_hist, t_hist, slack_hist, hs_hist = [], [], [], []
    steps = 0; stop = "max_time"
    while total_time < max_time and target_acc < max_target_acc:
        if steps >= max_steps: stop = "max_steps"; break
        if max_heavy_dist is not None and heavy_dec_dist >= max_heavy_dist: stop = "end of array"; break
        if heavy_speed <= 0.0: stop = "source stalled"; break
        steps += 1; total_time += time_unit
        net = g * M - average_react * target_reaction_force
        heavy_acc = net / M
        new_hs = heavy_speed + heavy_acc * time_unit
        dd = time_unit * (new_hs + heavy_speed) / 2.0
        heavy_dec_dist += dd; heavy_speed = new_hs
        new_gear = gear_fn(heavy_dec_dist); average_gear = (gear + new_gear) / 2.0; gear = new_gear
        new_react = react_fn(heavy_dec_dist); average_react = (react + new_react) / 2.0; react = new_react
        target_rope_end_dist += dd * average_gear
        free = (target_speed - 0.5 * g * time_unit) * time_unit
        taut = target_rope_end_dist - target_acc_dist
        delta = free if free > taut else taut
        slack = free > taut
        new_ts = 2.0 * (delta / time_unit) - target_speed
        new_ta = (new_ts - target_speed) / time_unit
        target_speed = new_ts; target_acc_dist += delta
        a1 = (g + target_acc) if target_acc > 0 else 0.0
        a2 = (g + new_ta) if new_ta > 0 else 0.0
        target_reaction_force = m * (a1 + a2) / 2.0
        target_acc = new_ta
        F_hist.append(target_reaction_force); t_hist.append(total_time); slack_hist.append(slack); hs_hist.append(heavy_speed)
    else:
        stop = "max_target_acc" if target_acc >= max_target_acc else "max_time"
    return dict(stop=stop, t=np.array(t_hist), F=np.array(F_hist), slack=np.array(slack_hist),
                v_exit=float(target_speed), vh_end=float(heavy_speed), d_end=float(heavy_dec_dist),
                y_end=float(target_acc_dist), elapsed=float(total_time))


def simulate_compliant(M, m, h, gear_fn, react_fn, k_rope, pretension, max_time=0.5,
                       max_target_acc=1e9, max_heavy_dist=None, time_unit=2e-5, max_steps=20_000_000):
    g = G
    gear = 0.0; react = 0.0; average_gear = 0.0; average_react = 0.0
    heavy_speed = math.sqrt(2.0 * g * h)
    target_speed = 0.0; target_dist = 0.0; heavy_dist = 0.0; total_time = 0.0
    F0 = float(pretension); rope_end_dist = F0 / k_rope; tension = F0; target_acc = 0.0
    F_hist, t_hist, hs_hist, dh_hist = [], [], [], []
    steps = 0; stop = "max_time"
    while total_time < max_time and target_acc < max_target_acc:
        if steps >= max_steps: stop = "max_steps"; break
        if max_heavy_dist is not None and heavy_dist >= max_heavy_dist: stop = "release"; break
        if heavy_speed <= 0.0: stop = "source stalled"; break
        steps += 1; total_time += time_unit
        net = g * M - average_react * tension
        heavy_acc = net / M
        new_hs = heavy_speed + heavy_acc * time_unit
        dh = time_unit * (new_hs + heavy_speed) / 2.0
        heavy_dist += dh; heavy_speed = new_hs
        new_gear = gear_fn(heavy_dist); average_gear = (gear + new_gear) / 2.0; gear = new_gear
        new_react = react_fn(heavy_dist); average_react = (react + new_react) / 2.0; react = new_react
        rope_end_dist += dh * average_gear
        stretch = rope_end_dist - target_dist
        tension = k_rope * stretch if stretch > 0.0 else 0.0
        new_ta = tension / m - g
        new_ts = target_speed + new_ta * time_unit
        target_dist += time_unit * (new_ts + target_speed) / 2.0
        target_speed = new_ts; target_acc = new_ta
        F_hist.append(tension); t_hist.append(total_time); hs_hist.append(heavy_speed); dh_hist.append(heavy_dist)
    else:
        stop = "max_target_acc" if target_acc >= max_target_acc else "max_time"
    return dict(stop=stop, t=np.array(t_hist), F=np.array(F_hist), hs=np.array(hs_hist), d=np.array(dh_hist),
                v_exit=float(target_speed), vh_end=float(heavy_speed), d_end=float(heavy_dist),
                y_end=float(target_dist), elapsed=float(total_time))


def evaluate(gear_fn, react_fn, M, m, h, F, d_max, k_rope=16471.0):
    """Same conventions as paper2/refit_16471.py: rigid dt 1e-5 with max_target_acc 3e4; compliant dt 2e-5,
    pre-tensioned to F, released at d_max; peak-to-mean over the first 99.5% of the trace."""
    rr = simulate_rigid(M, m, h, gear_fn, react_fn, max_heavy_dist=d_max, max_time=0.5, max_target_acc=3e4, time_unit=1e-5)
    ee = simulate_compliant(M, m, h, gear_fn, react_fn, k_rope, F, max_heavy_dist=d_max, time_unit=2e-5)
    cut = lambda a: a[:max(1, int(len(a) * 0.995))]
    Fr = cut(rr["F"]); Fc = cut(ee["F"])
    return dict(pR=float(Fr.max() / Fr.mean()), pC=float(Fc.max() / F), vC=ee["v_exit"], vR=rr["v_exit"],
                stopC=ee["stop"], stopR=rr["stop"], dC=ee["d_end"], vhC=ee["vh_end"], yC=ee["y_end"],
                E_payload=0.5 * m * ee["v_exit"] ** 2)
