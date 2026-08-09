"""Reproduce the table of section 5.3 from the stored geometries.

This is the fastest check that an installation is behaving: it re-simulates the
three published designs and prints the paper's table beside the values it just
computed.

Run:  python examples/05_reproduce_canonical.py
"""
from ropecomb import (NOMINAL_STIFFNESS, CANONICAL, comb_width, peak_to_mean,
                      simulate_compliant, simulate_rigid)

rows = []
for key, case in CANONICAL.items():
    g = case.ratio_fn()
    r = simulate_rigid(case.M, case.m, case.v0, g, max_heavy_dist=case.d_max,
                       max_target_acc=3e4)
    q = simulate_compliant(case.M, case.m, case.v0, g, NOMINAL_STIFFNESS,
                           pretension=case.F, max_heavy_dist=case.d_max)
    rows.append((case,
                 comb_width(case.R),
                 r["summary"]["target_final_speed"],
                 q["summary"]["target_final_speed"],
                 peak_to_mean(r["force_target"]),
                 peak_to_mean(q["force_target"])))

hdr = f"{'':<28}" + "".join(f"{c.label:>12}" for c, *_ in rows)
print(hdr)
print("-" * len(hdr))


def line(name, vals, fmt="{:>12.2f}"):
    print(f"{name:<28}" + "".join(fmt.format(v) for v in vals))


line("design payload force (N)", [c.F for c, *_ in rows], "{:>12,.0f}")
line("engagement members", [c.N for c, *_ in rows], "{:>12d}")
line("fixed-stage ratio k", [c.k for c, *_ in rows], "{:>12.0f}")
line("array width (m)", [r[1] for r in rows])
line("exit velocity, rigid", [r[2] for r in rows], "{:>12.1f}")
line("exit velocity, compliant", [r[3] for r in rows], "{:>12.1f}")
line("peak-to-mean, rigid", [r[4] for r in rows], "{:>12.2f}")
line("peak-to-mean, compliant", [r[5] for r in rows], "{:>12.2f}")

print("\nSheave inertia is NOT included above. Four 100 g sheaves in the fast path")
print("act as added payload mass and cost a further 15.5% of exit velocity:")
line("corrected for 4 sheaves", [r[3] * (1.0 / 1.4) ** 0.5 for r in rows], "{:>12.1f}")
