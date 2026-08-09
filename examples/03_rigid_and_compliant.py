"""Step 3: run a geometry rigid and compliant, and check stiffness robustness.

The rigid model is the parameter-free baseline; the compliant model adds the
pre-tensioned output member that the paper argues is a necessary feature of the
apparatus rather than a refinement of it.

The stiffness sweep at the end is the single most useful robustness check. A
geometry whose peak-to-mean climbs steeply with stiffness is leaning on a
modelling assumption rather than on its geometry.

Run:  python examples/03_rigid_and_compliant.py
"""
from ropecomb import (NOMINAL_STIFFNESS, load, peak_to_mean, simulate_compliant,
                      simulate_rigid, stiffness_sweep)

case = load("1000to1")
g = case.ratio_fn()

rigid = simulate_rigid(case.M, case.m, case.v0, g, max_heavy_dist=case.d_max,
                       max_target_acc=3e4)
comp = simulate_compliant(case.M, case.m, case.v0, g, NOMINAL_STIFFNESS,
                          pretension=case.F, max_heavy_dist=case.d_max)

for name, run in (("rigid", rigid), ("compliant", comp)):
    s = run["summary"]
    print(f"{name:>10}: exit {s['target_final_speed']:7.1f} m/s   "
          f"stroke {s['elapsed'] * 1e3:6.2f} ms   "
          f"peak/mean {peak_to_mean(run['force_target']):5.3f}   "
          f"slack {s['slack_fraction'] * 100:4.1f}%   "
          f"energy drift {s['energy_drift'] * 100:.3f}%")

print(f"\nFull-stroke peak/mean, rigid: {peak_to_mean(rigid['force_target'], 1.0):.1f}"
      f"  (vs {peak_to_mean(rigid['force_target']):.2f} over the first 99.5%)")
print("The difference is the terminal traction-loss pulse, which the release")
print("mechanism is specified to precede, so the payload never sees it.\n")

print("stiffness robustness:")
print(f"{'k_rope (N/m)':>14}{'x nominal':>11}{'peak/mean':>11}{'exit m/s':>10}")
for row in stiffness_sweep(case.M, case.m, case.v0, g,
                           [NOMINAL_STIFFNESS * f for f in (0.5, 1.0, 1.5, 2.0)],
                           pretension=case.F, max_heavy_dist=case.d_max):
    print(f"{row['k_rope']:>14,.0f}{row['k_rope'] / NOMINAL_STIFFNESS:>11.1f}"
          f"{row['peak_to_mean']:>11.3f}{row['exit']:>10.1f}")
