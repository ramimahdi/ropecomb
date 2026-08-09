"""Step 1: turn a stated deceleration into a design force and target profile.

The paper's specification: the source mass enters at 10 m/s and is decelerated
to 4 m/s over 0.1 s. Solve for the uniform payload force that achieves exactly
that, then sample the ratio profile it implies.

Run:  python examples/01_target_profile.py
"""
from ropecomb import member_floor, solve_design_force, target_profile

for label, M in (("100:1", 100.0), ("1,000:1", 1000.0), ("10,000:1", 10000.0)):
    F = solve_design_force(M, m=1.0, v0=10.0, v_end=4.0, stroke_time=0.1)
    spec = target_profile(M, m=1.0, F=F, v_end=4.0)
    ke_surrendered = 1.0 - (4.0 / 10.0) ** 2
    print(f"{label:>9}  F = {F:9,.1f} N   braking distance {spec.d_max:.3f} m   "
          f"terminal ratio {spec.G[-1]:6.1f}:1   ideal exit {spec.ideal_exit:7.1f} m/s")
    for k in (2, 5, 9):
        print(f"{'':>9}    at k={k}: at least {member_floor(spec.G[-1], k)} members "
              f"are needed to reach that ratio")

print(f"\nStopping at 4 m/s surrenders {ke_surrendered:.0%} of the source's kinetic "
      f"energy.\nThe braking distance is the same in all three cases because the "
      f"deceleration is.")
