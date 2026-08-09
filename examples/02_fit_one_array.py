"""Step 2: fit one array to the target, and look at what the fit produces.

Note the span pattern. It is non-monotonic: modest at the first member, widest
at the second, then narrowing to the floor. Engagement intervals contract
steadily. Neither pattern is expressible in two or three parameters, which is
why low-dimensional parameterisations cannot reach the optimum.

Run:  python examples/02_fit_one_array.py
"""
import numpy as np

from ropecomb import array_ratio, comb_width, fit_array, target_profile

M, m, k, N = 1000.0, 1.0, 5.0, 9
spec = target_profile(M, m, v0=10.0, v_end=4.0, stroke_time=0.1)
print(spec, "\n")

fit = fit_array(spec, N=N, k=k, seed=0)
print(f"{'member':>7}{'s_i (m)':>10}{'R_i (m)':>10}{'span (cm)':>11}{'2/R_i':>9}")
for i, (si, Ri) in enumerate(zip(fit.s, fit.R), 1):
    print(f"{i:>7}{si:>10.4f}{Ri:>10.4f}{200 * Ri:>11.0f}{2 / Ri:>9.1f}")
print(f"\ntotal array width {comb_width(fit.R):.2f} m, residual {fit.rms_pct:.2f}% "
      f"of full scale")

# restarts agree under a well-specified target: the landscape is unimodal
spread = [fit_array(spec, N=N, k=k, seed=s).rms_pct for s in range(8)]
print(f"residual across 8 restarts: {min(spread):.2f}% to {max(spread):.2f}%")

err = k * array_ratio(spec.d, fit.R, fit.s) - spec.G
print(f"worst pointwise ratio error: {np.abs(err).max():.3f} on a full scale of "
      f"{spec.full_scale:.1f}")
