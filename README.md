# RopeComb

Reference implementation for **"The RopeComb: A Configurable Rope-and-Pulley
Transmission for Impedance-Matched Mechanical Launch."**

A RopeComb is a passive transmission whose mechanical advantage rises
continuously through its stroke, which is what lets a slow heavy source
accelerate a fast light payload at near-constant force. A carriage carrying `N`
engagement members descends past a row of fixed supports, progressively
deflecting a tension member into successive spans. Two independent geometric
freedoms per member, span width and engagement offset, make essentially any
monotonically rising ratio profile reachable.

Because the resulting displacement ratio has a **closed form**,

```
G(d) = k * SUM_i  2 D_i / sqrt(R_i^2 + D_i^2),      D_i = max(0, d - s_i)
```

matching a target profile is a smooth least-squares fit rather than a search.
That is the central methodological claim, and this package is the code behind it.

## Install

```bash
git clone https://github.com/<user>/ropecomb
cd ropecomb
pip install -e .
```

Requires Python 3.9+, numpy and scipy. `pip install -e ".[dev]"` adds pytest.

## Sixty-second tour

```python
from ropecomb import target_profile, fit_array, net_ratio_fn
from ropecomb import simulate_compliant, peak_to_mean

# 1. state a deceleration, solve for the payload force it implies
spec = target_profile(M=1000.0, m=1.0, v0=10.0, v_end=4.0, stroke_time=0.1)
print(spec)          # F = 3,169.6 N, braking distance 0.854 m

# 2. fit a nine-member array with a 5:1 second stage
fit = fit_array(spec, N=9, k=5.0, seed=0)
print(fit)           # residual 0.78% of full scale

# 3. simulate it on a compliant, pre-tensioned output member
g = net_ratio_fn(fit.R, fit.s, k=5.0)
run = simulate_compliant(1000.0, 1.0, 10.0, g, k_rope=16471.0,
                         pretension=spec.F, max_heavy_dist=spec.d_max)
print(run["summary"]["target_final_speed"],       # 318 m/s
      peak_to_mean(run["force_target"]))          # 1.09
```

## What is here

| module | role |
|---|---|
| `geometry` | the closed-form array ratio, array width, member floor |
| `target` | **step 1**: deceleration to design force to target profile |
| `fit` | **step 2**: bounded least-squares geometry fit |
| `rigid` | inextensible-member simulator, and the ideal uniform-force profile |
| `compliant` | elastic pre-tensioned-member simulator, stiffness sweep |
| `metrics` | the paper's reporting conventions |
| `design` | **Algorithm 1** end to end |
| `cases` | the three published designs, ready to re-simulate |

## Examples

```bash
python examples/01_target_profile.py        # deceleration -> design force
python examples/02_fit_one_array.py         # fit one array, inspect the geometry
python examples/03_rigid_and_compliant.py   # both simulators + stiffness robustness
python examples/04_algorithm1.py            # Algorithm 1 on a reduced grid (~1 min)
python examples/05_reproduce_canonical.py   # reproduce the table of section 5.3
```

`05` is the fastest check that an installation is behaving. It reproduces:

```
                                   100:1     1,000:1    10,000:1
engagement members                     8           9          13
fixed-stage ratio k                    2           5           9
array width (m)                     2.82        2.42        1.92
exit velocity, compliant           100.2       318.2      1004.8
peak-to-mean, rigid                 2.05        2.07        2.10
peak-to-mean, compliant             1.12        1.09        1.19
```

## Reproducing the search

`examples/04` runs a reduced grid. The published run is

```python
from ropecomb import design, best
cands = design(M=1000.0, m=1.0)      # N = 2..16, k in {1,2,3,5,7,9}, 50 restarts
print(best(cands))
```

which is 4,500 fits and 450 simulations per case, roughly twenty minutes. No
dynamic simulation enters the fit, which is what makes that affordable: only the
450 survivors are ever simulated.

## Conventions worth knowing before you compare numbers

**Peak-to-mean is taken over the first 99.5% of the force trace.** The release
mechanism is specified to operate before the terminal traction-loss pulse, so
the payload never experiences it and including it would misstate the load. Pass
`peak_to_mean(f, 1.0)` for the full-stroke figure, which is larger by roughly a
factor of 2 at 10,000:1 and 20 at 100:1.

**The objective has no penalty terms.** It is the plain squared residual against
the target. Earlier drafts carried weighting, one-sided terms and curvature
penalties; none is needed once the target is specified by deceleration rather
than by a chosen force. `fit_array` has a `max_width` guard which is inactive at
these scales by two orders of magnitude and exists only to stop a pathological
restart running away.

**Pruning is evaluated on the longer of the rigid and compliant strokes.** A
compliant member lets the carriage travel further, so members that look
redundant in the rigid model do engage in the real machine.

**The compliant result is sensitive to member stiffness.** Use
`stiffness_sweep`. A geometry whose peak-to-mean climbs steeply with stiffness
is leaning on a modelling assumption rather than on its geometry.

## Known limitations

These are the model's, not the code's, and they are stated in the paper.

- **No sheave inertia.** Every sheave the output member traverses runs at full
  payload speed and acts as added payload mass. Four 100 g sheaves against a
  1 kg payload cost 15.5% of exit velocity. This is first-order and roughly
  scale-invariant. The velocities this package reports are upper bounds.
- **No friction, member mass or drag.**
- **The terminal transient is not time-step converged.** Exit velocities and
  stroke durations are reliable; peak force magnitudes and the precise
  divergence time are indicative only.
- **Nothing has been built or measured.** Every result is numerical.

## Citing

See `CITATION.cff`. The paper is the primary reference; cite this package if you
use the implementation itself.

## License & Intellectual Property

This repository contains a mix of open-source software, open-access academic media, and proprietary hardware designs. Please review the following specific intellectual property notices:

1. **Software Code (MIT License)**
   All simulation scripts, physics models, and interactive web code (`.py`, `.js`, `.html`, etc.) in this repository are released under the MIT License. You are free to use, copy, modify, distribute, and run these digital assets.

2. **Academic Paper & Media (CC BY 4.0)**
   The manuscript *The RopeComb: A Configurable Rope-and-Pulley Transmission for Impedance-Matched Mechanical Launch* and all associated diagrams, figures, and data are licensed under a Creative Commons Attribution 4.0 International License (CC BY 4.0). You may share and adapt this material, provided proper attribution is given to Rami N. Mahdi and Interesting Machines LLC.

3. **Physical Hardware & Mechanism (Patent Pending)**
   The MIT License in this repository applies solely to the software source code and digital assets. No license, express or implied, is granted to any physical hardware, mechanisms, or patents described, simulated, or depicted by this repository.

   A provisional patent application covering the physical RopeComb transmission apparatus and its specific geometries has been filed by Interesting Machines LLC (Patent Pending). The physical mechanisms remain the proprietary intellectual property of the author and Interesting Machines LLC. For details, see [PATENTS.txt](PATENTS.txt).
