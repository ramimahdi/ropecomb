# RopeComb

Reference implementation for:
- **"The RopeComb: A Configurable Rope-and-Pulley Transmission for Impedance-Matched Mechanical Launch"** (Paper 1)
- **"Dual-Array Variable Mechanical Advantage Transmissions with Guide-Load Balancing for Mechanical Launchers"** (Paper 2)

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

## The dual array

The package above configures a **single** engagement array. A second paper,
*Dual-Array Variable Mechanical Advantage Transmissions with Guide-Load Balancing
for Mechanical Launchers*, extends the same machine to **two arrays sharing one
fixed stage**, and `dualarray_ropecomb` is the code behind it.

The two arrays need not be mirror images, and the central result is that they
should not be. If the stage has ratio `k`, the two movable elements carry `n1`
and `n2` falls with `n1 + n2 = k`, and the net ratio the payload sees is
**fall-count weighted**:

```
G_net(d) = n1 * G1(d) + n2 * G2(d),        n1 + n2 = k
```

Mirror symmetry is the case `G1 = G2` and `n1 = n2`. It is a reasonable default
and the second-best choice, but it cannot improve the force profile, and the
identity shows why: with `G1 = G2 = G`, `n1*G + n2*G = k*G`, which is exactly the
net ratio of a *single* array of the same `N` and `k`. A mirror-symmetric pair
buys mechanical redundancy and lateral balance and nothing else. The freedom that
does buy force uniformity is letting the two arrays differ.

Balancing the transverse load on the guide rails then gives the design condition

```
G1 : G2  =  n2*T2 : n1*T1
```

— the array serving more falls, or losing less tension along its path, takes the
smaller ratio. When the paths are identical this reduces to `G1 : G2 = n2 : n1`,
and adding `n1 = n2` recovers mirror symmetry. Odd `k` cannot satisfy `n1 = n2`
at all, so the structural imbalance `|n1 - n2| / k` is 33% at `k=3`, 20% at
`k=5`, 14% at `k=7`; odd `k` is common for `k >= 5` because two counter-moving
travelling blocks give `k = 2p+1`, so the asymmetric case is the usual one rather
than the exception.

### Sixty-second tour

```python
from dualarray_ropecomb import load_case, net_ratio_fn, balance_ratio

# the two configured designs of the paper, at 16,471 N/m
d = load_case("k7")                  # N = 7 per array, k = 7, n1 = 3, n2 = 4

print(balance_ratio(n1=3, n2=4))     # 4:3 - the lead array takes the LARGER ratio
print(d.width_lo, d.width_hi)        # 1.95 m, 1.98 m
print(d.imbalance)                   # 0.081 - residual guide-load imbalance

g = net_ratio_fn(d)                  # fall-count weighted net ratio
```

### Reproducing the paper's designs

```bash
python examples/09_reproduce_paper2_designs.py
```

reproduces, against a matched single array of the same `N` and `k`:

```
                                    <7,7>      <5,9>
engagement members per array            7          5
fixed-stage ratio k                     7          9
fall split  n1 : n2                   3:4        4:5
array widths (m)                1.95/1.98  1.89/1.55
guide-load imbalance                 8.1%       7.0%

peak-to-mean, compliant
  single array (= mirror pair)        1.40       2.19
  unequal dual array                  1.12       1.48
```

The "single array" row is also the mirror-symmetric row, by the identity above.
That gap — 1.40 to 1.12, and 2.19 to 1.48 — is what the second array buys once
it is allowed to differ from the first.

### Engagement ordering

Which array engages first is not free. The array that engages first must drive
the movable element carrying the **fewer** falls, `floor(k/2)`. Assigning it the
other way costs roughly half again as much residual imbalance at every point on
the force-uniformity trade curve (at `k=5`, median 8.9% against 14.5%). The
alternation of engagements between the two arrays is a *consequence* of the fit,
not a constraint imposed on it — a free fit alternates unaided.

### What is here

| module | role |
|---|---|
| `weighting` | fall counts, `n1`/`n2` splits, the balance condition |
| `geometry` | dual-array net ratio, width budget, per-array member floors |
| `fit` | the staged solve, order-preserving parametrisation, two-stage seed |
| `ordering` | lead assignment, interleaving verification |
| `parity` | even-against-odd stage comparison, the stage energy index `p/k²` |
| `cases` | the two configured designs, ready to re-simulate |

| script (in `paper2/`) | what it produced in the paper |
|---|---|
| `refit_16471.py` | both configured designs at the derived rope stiffness; writes `designs_16471.json` |
| `lead_order_study.py` | the lead-assignment comparison, 40 restarts per cell, medians as well as best fits |
| `lead_order_curves.py` | the two force-uniformity / imbalance trade curves |
| `fit_k5.py` | the `k=5` fits, including the order-free variant showing alternation arises unaided |
| `k_parity_study.py` | the even-against-odd stage-ratio comparison |
| `stage_parity.py` | the stage energy index `p/k²` |
| `check_release_window.py` | sensitivity of peak-to-mean to the release window |
| `stiffness_sensitivity.py` | the rope-stiffness band |
| `make_run_figures.py`, `make_fig_arch.py` | the paper's run and architecture figures (Figures 2, 12, 13) |

*Note on figures:* Paper 1's figures were drawn by external scripts omitted from the repository. For Paper 2, the repo goes beyond that promise by including full figure-generation scripts (`make_run_figures.py` and `make_fig_arch.py`) under `paper2/`.

### Conventions carried over, and one worth restating

Everything in the single-array conventions section above still applies. One is
worth restating because the dual-array numbers are sensitive to it: the rope
stiffness is `k = EA/L = 16,471 N/m` for 2 mm Dyneema R3, with `L` the 15.8 m
runway plus about a metre at the blocks and a metre spare. Across the grades and
routing lengths a designer would actually choose — 15,730 N/m for SK75 over
17.8 m up to 19,663 N/m for a top-end SK99 fibre — the compliant peak-to-mean
moves by 0.01 at `k=7` and 0.03 at `k=9`. It is *not* insensitive to stiffness in
general; taken over an unphysical 8,000 to 30,000 N/m it ranges from 1.04 to
1.61. What makes it stable is that the material and length choices are themselves
constrained.

### Limitations specific to the dual array

- The path-loss correction `T1 != T2` is derived but has no worked example; the
  configured designs assume equal path tensions.
- One mass ratio only (1,000:1), inherited from paper 1's canonical case.
- Nothing has been built or measured. Every result is numerical.

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

   Provisional patent applications covering the physical single-array and dual-array RopeComb transmission apparatus, guide-load balancing mechanisms, and specific geometries have been filed by Interesting Machines LLC (Patent Pending). The physical mechanisms remain the proprietary intellectual property of the author and Interesting Machines LLC. For details, see [PATENTS.txt](PATENTS.txt).

