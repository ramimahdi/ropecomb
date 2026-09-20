# RopeComb

Synthesis and simulation code for the dual-array RopeComb, a variable mechanical advantage transmission whose
rising ratio profile is fitted from an array of discrete constant-radius sheaves acting through transverse
deflections of a tension member.

This repository accompanies:

> R. N. Mahdi, *The dual-array RopeComb: a guide-balanced transmission with synthesisable rising mechanical
> advantage for impedance-matched launch*, preprint, 20 September 2026.

Every number and figure in the paper and its supplement is produced by the scripts here from the frozen design
files. Nothing has been built; all results are numerical.

## Install

Python 3.11 with numpy, scipy and matplotlib:

    pip install -r requirements.txt

No installation step is needed: the scripts add their own directories to `sys.path` and can be run from anywhere.

## Layout

    sim/code/       integrators and the single-array fit
    sim/paper2/     dual-array kinematics and the fall-count weighting
    sim/            synthesis, freezing and study scripts, and the frozen designs (JSON)
    sim/arrest/     the arrest mode of Supplementary Section S17 (separate target solver and integrator)
    figsrc/         figure scripts; output goes to figs/
    scaling_check.py, loads_check.py   verification of Table 3 and Table 18

This directory is a frozen, self-contained snapshot of everything used for this paper, so that `./reproduce.sh`
keeps working whatever happens to the rest of the repository. `sim/code/`, `sim/paper2/` and `figsrc/code/` are
therefore copies of the sibling directories they are named after, at the state used for this paper: about 117 kB
of the 2.1 MB here, fifteen files byte-identical and four differing only in that hard-coded absolute paths have
been made relative to the file. Where a copy here and the sibling directory disagree, the copy here is what
produced the paper's numbers.

### Core

| file | role |
|---|---|
| `sim/code/catabult_sim.py` | rigid-body integrator (Supplementary Section S1) |
| `sim/code/catabult_sim_elastic.py` | compliant, pre-tensioned integrator (S2) |
| `sim/code/gear_fn_bridge.py` | closed-form array ratio and take-up, equation (10) |
| `sim/code/ropecomb_fit4.py` | target ODE solver and single-array least-squares fit, equations (1) and (13) |
| `sim/paper2/semisym_fit.py` | dual array: fall-count weighting, balance term, staged solve, order-preserving parametrisation |
| `sim/dual_fit_free.py` | staged solve with the engagement order as a free parameter |
| `sim/unequal_members.py` | count-balanced fits: member floors, seeding, width budget |
| `sim/unequal_merged_seed.py` | merged-seed method: fit N1+N2 members at k/2, deal alternately, excess last |
| `sim/friction_sim.py` | per-contact friction carried through both integrators |

### Frozen designs

| file | contents |
|---|---|
| `sim/locked_4ms.json` | single-array reference designs at 100:1, 1,000:1 and 10,000:1 |
| `sim/paper2/designs_16471.json` | the equal-count dual designs <7,7> and <5,9> with dynamics |
| `sim/designs_counts_16471.json` | the count-balanced designs <8:6,7>, <5:4,9> and <6:5,9>, by balance weight |
| `sim/single_k7.json`, `sim/single_k9.json` | single arrays matched in member count and stage ratio |

`sim/validate.py` re-runs the integrators against these files and reports any drift.

## Reproducing the paper

`./reproduce.sh` runs the quick path: validation, statics, rope flow, the arrest case and all figures, about two
minutes on a laptop. The table below maps each item in the paper to the script that produces it.

| paper item | script | output |
|---|---|---|
| Table 3, scaling relations | `scaling_check.py` | stdout |
| Table 5, Figure 7, stage parity | `sim/paper2/k_parity_study.py`, `stage_parity.py` | stdout |
| Table 6, engagement order | `sim/paper2/lead_order_study.py`, `lead_order_curves.py` | stdout |
| Tables 9, 10, Figures 10, 11 | `sim/paper2/refit_16471.py` | `designs_16471.json` |
| Tables 11, 12, Figure 12 | `sim/freeze_counts.py`, `sim/moment_statics.py` | `designs_counts_16471.json`, `moment_statics.json` |
| Table 16, per-contact friction | `sim/friction_study.py` | `friction_study.json` |
| Table 17, balance frontier | `sim/balanced_study.py`, `balanced_study2.py`, `balanced_study3.py` | `balanced_study*.json` |
| Table 18, component loads | `loads_check.py` | stdout |
| Supplementary S11, fitting against search | `sim/paper2/seed_benefit.py` | stdout |
| Supplementary S12, rope flow | `sim/rope_flow.py` | `rope_flow.json` |
| Supplementary S14, friction model | `sim/friction_study.py`, `sim/balanced_friction.py` | JSON beside the script |
| Supplementary S15, objective variants | `sim/balanced_relative.py`, `balanced_slope.py`, `balance_diagnosis.py` | JSON beside the script |
| Supplementary S16, count-balanced designs | `sim/unequal_*.py` | JSON beside each script |
| Supplementary S17, arrest mode | `sim/arrest/run_arrest.py` | `arrest_results.json` |
| Figures 1, 3, 7-11, 13-15, graphical abstract | `figsrc/make_figures.py` | `figs/` |
| Figure 6 | `figsrc/make_fig_arch.py` | `figs/fig_arch.pdf` |
| Figure 12 | `figsrc/make_fig_counts.py` | `figs/fig_counts.pdf` |
| Figure S17 | `figsrc/make_fig_arrest.py` | `figs/fig_si_arrest.pdf` |

Figures 2, 4 and 5, the hero image and the embodiment renders of Supplementary Section S8 are drawings, not
computed output, and are not reproduced by any script here. Two drawings that the figure scripts composite with
computed panels, `figs/drawn/fig04_single_member.png` (Figure 3) and `fig_configurations.png` (Figure 4), are
included for that reason.

Two caveats on run time and inputs. `sim/freeze_counts.py` takes minutes because it refits and simulates every
design; the frozen JSON in the repository is its output. `sim/code/_grid4.py`, `fit_case.py` and `_lock_case.py`
are the joint member-count and stage-ratio selection of Supplementary Section S3, which took 4,500 fits and 450
simulations per mass ratio; they are included as provenance for `locked_4ms.json` and expect the grid caches that
run produces, so re-running them means re-running the whole selection.

Fits use random restarts, so a refit reproduces a design to within the multi-start spread reported in Section 4.1,
not bit for bit. The frozen JSON files are what the paper's tables quote.

## Licence

Code: MIT, see `LICENSE`. This is a copyright licence only; see `PATENTS`.
