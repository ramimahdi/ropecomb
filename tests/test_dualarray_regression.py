"""Regression tests pinning the dual-array numbers published in Paper 2.

If one of these fails, either the physics changed or the paper is now wrong.
Neither should happen silently.
"""

import numpy as np
import pytest

from dualarray_ropecomb import (CASES, balance_ratio, comb_width,
                                dual_net_ratio, energy_index, falls,
                                structural_imbalance, verify_interleaving)
from ropecomb import (fit_array, simulate_compliant, simulate_rigid,
                      target_profile)
from ropecomb.geometry import array_ratio

DUAL_CASES = list(CASES.items())


@pytest.mark.parametrize("key,case", DUAL_CASES)
def test_dual_geometry_widths(key, case):
    """Array span widths must match frozen values within 5 mm."""
    assert comb_width(case.R_lo) == pytest.approx(case.width_lo, abs=0.005)
    assert comb_width(case.R_hi) == pytest.approx(case.width_hi, abs=0.005)


@pytest.mark.parametrize("key,case", DUAL_CASES)
def test_dual_interleaving(key, case):
    """Engagement offsets must strictly alternate between arrays."""
    ok, pattern = verify_interleaving((case.s_lo, case.s_hi))
    assert ok, f"Interleaving check failed for {key}: got {pattern}"


@pytest.mark.parametrize("key,case", DUAL_CASES)
def test_lead_assignment_rule(key, case):
    """The array that engages first must carry fewer falls: n_lo = floor(k/2)."""
    assert case.n_lo < case.n_hi
    assert case.n_lo + case.n_hi == case.k
    assert case.n_lo == case.k // 2
    assert case.n_hi == (case.k + 1) // 2


@pytest.mark.parametrize("key,case", DUAL_CASES)
def test_dual_rigid_matches_paper(key, case):
    """Rigid simulation must reproduce the published peak-to-mean."""
    run = simulate_rigid(case.M, case.m, case.v0, case.ratio_fn(),
                         max_heavy_dist=case.d_max, max_target_acc=3e4, dt=1e-5)
    f = run["force_target"]
    cut = max(2, int(0.995 * len(f)))
    pR = float(f[:cut].max() / f[:cut].mean())
    assert pR == pytest.approx(case.pR, abs=0.01)


@pytest.mark.parametrize("key,case", DUAL_CASES)
def test_dual_compliant_matches_paper(key, case):
    """Compliant simulation must reproduce published exit speed and peak-to-mean."""
    run = simulate_compliant(case.M, case.m, case.v0, case.ratio_fn(), case.k_rope,
                             pretension=case.F, max_heavy_dist=case.d_max, dt=2e-5)
    f = run["force_target"]
    cut = max(2, int(0.995 * len(f)))
    pC = float(f[:cut].max() / case.F)
    vC = float(run["summary"]["target_final_speed"])

    assert pC == pytest.approx(case.pC, abs=0.01)
    assert vC == pytest.approx(case.vC, rel=1e-3)


@pytest.mark.parametrize("key,case", DUAL_CASES)
def test_dual_improves_over_single_and_mirror(key, case):
    """Unequal dual array strictly improves peak-to-mean over mirror pair."""
    assert case.pC < case.single["pC"]
    assert case.pR < case.single["pR"]


def test_fall_count_weighting_identity():
    """Identity: n1*G(d) + n2*G(d) == k*G(d). Mirror symmetry cannot alter profile."""
    R = [0.30, 0.22, 0.14]
    s = [0.00, 0.20, 0.40]
    k = 7
    n1, n2 = falls(k)
    d = np.linspace(0.01, 0.8, 200)

    g_single = k * array_ratio(d, R, s)
    g_mirror_dual = dual_net_ratio(d, R, s, n1, R, s, n2)
    np.testing.assert_allclose(g_mirror_dual, g_single, rtol=1e-12)


def test_balance_ratio_and_structural_imbalance():
    """Balance condition G1:G2 = n2:n1 and odd-k structural imbalance."""
    b7 = balance_ratio(3, 4)
    assert float(b7) == pytest.approx(4.0 / 3.0)
    assert str(b7) == "4:3"

    assert structural_imbalance(3) == pytest.approx(1.0 / 3.0)
    assert structural_imbalance(5) == pytest.approx(1.0 / 5.0)
    assert structural_imbalance(7) == pytest.approx(1.0 / 7.0)
    assert structural_imbalance(9) == pytest.approx(1.0 / 9.0)


def test_stage_parity_energy_index():
    """Sheave parasitic energy scales as p / k^2."""
    assert energy_index(2, 5) == pytest.approx(2.0 / 25.0)
    assert energy_index(2, 4) == pytest.approx(2.0 / 16.0)
    assert energy_index(2, 5) < energy_index(2, 4)


def test_single_baseline_fit_reproduction():
    """Step 1 single array fits reproduce frozen baseline residuals."""
    case7 = CASES["k7"]
    spec = target_profile(case7.M, case7.m, case7.v0, 4.0, F=case7.F, n=400)
    fit7 = fit_array(spec, case7.N, case7.k, seed=0, max_width=case7.width_budget)
    assert fit7.rms == pytest.approx(case7.single["rms"], rel=1e-4)
