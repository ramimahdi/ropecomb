"""Regression tests pinning the numbers published in the paper.

If one of these fails, either the physics changed or the paper is now wrong.
Neither should happen silently.
"""
import numpy as np
import pytest

from ropecomb import (CANONICAL, NOMINAL_STIFFNESS, array_ratio, comb_width,
                      fit_array, member_floor, net_ratio_fn, peak_to_mean,
                      simulate_compliant, simulate_ideal, simulate_rigid,
                      solve_design_force, target_profile)

CASES = list(CANONICAL.items())


@pytest.mark.parametrize("key,case", CASES)
def test_geometry_width(key, case):
    assert comb_width(case.R) == pytest.approx(case.expect["width"], abs=0.005)


@pytest.mark.parametrize("key,case", CASES)
def test_rigid_matches_paper(key, case):
    run = simulate_rigid(case.M, case.m, case.v0, case.ratio_fn(),
                         max_heavy_dist=case.d_max, max_target_acc=3e4, dt=1e-5)
    assert run["summary"]["target_final_speed"] == pytest.approx(
        case.expect["exit_rigid"], rel=1e-3)
    assert peak_to_mean(run["force_target"]) == pytest.approx(
        case.expect["p2m_rigid"], abs=0.005)


@pytest.mark.parametrize("key,case", CASES)
def test_compliant_matches_paper(key, case):
    run = simulate_compliant(case.M, case.m, case.v0, case.ratio_fn(),
                             NOMINAL_STIFFNESS, pretension=case.F,
                             max_heavy_dist=case.d_max, dt=2e-5)
    assert run["summary"]["target_final_speed"] == pytest.approx(
        case.expect["exit_compliant"], rel=1e-3)
    assert peak_to_mean(run["force_target"]) == pytest.approx(
        case.expect["p2m_compliant"], abs=0.005)


@pytest.mark.parametrize("key,case", CASES)
def test_compliant_meets_the_force_criterion(key, case):
    """Section 5.6: every canonical design holds peak-to-mean at or below 1.3."""
    run = simulate_compliant(case.M, case.m, case.v0, case.ratio_fn(),
                             NOMINAL_STIFFNESS, pretension=case.F,
                             max_heavy_dist=case.d_max, dt=2e-5)
    assert peak_to_mean(run["force_target"]) <= 1.3


def test_design_force_bisection():
    """Section 5.2: the solved force for the 1,000:1 case is 3,170 N."""
    F = solve_design_force(1000.0, 1.0, v0=10.0, v_end=4.0, stroke_time=0.1)
    assert F == pytest.approx(3169.6, rel=2e-3)


def test_braking_distance_is_common_to_all_cases():
    """The deceleration fixes the braking distance, so mass ratio cannot move it."""
    dmax = [target_profile(c.M, c.m, F=c.F, v_end=c.v_end).d_max
            for c in CANONICAL.values()]
    assert max(dmax) - min(dmax) < 1e-3


def test_closed_form_matches_a_polyline():
    """Appendix C: the closed form against explicit rope length, finite differenced."""
    R = np.array([0.30, 0.22, 0.14])
    s = np.array([0.00, 0.20, 0.40])
    d = np.linspace(0.01, 0.8, 400)

    def length(dd):
        D = np.maximum(dd - s, 0.0)
        return float(np.sum(2.0 * (np.sqrt(R ** 2 + D ** 2) - R)))

    h = 1e-7
    numeric = np.array([(length(x + h) - length(x - h)) / (2 * h) for x in d])
    closed = array_ratio(d, R, s)
    assert np.max(np.abs(numeric - closed)) < 1e-6


def test_member_floor():
    """Each member contributes at most 2 to the array ratio."""
    assert member_floor(79.3, 5.0) == 8
    assert member_floor(25.0, 2.0) == 7


def test_ideal_and_real_share_one_loop():
    """The ideal profile holds a genuinely uniform payload force."""
    run = simulate_ideal(1000.0, 1.0, 10.0, 3169.6, min_heavy_speed=4.0)
    f = run["force_target"]
    assert peak_to_mean(f, 1.0) < 1.02


def test_fit_reproduces_a_canonical_residual():
    """A fresh fit at the published N and k lands at the published residual."""
    case = CANONICAL["1000to1"]
    spec = target_profile(case.M, case.m, F=case.F, v_end=case.v_end)
    best = min(fit_array(spec, N=case.N, k=case.k, seed=s).rms_pct
               for s in range(10))
    assert best == pytest.approx(case.expect["residual_pct"], abs=0.1)


def test_pruning_removes_nothing_in_the_canonical_designs():
    """Section 4.6: every fitted member engages under the target of section 5.2."""
    for case in CANONICAL.values():
        run = simulate_compliant(case.M, case.m, case.v0, case.ratio_fn(),
                                 NOMINAL_STIFFNESS, pretension=case.F,
                                 max_heavy_dist=case.d_max, dt=2e-5)
        reach = run["summary"]["heavy_travel"]
        assert all(si < reach for si in case.s)
