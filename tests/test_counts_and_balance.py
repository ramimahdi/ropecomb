"""Tests for balance by member count, merged seeding, and imbalance reporting.

The numbers pinned here are the paper's: the member floors and count-balanced
pairs at k = 7 and k = 9, the guide moment of the configured <7,7> design, and
the fact that a count-balanced pair leaves nothing at the end of the stroke
while an equal-count pair at odd k leaves the mirror residual 1/k.
"""

import numpy as np
import pytest

from dualarray_ropecomb import (CASES, count_balanced_counts, count_floors,
                                count_ladder, fit_dual_staged, imbalance_trace,
                                member_positions, pitch_moment, seed_merged,
                                terminal_imbalance)
from ropecomb import simulate_compliant, target_profile

G_TERMINAL_1000 = 79.3          # terminal target ratio of the 1,000:1 case


# --------------------------------------------------------------- count condition

def test_member_floors_match_paper():
    """N_j >= G*(d_max) / (4 n_j): (7, 5) at k = 7 and (5, 4) at k = 9."""
    assert count_floors(G_TERMINAL_1000, 7) == (7, 5)
    assert count_floors(G_TERMINAL_1000, 9) == (5, 4)


def test_count_balanced_pairs_match_paper():
    """The smallest balanced counts at or above the floors."""
    assert count_balanced_counts(7, G_TERMINAL_1000) == (8, 6)
    assert count_balanced_counts(9, G_TERMINAL_1000) == (5, 4)


def test_terminal_imbalance_zero_only_when_balanced():
    """n1 N1 = n2 N2 leaves nothing; equal counts at odd k leave 1/k."""
    assert terminal_imbalance(3, 8, 4, 6) == 0.0          # <8:6,7>
    assert terminal_imbalance(4, 5, 5, 4) == 0.0          # <5:4,9>
    assert terminal_imbalance(3, 7, 4, 7) == pytest.approx(1.0 / 7.0)
    assert terminal_imbalance(4, 5, 5, 5) == pytest.approx(1.0 / 9.0)


def test_near_miss_counts():
    """<6:5,9> misses the condition by one part in 49."""
    assert terminal_imbalance(4, 6, 5, 5) == pytest.approx(1.0 / 49.0)


def test_count_ladder_flags_balanced_rows():
    rows = count_ladder(7, (7, 5), n_rows=4)
    assert [r["total"] for r in rows] == [12, 13, 14, 15]
    balanced = [r for r in rows if r["balanced"]]
    assert balanced and (balanced[0]["N_lo"], balanced[0]["N_hi"]) == (8, 6)


# ------------------------------------------------------------------- imbalance

@pytest.mark.parametrize("key", ["k7", "k9"])
def test_imbalance_trace_reproduces_frozen_geometric_figure(key):
    """At uniform tension the trace must return the imbalance stored with the case."""
    c = CASES[key]
    d = np.linspace(0.0, c.d_max, 400)
    tr = imbalance_trace(c, d=d, tension=c.F)
    assert tr["eps_geometric"] == pytest.approx(c.imbalance, abs=1e-4)
    assert tr["mirror_residual"] == pytest.approx(abs(c.n_hi - c.n_lo) / c.k)


def test_imbalance_trace_from_a_run_is_in_newtons():
    """A run supplies the tension, so the reaction comes back as a force."""
    c = CASES["k7"]
    run = simulate_compliant(c.M, c.m, c.v0, c.ratio_fn(), c.k_rope,
                             pretension=c.F, max_heavy_dist=c.d_max, dt=2e-5)
    tr = imbalance_trace(c, run)
    assert tr["has_tension"]
    assert 200e3 < tr["F_carriage_peak"] < 300e3          # about 240 kN at 1,000:1
    assert 0.0 < tr["eps_peak"] < tr["mirror_residual"]
    assert tr["dF_peak"] == pytest.approx(tr["eps_peak"] * tr["F_carriage_peak"], rel=1e-9)


def test_pitch_moment_matches_paper_table_12():
    """<7,7> at uniform tension: 21.3 kN m peak, 90 mm equivalent offset."""
    c = CASES["k7"]
    d = np.linspace(0.0, c.d_max, 400)
    pm = pitch_moment(c, d=d, tension=c.F, bearing_spacing=1.0)
    assert pm["moment_peak"] / 1e3 == pytest.approx(21.3, abs=0.2)
    assert 1000.0 * pm["offset_equivalent"] == pytest.approx(90.0, abs=2.0)
    assert pm["bearing_couple_peak"] == pytest.approx(pm["moment_peak"])


def test_member_positions_lay_spans_contiguously():
    R = [0.3, 0.2, 0.1]
    s = [0.0, 0.1, 0.2]
    x = member_positions(R, s)
    np.testing.assert_allclose(x, [0.3, 0.8, 1.1])   # spans 0.6, 0.4, 0.2 end to end
    assert x[-1] + R[-1] == pytest.approx(2.0 * sum(R))


# --------------------------------------------------------------------- seeding

def test_merged_seed_deals_alternately_with_the_excess_last():
    """8 and 6 members deal L H L H ... L L: the excess engages at the end."""
    c = CASES["k7"]
    spec = target_profile(c.M, c.m, c.v0, 4.0, F=c.F, n=200)
    R_lo, s_lo, R_hi, s_hi, order = seed_merged(spec, 8, 6, 7, width_budget=2.11, seed=0)

    assert len(R_lo) == 8 and len(R_hi) == 6
    assert order == [0, 1] * 6 + [0, 0]
    events = sorted([(float(v), 0) for v in s_lo] + [(float(v), 1) for v in s_hi])
    assert [tag for _, tag in events] == order
    assert np.all(np.diff(np.sort(s_lo)) >= 0.0)


def test_count_balanced_fit_reproduces_the_paper_design():
    """A merged-seeded fit at (8, 6) must land on the published <8:6,7> design."""
    c = CASES["k7"]
    spec = target_profile(c.M, c.m, c.v0, 4.0, F=c.F, n=400)
    fit = fit_dual_staged(spec, k=7, N_lo=8, N_hi=6, width_budget=2.11,
                          lam_balance=0.03, restarts=8, seed=0)

    assert (fit.raw["N_lo"], fit.raw["N_hi"]) == (8, 6)
    assert fit.rms == pytest.approx(0.413, abs=0.03)
    assert fit.imbalance == pytest.approx(0.023, abs=0.006)
    assert fit.terminal == pytest.approx(74.0, abs=1.0)
    assert max(fit.width_lo, fit.width_hi) <= 2.11 * 1.02

    tr = imbalance_trace(fit, d=spec.d, tension=spec.F)
    assert tr["eps_terminal"] < 0.005          # the counts carry the end of the stroke


def test_equal_count_seed_path_still_works():
    """The original (R_star, s_star) signature must keep working unchanged."""
    from ropecomb import fit_array
    c = CASES["k7"]
    spec = target_profile(c.M, c.m, c.v0, 4.0, F=c.F, n=200)
    single = fit_array(spec, c.N, c.k, seed=0, max_width=c.width_budget)
    fit = fit_dual_staged(spec, R_star=single.R, s_star=single.s, k=7,
                          width_budget=c.width_budget, lam_balance=1.0)
    assert len(fit.R_lo) == len(fit.R_hi) == c.N
    assert fit.k == 7
