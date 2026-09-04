"""Example 06: Dual array vs Single array (Mirror Pair).

Demonstrates the central identity of Paper 2:
    G_net(d) = n1 * G1(d) + n2 * G2(d)

When the two arrays are identical (mirror symmetry: G1 = G2 = G and n1 + n2 = k):
    n1 * G(d) + n2 * G(d) = (n1 + n2) * G(d) = k * G(d)

which is mathematically identical to a single array of ratio k.
Mirror symmetry therefore buys lateral load symmetry and redundancy, but CANNOT
alter the force profile. The freedom that reduces peak-to-mean force is allowing
the two arrays to differ.

Run:  python examples/06_dual_vs_single.py
"""

from dualarray_ropecomb import load_case
from ropecomb import simulate_compliant, simulate_rigid

for tag in ("k7", "k9"):
    case = load_case(tag)
    print(f"\n================ Case {case.label} ================")
    print(f"k = {case.k}, N = {case.N} members per array, falls split = {case.n_lo}:{case.n_hi}")
    print(f"Array widths: lead = {case.width_lo:.2f} m, second = {case.width_hi:.2f} m")

    # 1. Dual array simulation
    g_dual = case.ratio_fn()
    run_rigid = simulate_rigid(case.M, case.m, case.v0, g_dual,
                               max_heavy_dist=case.d_max, max_target_acc=3e4, dt=1e-5)
    run_comp = simulate_compliant(case.M, case.m, case.v0, g_dual, case.k_rope,
                                  pretension=case.F, max_heavy_dist=case.d_max, dt=2e-5)

    cut_r = max(2, int(0.995 * len(run_rigid["force_target"])))
    Fr = run_rigid["force_target"][:cut_r]
    pR_dual = float(Fr.max() / Fr.mean())

    cut_c = max(2, int(0.995 * len(run_comp["force_target"])))
    Fc = run_comp["force_target"][:cut_c]
    pC_dual = float(Fc.max() / case.F)

    # 2. Single array / mirror-symmetric baseline
    single = case.single
    pR_single = single["pR"]
    pC_single = single["pC"]

    print("\nPerformance comparison:")
    print(f"  Rigid peak-to-mean:      single = {pR_single:.2f}  -->  dual = {pR_dual:.2f}")
    print(f"  Compliant peak-to-mean:  single = {pC_single:.2f}  -->  dual = {pC_dual:.2f}")
    print(f"  Guide-load imbalance:    structural = {100*abs(case.n_hi - case.n_lo)/case.k:.1f}%  -->  dual = {100*case.imbalance:.1f}%")
