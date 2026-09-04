"""Example 07: Transverse guide-load balance condition.

In a dual-array launcher, the force each array applies to the central carriage
guide is (fall count) x T x G_array(d).

Balancing the net lateral force across the guide rails requires:
    G1 : G2  =  n2 * T2 : n1 * T1

When rope paths experience equal tension (T1 = T2), this simplifies to:
    G1 : G2  =  n2 : n1

For odd stage ratios k, an identical (mirror) pair cannot satisfy n1 = n2.
The inherent structural imbalance |n1 - n2| / k is:
  - k = 3 : 33.3%
  - k = 5 : 20.0%
  - k = 7 : 14.3%
  - k = 9 : 11.1%

Allowing the two arrays to differ in the inverse ratio of their fall counts
cancels this lateral imbalance.

Run:  python examples/07_balance_condition.py
"""

from dualarray_ropecomb import balance_ratio, falls, structural_imbalance, load_case

print("=== Inherent structural imbalance at odd stage ratios k ===")
print(f"{'stage ratio k':<16} {'falls split n1:n2':<20} {'balance ratio G1:G2':<22} {'mirror imbalance':<18}")
print("-" * 76)
for k in (3, 5, 7, 9, 11):
    n_lo, n_hi = falls(k)
    b_ratio = balance_ratio(n_lo, n_hi)
    imb = structural_imbalance(k)
    print(f"{k:<16d} {n_lo}:{n_hi:<18} {str(b_ratio):<22} {100*imb:>6.1f}%")

print("\n=== Configured design balance verification ===")
for tag in ("k7", "k9"):
    case = load_case(tag)
    n_lo, n_hi = case.n_lo, case.n_hi
    b_rat = balance_ratio(n_lo, n_hi)
    struct_imb = structural_imbalance(case.k)
    print(f"\nDesign {case.label}:")
    print(f"  Fixed stage: k = {case.k}, falls split = {n_lo}:{n_hi}")
    print(f"  Required ratio G1 : G2: {b_rat} (lead array takes larger ratio)")
    print(f"  Mirror pair lateral imbalance:       {100*struct_imb:.1f}%")
    print(f"  Optimized dual-array net imbalance:   {100*case.imbalance:.1f}%")
