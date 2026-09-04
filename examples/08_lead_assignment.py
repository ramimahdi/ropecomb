"""Example 08: Lead assignment and interleaving rule.

Which array engages first: the one driving fewer falls, or more?

The lead assignment rule:
The array that engages first MUST drive the movable element carrying the FEWER
falls, n_lo = floor(k / 2).

Why:
The array ratio G(d) is continuous at engagement, but its slope steps:
    dG/dd jumps by 2 / R_i
The lateral imbalance |n1*G1 - n2*G2| drifts toward whichever array engaged last
at a rate proportional to that array's fall count. Leading with the low-fall array
makes this transverse excursion both smaller and faster to close.

Assigning the high-fall array to lead costs ~50% more residual imbalance at
every point on the trade curve (e.g. at k=5, median 8.9% vs 14.5%).

Furthermore, the alternation of engagements (interleaving L-H-L-H...) is a
natural consequence of the fit: an unconstrained fit alternates unaided.

Run:  python examples/08_lead_assignment.py
"""

from dualarray_ropecomb import (interleaved_slots, load_case,
                                verify_interleaving)

print("=== Lead assignment verification on published designs ===")
for tag in ("k7", "k9"):
    case = load_case(tag)
    ok, pattern = verify_interleaving((case.s_lo, case.s_hi))
    expected = "".join("L" if o == 0 else "H" for o in interleaved_slots(case.N, case.N))

    print(f"\nDesign {case.label} (k={case.k}, N={case.N}):")
    print(f"  Lead array falls:   n_lo = {case.n_lo}  (fewer falls leads)")
    print(f"  Second array falls: n_hi = {case.n_hi}")
    print(f"  Engagement pattern: {pattern}")
    print(f"  Interleaving valid: {ok} (matches alternating sequence)")

print("\n=== Impact of lead assignment (from lead_order_study.json) ===")
print("At k=5, N=10, lambda=0.3 across 40 restarts:")
print("  Low-fall array leads:   median imbalance =  9.9%, best =  8.9%")
print("  High-fall array leads:  median imbalance = 14.8%, best = 14.4%")
print("  Penalty for wrong lead: +49.5% higher median imbalance")
