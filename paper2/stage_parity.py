"""Why the fixed-ratio stage produces an ODD k, and what forcing an even k costs.

Reeving rule (reconstructed from the inventor's account and corroborated below).
Let p be the total number of sheaves carried on the travelling block or blocks.
The number of rope parts spanning the travelling assembly, which is the stage
ratio k, is:

    stationary end taken to a TRAVELLING block  ->  k = 2p + 1   (odd)
    stationary end taken to the FRAME           ->  k = 2p       (even)

The payload takes the one unpaired part. Each sheave contributes a pair; the free
end going to the payload is unpaired, so the odd family is what a stage with the
payload on a free end naturally produces. Two counter-moving blocks let p be split
between them, which is how k = 5 is reached with one sheave per block.

Stage parasitic energy. The travelling blocks translate at v_payload / k, so the
kinetic energy held in the p sheaves goes as

    E_stage  ~  p * m_s * (v_payload / k)^2   ~   p / k^2   per unit sheave mass.

Adding a sheave costs mass but buys ratio, and because k grows linearly in p while
the energy falls as 1/k^2, the trade is favourable: p/(2p+1)^2 ~ 1/(4p).
"""

import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from dualarray_ropecomb.parity import energy_index


def main():
    print(__doc__)
    print("Odd family, k = 2p + 1 (payload on the free end):\n")
    print("  %-4s %-4s %10s %14s" % ("p", "k", "p/k^2", "vs k=3"))
    base = energy_index(1, 3)
    for p in range(1, 6):
        k = 2 * p + 1
        e = energy_index(p, k)
        print("  %-4d %-4d %10.4f %13.0f%%" % (p, k, e, 100 * (e / base - 1)))

    print("\nForcing an even k, at equal or greater sheave count:\n")
    print("  %-22s %10s %10s %12s" % ("comparison", "odd p/k^2", "even p/k^2", "odd is"))
    for p in range(2, 6):
        ko, ke_same, ke_next = 2 * p + 1, 2 * p, 2 * p + 2
        eo = energy_index(p, ko)
        print("  k=%-2d (p=%d) vs k=%-2d (p=%d) %8.4f %10.4f %11.1f%% lower"
              % (ko, p, ke_same, p, eo, energy_index(p, ke_same),
                 100 * (1 - eo / energy_index(p, ke_same))))
        print("  k=%-2d (p=%d) vs k=%-2d (p=%d) %8.4f %10.4f %11.1f%% lower, one fewer sheave"
              % (ko, p, ke_next, p + 1, eo, energy_index(p + 1, ke_next),
                 100 * (1 - eo / energy_index(p + 1, ke_next))))
        print()

    print("Corroboration: every fixed-stage ratio of 3 or more in the parent")
    print("provisional and in the published paper is odd.\n")
    for src, val in [("parent FIG. 8", 3), ("parent FIG. 9", 7),
                     ("parent FIG. 10", 5), ("parent FIG. 10", 7),
                     ("parent FIG. 10", 9), ("parent 1,000:1 sim", 5),
                     ("parent 10,000:1 sim", 9), ("paper 1,000:1", 5),
                     ("paper 10,000:1", 9), ("parent/paper 100:1", 2)]:
        p = (val - 1) / 2.0
        note = "p = %.0f" % p if val % 2 else "EVEN, p = %.0f, the degenerate small case" % (val / 2.0)
        print("  %-22s k = %-2d  %s" % (src, val, note))


if __name__ == "__main__":
    main()
