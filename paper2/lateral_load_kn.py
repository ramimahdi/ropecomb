"""The lateral guide load of the configured designs in newtons.

Section 9 reports the imbalance as a fraction of the peak carriage reaction.
This converts it to force. The output member carries the design tension T
throughout its length, so array i applies n_i * T * G_i to the carriage; the
net transverse load is the difference and the total reaction is the sum.

    python3 lateral_load_kn.py
"""

import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from ropecomb import target_profile
from ropecomb.geometry import array_ratio

D = json.load(open(os.path.join(HERE, "designs_16471.json")))
T = D["_meta"]["F"]
spec = target_profile(D["_meta"]["M"], D["_meta"]["m"], D["_meta"]["v0"],
                      D["_meta"]["v_stop"], F=T, n=400)
dd = spec.d

print("output-member tension T = %.1f N\n" % T)
for key, n1, n2 in [("k7", 3, 4), ("k9", 4, 5)]:
    du, k = D[key]["dual"], D[key]["k"]
    g1 = array_ratio(dd, np.array(du["R_lo"]), np.array(du["s_lo"]))
    g2 = array_ratio(dd, np.array(du["R_hi"]), np.array(du["s_hi"]))
    net = n1 * g1 + n2 * g2
    imb = np.abs(n1 * g1 - n2 * g2).max() / net.max()
    reaction = T * net.max()
    mirror = abs(n2 - n1) / k
    print("%s   n1=%d  n2=%d  k=%d" % (key, n1, n2, k))
    print("   peak carriage reaction        %7.1f kN" % (reaction / 1e3))
    print("   mirror-symmetric, %4.1f%%       %7.1f kN" % (100 * mirror, mirror * reaction / 1e3))
    print("   this design,      %4.1f%%       %7.1f kN" % (100 * imb, imb * reaction / 1e3))
    print("   removed from the guide        %7.1f kN   (%.0f%%)\n"
          % ((mirror - imb) * reaction / 1e3, 100 * (1 - imb / mirror)))
