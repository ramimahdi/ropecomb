"""Reporting conventions used throughout the paper."""

from __future__ import annotations

import numpy as np

__all__ = ["peak_to_mean", "fraction_of_ideal"]


def peak_to_mean(force, fraction=0.995):
    """Peak-to-mean payload force over the first ``fraction`` of the trace.

    The release mechanism is specified to operate before the onset of the
    terminal traction-loss pulse, so that pulse is not experienced by the
    payload and including it would misstate the load the payload actually sees.
    Every peak-to-mean figure in the paper uses fraction=0.995.

    Pass fraction=1.0 for the full-stroke figure, which is larger by roughly a
    factor of 2 at 10,000:1 and 20 at 100:1.
    """
    f = np.asarray(force, dtype=float)
    cut = max(2, int(fraction * len(f)))
    window = f[:cut]
    return float(window.max() / window.mean())


def fraction_of_ideal(exit_velocity, ideal_exit_velocity):
    """Exit velocity as a fraction of the ideal profile's own exit velocity."""
    return float(exit_velocity) / float(ideal_exit_velocity)
