#!/bin/sh
# Quick reproduction path: validation, statics, rope flow, the arrest case, and every computed figure.
# About two minutes. The long refits (freeze_counts.py, refit_16471.py) and the selection grid are not run here;
# their outputs are the frozen JSON files in sim/. See README.md.
set -e
echo "== validating the frozen designs against the integrators"; python3 sim/validate.py
echo "== scaling relations (Table 3)";                           python3 scaling_check.py
echo "== component loads (Table 18)";                            python3 loads_check.py
echo "== guide moment statics (Table 12, S11)";                  python3 sim/moment_statics.py
echo "== rope passed through the contacts (S12)";                python3 sim/rope_flow.py
echo "== arrest mode (S17)";                                     python3 sim/arrest/run_arrest.py
echo "== figures";                                               python3 figsrc/make_figures.py
python3 figsrc/make_fig_arch.py
python3 figsrc/make_fig_counts.py
python3 figsrc/make_unequal_figs.py
python3 figsrc/make_ga_counts.py
python3 figsrc/make_fig_arrest.py
echo "== done; figures are in figs/"
