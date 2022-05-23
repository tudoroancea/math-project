import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cstr_package import run_closed_loop_simulation
from cstr_package.cstr import *

xr = xr3
ur = ur3

# compute optimal solution of the infinite horizon problem (with a big big horizon)
inf_cstr = CSTR(xr=xr, ur=ur)
inf_cstr.T *= 10
inf_cstr.N *= 10
prediction = inf_cstr.initial_prediction(np.array([1.0, 0.5, 100.0, 100.0]), RRLB=False)
total_cost_inf = 0.0
for k in range(inf_cstr.N):
    total_cost_inf += inf_cstr.l(
        prediction[inf_cstr.get_state_idx(k)], prediction[inf_cstr.get_control_idx(k)]
    )
print("Total cost of the infinite horizon problem: {}".format(total_cost_inf))

# compute total costs of closed loop trajectories of RRLB and regular MPC
(total_cost_RRLB, _, _, _, _) = run_closed_loop_simulation(
    RRLB=True,
    xr=xr,
    ur=ur,
)
print("Total cost of the RRLB problem: {}".format(total_cost_RRLB))
(total_cost_reg, _, _, _, _) = run_closed_loop_simulation(
    RRLB=False,
    xr=xr,
    ur=ur,
)
print("Total cost of the regular problem: {}".format(total_cost_reg))
