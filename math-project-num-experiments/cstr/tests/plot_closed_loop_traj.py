import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
from cstr_package import *

import pandas as pd

initial_states = pd.read_csv("exp_all_initial_states.csv").to_numpy()
assert len(initial_states) == 4

print("Running RRLB MPC --------------------------------------------------")

plt.figure(figsize=(15, 9))

(
    total_cost,
    nbr_iterations_to_convergence,
    constraints_violated,
    _,
    _,
    _,
) = run_closed_loop_simulation(
    custom_x_init=initial_states[:, 0],
    scheme=Scheme.RRLB,
    N=50,
    max_nbr_feedbacks=350,
    xr=xr1,
    ur=ur1,
    verbose=True,
)
print("RRLB MPC:")
print("Total cost: {}".format(total_cost))
print("Nbr iterations to convergence: {}".format(nbr_iterations_to_convergence))
print("Constraints violated: {}".format(constraints_violated))
plt.savefig("closedloop_traj_rrlb_1_1_50.png", dpi=300)

# plt.show()
