import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
from cstr_package import *

import pandas as pd

initial_states = pd.read_csv("exp_all_initial_states.csv").to_numpy()
assert len(initial_states) == 4

# np.random.seed(16032001)
# nbr_initial_states = 1
# initial_states = np.zeros((4, nbr_initial_states))
# initial_states[0, :] = np.random.random_sample(nbr_initial_states) * (4.0 - 1.0) + 1.0
# initial_states[1, :] = np.random.random_sample(nbr_initial_states) * (1.5 - 0.5) + 0.5
# initial_states[2, :] = (
#     np.random.random_sample(nbr_initial_states) * (108.0 - 99.0) + 99.0
# )
# initial_states[3, :] = (
#     np.random.random_sample(nbr_initial_states) * (115.0 - 93.0) + 93.0
# )

print("Running RRLB MPC --------------------------------------------------")

plt.figure(figsize=(15, 9))

global_epsilon = 1.0e-3
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
    N=25,
    max_nbr_feedbacks=350,
    xr=xr1,
    ur=ur1,
    verbose=True,
)
print("RRLB MPC:")
print("Total cost: {}".format(total_cost))
print("Nbr iterations to convergence: {}".format(nbr_iterations_to_convergence))
print("Constraints violated: {}".format(constraints_violated))
# plt.savefig("closedloop_traj_rrlb_1_3.png", dpi=300)

# print("Running regular MPC --------------------------------------------------")
#
# plt.figure(figsize=(15, 7))
#
# (
#     total_cost,
#     nbr_iterations_to_convergence,
#     constraints_violated,
#     _,
#     _,
#     _,
# ) = run_closed_loop_simulation(
#     custom_x_init=xinit,
#     scheme=Scheme.REGULAR,
#     N=100,
#     max_nbr_feedbacks=350,
#     xr=xr1,
#     ur=ur1,
#     verbose=True,
# )
# print("Regular MPC:")
# print("Total cost: {}".format(total_cost))
# print("Nbr iterations to convergence: {}".format(nbr_iterations_to_convergence))
# print("Constraints violated: {}".format(constraints_violated))

plt.show()
