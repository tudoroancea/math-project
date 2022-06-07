import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
from cstr_package import *

import pandas as pd

init_states = pd.read_csv("exp_all_initial_states.csv", header=None)
xinit = init_states.to_numpy()[:, 0]

print("Running RRLB MPC --------------------------------------------------")

plt.figure(figsize=(15, 7))

run_closed_loop_simulation(
    custom_x_init=xinit,
    scheme=Scheme.RRLB,
    N=25,
    xr=xr1,
    ur=ur1,
)

print("Running regular MPC --------------------------------------------------")

plt.figure(figsize=(15, 7))

run_closed_loop_simulation(
    custom_x_init=xinit,
    scheme=Scheme.REGULAR,
    N=25,
    xr=xr1,
    ur=ur1,
)

plt.show()
