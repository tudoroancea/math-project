import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
from cstr_package import *

import pandas as pd

init_states = pd.read_csv("exp_all_initial_states.csv", header=None)
xinit = init_states.to_numpy()[:, 0]

run_open_loop_simulation(
    custom_x_init=xinit, xr=xr1, ur=ur1, scheme=Scheme.INFINITE_HORIZON
)
