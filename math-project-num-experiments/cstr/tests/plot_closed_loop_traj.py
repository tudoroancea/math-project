import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
from cstr_package import *


print("Running RRLB MPC --------------------------------------------------")

fig = plt.figure(figsize=(15, 7))

run_closed_loop_simulation(
    scheme=Scheme.RRLB,
    xr=xr1,
    ur=ur1,
)

print("Running regular MPC --------------------------------------------------")

fig = plt.figure(figsize=(15, 7))

run_closed_loop_simulation(
    scheme=Scheme.REGULAR,
    xr=xr1,
    ur=ur1,
)

plt.show()
