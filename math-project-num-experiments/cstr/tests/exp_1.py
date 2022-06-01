import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
from cstr_package import *

fig = plt.figure(figsize=(15, 7))

run_closed_loop_simulation(
    scheme=Scheme.RRLB,
    xr=xr1,
    ur=ur1,
    graphics=True,
)
plt.show()
