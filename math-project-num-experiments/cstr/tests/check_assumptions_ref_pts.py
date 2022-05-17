import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cstr_package import CSTR
import numpy as np

cstr2 = CSTR(
    initial_xr=np.array([3.5, 0.75, 97.0, 88.0]),
    initial_ur=np.array([14.19, -7000.0]),
)
cstr3 = CSTR(
    initial_xr=np.array([2.7, 1.0, 105.0, 100.0]),
    initial_ur=np.array([14.19, -7000.0]),
)
