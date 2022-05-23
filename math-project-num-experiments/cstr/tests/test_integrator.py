import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cstr_package.cstr import *
import numpy as np

ref_points = [(xr1, ur1), (xr2, ur2), (xr3, ur3)]

for i in range(3):
    cstr = CSTR(
        xr=ref_points[i][0],
        ur=ref_points[i][1],
        RRLB=False,  # does not matter for the computation, but avoids the unuseful computation of RRLB functions
    )
    print(np.abs(cstr.f(cstr.xr, cstr.ur) - cstr.xr))
