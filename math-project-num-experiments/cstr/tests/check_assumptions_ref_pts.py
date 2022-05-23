import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cstr_package import *
import numpy as np

cstr = CSTR(
    xr=xr2,
    ur=ur2,
)
print(np.abs(cstr.f(cstr.xr, cstr.ur) - cstr.xr))
cstr = CSTR(
    xr=xr3,
    ur=ur3,
)
print(np.abs(cstr.f(cstr.xr, cstr.ur) - cstr.xr))
