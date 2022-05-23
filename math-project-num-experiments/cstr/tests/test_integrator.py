import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cstr_package.cstr import CSTR
import numpy as np

cstr = CSTR()
print(np.abs(cstr.f(cstr.xr, cstr.ur) - cstr.xr))
