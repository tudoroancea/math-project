import numpy as np
from simulation import run_closed_loop_simulation

total_cost, constraints_violated, times = run_closed_loop_simulation()
print(
    "total_cost = {}, constraints_violated = {}".format(
        total_cost, constraints_violated
    )
)

print(
    "solve times RRLB MPC :\n\tmean={}\n\tstd={}".format(
        np.mean(times), np.std(times)
    )
)