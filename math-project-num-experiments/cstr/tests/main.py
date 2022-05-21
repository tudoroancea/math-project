import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from cstr_package import run_closed_loop_simulation

(
    total_cost,
    constraints_violated,
    sensitivites_computation_times,
    condensation_times,
    solve_times,
) = run_closed_loop_simulation(RRLB=False)
# plt.gcf()
# plt.savefig(
#     os.path.join(os.path.dirname(__file__), "cstr_rrlb.png"), dpi=300, format="png"
# )

print(
    "total_cost = {}, constraints_violated = {}".format(
        total_cost, constraints_violated
    )
)
total_times = solve_times + sensitivites_computation_times + condensation_times
solve_times_percentage = solve_times / total_times
sensitivites_computation_times_percentage = sensitivites_computation_times / total_times
condensation_times_percentage = condensation_times / total_times

print(
    "solve times RRLB MPC :\n\tmean={:.3f} ms\n\tstd={:.3f} ms\n\taverage percentage of total time={:.2f} %".format(
        np.mean(solve_times), np.std(solve_times), 100 * np.mean(solve_times_percentage)
    )
)
print(
    "sensitivities computation times RRLB MPC :\n\tmean={:.3f} ms\n\tstd={:.3f} ms\n\taverage percentage of total time={:.2f} %".format(
        np.mean(sensitivites_computation_times),
        np.std(sensitivites_computation_times),
        100 * np.mean(sensitivites_computation_times_percentage),
    )
)
print(
    "condensation times RRLB MPC :\n\tmean={:.3f} ms\n\tstd={:.3f} ms\n\taverage percentage of total time={:.2f} %".format(
        np.mean(condensation_times),
        np.std(condensation_times),
        100 * np.mean(condensation_times_percentage),
    )
)
plt.show()
