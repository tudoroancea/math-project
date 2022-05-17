import numpy as np
from simulation import run_closed_loop_simulation

(
    total_cost,
    constraints_violated,
    sensitivites_computation_times,
    condensation_times,
    solve_times,
) = run_closed_loop_simulation()
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
    "solve times RRLB MPC :\n\tmean={}\n\tstd={}\n\taverage percentage of total time={}".format(
        np.mean(solve_times), np.std(solve_times), np.mean(solve_times_percentage)
    )
)
print(
    "sensitivities computation times RRLB MPC :\n\tmean={}\n\tstd={}\n\taverage percentage of total time={}".format(
        np.mean(sensitivites_computation_times),
        np.std(sensitivites_computation_times),
        np.mean(sensitivites_computation_times_percentage),
    )
)
print(
    "condensation times RRLB MPC :\n\tmean={}\n\tstd={}\n\taverage percentage of total time={}".format(
        np.mean(condensation_times),
        np.std(condensation_times),
        np.mean(condensation_times_percentage),
    )
)
