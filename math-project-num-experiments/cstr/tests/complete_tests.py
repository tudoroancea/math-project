import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from cstr_package import *

rrlb = [True, False]
ref_points = [(xr1, ur1), (xr2, ur2), (xr3, ur3)]

df = pd.DataFrame(
    columns=[
        "ref_point",
        "RRLB",
        "nbr_iterations_to_convergence",
        "total_cost",
        "sensitivites_computation_times",
        "condensation_times",
        "solve_times",
    ]
)


for i in range(3):
    for is_rrlb in rrlb:
        (
            total_cost,
            nbr_iterations_to_convergence,
            constraints_violated,
            sensitivites_computation_times,
            condensation_times,
            solve_times,
        ) = run_closed_loop_simulation(
            RRLB=is_rrlb,
            xr=ref_points[i][0],
            ur=ref_points[i][1],
        )

        df = df.append(
            {
                "ref_point": i + 1,
                "RRLB": is_rrlb,
                "nbr_iterations_to_convergence": nbr_iterations_to_convergence,
                "total_cost": total_cost,
                "sensitivites_computation_times": np.mean(
                    sensitivites_computation_times
                ),
                "condensation_times": np.mean(condensation_times),
                "solve_times": np.mean(solve_times),
            },
            ignore_index=True,
        )

df.to_csv("cstr_closed_loop_simulation_results.csv", index=False)
df.style.to_latex("cstr_closed_loop_simulation_results.tex")
