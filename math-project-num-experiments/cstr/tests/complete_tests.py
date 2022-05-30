import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cstr_package import *

schemes = [
    "rrlb",
    "reg",
    "infhorz",
]
ref_points = [(xr1, ur1), (xr2, ur2), (xr3, ur3)]

df = pd.DataFrame(
    columns=[
        "RefPoint",
        "Scheme",
        "NbrIterationsToConvergence",
        "PerformanceMeasure",
        "AverageSensitivitiesComputationTime",
        "StdErrSensitivitiesComputationTime",
        "AverageCondensationTime",
        "StdErrCondensationTime",
        "AverageSolvingTime",
        "StdErrSolvingTime",
    ]
)

fig = plt.figure(figsize=(15, 7))

for i in range(3):
    for scheme_str in schemes:
        print(
            "Running experiment for reference point n°{} with {} scheme".format(
                i + 1, scheme_str
            )
        )
        if scheme_str == "rrlb":
            scheme = Scheme.RRLB
        elif scheme_str == "reg":
            scheme = Scheme.REGULAR
        elif scheme_str == "infhorz":
            scheme = Scheme.INFINITE_HORIZON

        plt.clf()

        if scheme_str == "infhorz":
            inf_cstr = CSTR(xr=ref_points[i][0], ur=ref_points[i][1], scheme=scheme)
            prediction = inf_cstr.initial_prediction(np.array([1.0, 0.5, 100.0, 100.0]))
            total_cost_inf = 0.0
            convergence = 0
            for k in range(inf_cstr.N):
                if (
                    convergence == 0
                    and np.linalg.norm(
                        prediction[inf_cstr.get_state_idx(k)] - inf_cstr.xr
                    )
                    < 1e-3
                ):
                    convergence = k

                total_cost_inf += inf_cstr.l(
                    prediction[inf_cstr.get_state_idx(k)],
                    prediction[inf_cstr.get_control_idx(k)],
                )
            df = pd.concat(
                (
                    df,
                    pd.DataFrame.from_dict(
                        {
                            "RefPoint": [i + 1],
                            "Scheme": [scheme_str],
                            "NbrIterationsToConvergence": [None],
                            "PerformanceMeasure": [round(float(total_cost_inf), 3)],
                            "AverageSensitivitiesComputationTime": [None],
                            "StdErrSensitivitiesComputationTime": [None],
                            "AverageCondensationTime": [None],
                            "StdErrCondensationTime": [None],
                            "AverageSolvingTime": [None],
                            "StdErrSolvingTime": [None],
                        }
                    ),
                ),
                axis=0,
            )

        else:
            (
                total_cost,
                nbr_iterations_to_convergence,
                constraints_violated,
                sensitivites_computation_times,
                condensation_times,
                solve_times,
            ) = run_closed_loop_simulation(
                xr=ref_points[i][0],
                ur=ref_points[i][1],
                scheme=scheme,
                verbose=False,
                graphics=True,
            )

            plt.gcf()
            plt.savefig(
                os.path.join(
                    os.path.dirname(__file__),
                    "cstr_" + scheme_str + "_" + str(i + 1) + ".png",
                ),
                dpi=300,
                format="png",
            )

            df = pd.concat(
                (
                    df,
                    pd.DataFrame.from_dict(
                        {
                            "RefPoint": [i + 1],
                            "Scheme": [scheme_str],
                            "NbrIterationsToConvergence": [
                                nbr_iterations_to_convergence
                            ],
                            "PerformanceMeasure": [round(total_cost, 3)],
                            "AverageSensitivitiesComputationTime": [
                                round(np.mean(sensitivites_computation_times), 2)
                            ],
                            "StdErrSensitivitiesComputationTime": [
                                round(np.std(sensitivites_computation_times), 2)
                            ],
                            "AverageCondensationTime": [
                                round(np.mean(condensation_times), 2)
                            ],
                            "StdErrCondensationTime": [
                                round(np.std(condensation_times), 2)
                            ],
                            "AverageSolvingTime": [round(np.mean(solve_times), 2)],
                            "StdErrSolvingTime": [round(np.std(solve_times), 2)],
                        }
                    ),
                ),
                axis=0,
            )

        print("\tDone")


df.to_csv("cstr_closed_loop_simulation_results.csv", index=False)
