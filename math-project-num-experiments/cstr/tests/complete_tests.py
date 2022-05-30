import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
import matplotlib.pyplot as plt
from cstr_package import *

schemes = [
    "rrlb",
    "reg",
    "infhorz",
]
ref_points = [(xr1, ur1), (xr2, ur2), (xr3, ur3)]

df = pd.DataFrame(
    columns=[
        "Ref point",
        "Scheme",
        "Nbr iterations to convergence",
        "Performance measure",
        "Average Sensitivities Computation time",
        "Std err Sensitivities Computation time",
        "Average Condensation time",
        "Std err Condensation time",
        "Average Solving time",
        "Std err Solving time",
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
            for k in range(inf_cstr.N):
                total_cost_inf += inf_cstr.l(
                    prediction[inf_cstr.get_state_idx(k)],
                    prediction[inf_cstr.get_control_idx(k)],
                )
            df = pd.concat(
                (
                    df,
                    pd.DataFrame.from_dict(
                        {
                            "Ref point": [i + 1],
                            "Scheme": [scheme_str],
                            "Nbr iterations to convergence": [None],
                            "Performance measure": [round(float(total_cost_inf), 3)],
                            "Average Sensitivities Computation time": [None],
                            "Std err Sensitivities Computation time": [None],
                            "Average Condensation time": [None],
                            "Std err Condensation time": [None],
                            "Average Solving time": [None],
                            "Std err Solving time": [None],
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
                            "Ref point": [i + 1],
                            "Scheme": [scheme_str],
                            "Nbr iterations to convergence": [
                                nbr_iterations_to_convergence
                            ],
                            "Performance measure": [round(total_cost, 3)],
                            "Average Sensitivities Computation time": [
                                round(np.mean(sensitivites_computation_times), 2)
                            ],
                            "Std err Sensitivities Computation time": [
                                round(np.std(sensitivites_computation_times), 2)
                            ],
                            "Average Condensation time": [
                                round(np.mean(condensation_times), 2)
                            ],
                            "Std err Condensation time": [
                                round(np.std(condensation_times), 2)
                            ],
                            "Average Solving time": [round(np.mean(solve_times), 2)],
                            "Std err Solving time": [round(np.std(solve_times), 2)],
                        }
                    ),
                ),
                axis=0,
            )

        print("\tDone")


df.to_csv("cstr_closed_loop_simulation_results.csv", index=False)
