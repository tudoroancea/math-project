import os
import sys

import numpy
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
from cstr_package import *

fig = plt.figure(figsize=(15, 7))

# setting the reference point we are going to use
xr = xr1
ur = ur1

# generate random initial states within the following bounds :
# 0.0 <= c_A, c_B <= 50.0 ; 98.0 <= theta <= 500.0 ; 92.0 <= theta_K <= 500.0
np.random.seed(16032001)
nbr_initial_states = 20
initial_states = np.zeros((4, nbr_initial_states))
initial_states[0, :] = np.random.random_sample(nbr_initial_states) * 10.0
initial_states[1, :] = np.random.random_sample(nbr_initial_states) * 10.0
initial_states[2, :] = (
    np.random.random_sample(nbr_initial_states) * (150.0 - 98.0) + 98.0
)
initial_states[3, :] = (
    np.random.random_sample(nbr_initial_states) * (150.0 - 92.0) + 92.0
)
numpy.savetxt("exp1_initial_states.csv", initial_states, delimiter=",")

# run closed loop simulations for each initial state with every MPC scheme and save data
# in a csv file
schemes = [
    "rrlb",
    "reg",
    # "infhorz",
]
df = pd.DataFrame(
    columns=[
        "InitialState",
        "Scheme",
        "ConstraintsViolated",
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

for i in range(initial_states.shape[1]):
    for scheme_str in schemes:
        scheme = Scheme.from_str(scheme_str)
        print(
            "Running experiment for starting point n°{} with {} scheme".format(
                i + 1, scheme_str
            )
        )
        (
            total_cost,
            nbr_iterations_to_convergence,
            constraints_violated,
            sensitivities_computation_times,
            condensation_times,
            solve_times,
        ) = run_closed_loop_simulation(
            scheme=scheme,
            custom_x_init=initial_states[:, i],
            xr=xr1,
            ur=ur1,
            graphics=False,
        )

        df = pd.concat(
            (
                df,
                pd.DataFrame.from_dict(
                    {
                        "InitialState": [i + 1],
                        "Scheme": [scheme_str],
                        "ConstraintsViolated": [constraints_violated],
                        "NbrIterationsToConvergence": [nbr_iterations_to_convergence],
                        "PerformanceMeasure": [total_cost],
                        "AverageSensitivitiesComputationTime": [
                            np.round(np.mean(sensitivities_computation_times), 2)
                        ],
                        "StdErrSensitivitiesComputationTime": [
                            np.round(np.std(sensitivities_computation_times), 2)
                        ],
                        "AverageCondensationTime": [
                            np.round(np.mean(condensation_times), 2)
                        ],
                        "StdErrCondensationTime": [
                            np.round(np.std(condensation_times), 2)
                        ],
                        "AverageSolvingTime": [np.round(np.mean(solve_times), 2)],
                        "StdErrSolvingTime": [np.round(np.std(solve_times), 2)],
                    }
                ),
            ),
            axis=0,
        )

        print("\tDone")

df.to_csv("exp1_results.csv", index=False)
