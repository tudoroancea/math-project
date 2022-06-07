import concurrent.futures
import os
import sys

import pandas as pd
import psutil as psutil

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cstr_package import *

# generate random initial states within the following bounds :
# 0.0 <= c_A, c_B <= 5.0 ; 98.0 <= theta <= 110.0 ; 92.0 <= theta_K <= 110.0
np.random.seed(16032001)
nbr_initial_states = 5
initial_states = np.zeros((4, nbr_initial_states))
initial_states[0, :] = np.random.random_sample(nbr_initial_states) * 5.0
initial_states[1, :] = np.random.random_sample(nbr_initial_states) * 5.0
initial_states[2, :] = (
    np.random.random_sample(nbr_initial_states) * (110.0 - 98.0) + 98.0
)
initial_states[3, :] = (
    np.random.random_sample(nbr_initial_states) * (110.0 - 92.0) + 92.0
)
np.savetxt("exp_all_initial_states.csv", initial_states, delimiter=",")

# setting the reference points we are going to use
ref_points = [(xr1, ur1), (xr2, ur2), (xr3, ur3)]

input_values = []
for i in range(initial_states.shape[1]):
    for j in range(len(ref_points)):
        input_values.append(
            {
                "InitialState": i,
                "RefPoint": j,
            }
        )
input_values = np.array(input_values, dtype=dict)
num_processes = psutil.cpu_count(logical=False)
splitted_input_values = np.array_split(input_values, num_processes)

# define the horizon sizes we want to test
N = [25, 50, 100, 250]


# declare big function that does stuff
def do_stuff(partial_input_values: np.ndarray) -> pd.DataFrame:
    """TODO : rename it"""
    all_dfs = []

    for k in range(partial_input_values.shape[0]):
        initial_state_id: int = partial_input_values[k]["InitialState"]
        ref_point_id: int = partial_input_values[k]["RefPoint"]

        # treat first the infinite horizon MPC (open-loop simulation)
        (
            total_cost,
            nbr_iterations_to_convergence,
            constraints_violated,
        ) = run_open_loop_simulation(
            custom_x_init=initial_states[:, initial_state_id],
            xr=ref_points[ref_point_id][0],
            ur=ref_points[ref_point_id][1],
            scheme=Scheme.INFINITE_HORIZON,
            stop_tol=1.0e-3,
        )
        all_dfs.append(
            pd.DataFrame.from_dict(
                {
                    "InitialState": [initial_state_id + 1],
                    "RefPoint": [ref_point_id + 1],
                    "Scheme": ["infhorz"],
                    "HorizonSize": [INF_HORIZON_SIZE],
                    "ConstraintsViolated": [constraints_violated],
                    "NbrIterationsToConvergence": [nbr_iterations_to_convergence],
                    "PerformanceMeasure": [round(float(total_cost), 3)],
                    "AverageSensitivitiesComputationTime": [None],
                    "StdErrSensitivitiesComputationTime": [None],
                    "AverageCondensationTime": [None],
                    "StdErrCondensationTime": [None],
                    "AverageSolvingTime": [None],
                    "StdErrSolvingTime": [None],
                }
            )
        )
        print(
            "Finished initial state n°{}, ref point n°{} with inf horz from process {}".format(
                initial_state_id + 1, ref_point_id + 1, os.getpid()
            )
        )
        sys.stdout.flush()

        # now treat the RRLB and regular MPC schemes (closed-loop simulation)
        for scheme_str in ["rrlb", "reg"]:
            scheme = Scheme.from_str(scheme_str)
            for n in N:
                (
                    total_cost,
                    nbr_iterations_to_convergence,
                    constraints_violated,
                    sensitivities_computation_times,
                    condensation_times,
                    solve_times,
                ) = run_closed_loop_simulation(
                    scheme=scheme,
                    custom_x_init=initial_states[:, initial_state_id],
                    xr=xr1,
                    ur=ur1,
                    max_nbr_feedbacks=350,
                    stop_tol=1.0e-3,
                    graphics=False,
                    N=n,
                )

                all_dfs.append(
                    pd.DataFrame.from_dict(
                        {
                            "InitialState": [initial_state_id + 1],
                            "RefPoint": [ref_point_id + 1],
                            "Scheme": [scheme_str],
                            "HorizonSize": [n],
                            "ConstraintsViolated": [constraints_violated],
                            "NbrIterationsToConvergence": [
                                nbr_iterations_to_convergence
                            ],
                            "PerformanceMeasure": [total_cost],
                            "AverageSensitivitiesComputationTime": [
                                np.round(np.mean(sensitivities_computation_times), 3)
                            ],
                            "StdErrSensitivitiesComputationTime": [
                                np.round(np.std(sensitivities_computation_times), 3)
                            ],
                            "AverageCondensationTime": [
                                np.round(np.mean(condensation_times), 3)
                            ],
                            "StdErrCondensationTime": [
                                np.round(np.std(condensation_times), 3)
                            ],
                            "AverageSolvingTime": [np.round(np.mean(solve_times), 3)],
                            "StdErrSolvingTime": [np.round(np.std(solve_times), 3)],
                        }
                    ),
                )
                print(
                    "Finished initial state n°{}, ref point n°{}, horizon size {} with {} scheme from process {}".format(
                        initial_state_id + 1,
                        ref_point_id + 1,
                        n,
                        scheme_str,
                        os.getpid(),
                    )
                )
                sys.stdout.flush()

    return pd.concat(all_dfs)


if __name__ == "__main__":
    # simulation time =========================================================
    start_time = time()

    df_results = []

    # parallel version
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = [
            executor.submit(do_stuff, partial_input_values=partial_input_values)
            for partial_input_values in splitted_input_values
        ]
        for result in concurrent.futures.as_completed(results):
            try:
                df_results.append(result.result())
            except Exception as ex:
                print(str(ex))

    # sequential version
    # for partial_input_values in splitted_input_values:
    #     df_results.append(do_stuff(partial_input_values))

    stop = time()
    print("Simulation time: {}".format(stop - start_time))

    # save results
    df = pd.concat(df_results)
    df.sort_values(["InitialState", "RefPoint"], axis=0, ascending=True, inplace=True)
    df.to_csv("exp_all_results.csv", index=False)
