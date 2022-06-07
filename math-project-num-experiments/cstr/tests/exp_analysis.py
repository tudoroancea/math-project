import itertools
import os
import sys

import pandas as pd
from matplotlib import pyplot as plt

from exp_all import nbr_initial_states

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cstr_package import *

# load the csv file with the results
df = pd.read_csv("exp_all_results.csv")

# First see if there are any aberrant data (situations where the RRLB MPC or the
# regular MPC reach better performance measures than the infinite horizon MPC, and
# without any constraint violations)
for i in range(nbr_initial_states):
    for j in range(3):
        df_tpr = df[(df["InitialState"] == (i + 1)) & (df["RefPoint"] == (j + 1))]
        perf_measure: pd.Series = df_tpr["PerformanceMeasure"]
        constraints_violated: pd.Series = df_tpr["ConstraintsViolated"]

        for k in range(len(df_tpr) - 1):
            if (
                perf_measure.iloc[0] > perf_measure.iloc[k + 1]
                and not constraints_violated.iloc[k + 1]
            ):
                print(
                    "problem with initial state {} and ref point {}".format(
                        i + 1, j + 1
                    )
                )
                break


# Boxplots for as described in the project report
vectorized_log = np.vectorize(np.log10)


def boxplot_across_exps(val: str, title: str, id: str):
    index = pd.MultiIndex.from_tuples(
        [
            t
            for t in itertools.product(
                range(1, 6), range(1, 4), ["rrlb", "reg"], [25, 50, 100, 250]
            )
        ],
        names=[
            "InitialState",
            "RefPoint",
            "Scheme",
            "HorizonSize",
        ],
    )
    df_plot = pd.DataFrame(
        vectorized_log(df[df["Scheme"] != "infhorz"][val].to_numpy(dtype=float)),
        index=index,
    )
    group = df_plot.groupby(["Scheme", "HorizonSize"])
    plt.figure(figsize=(10, 5))
    group.boxplot(subplots=False)
    # plt.suptitle(title)
    plt.savefig("exp_all_{}.png".format(id), dpi=300)


# plot 1 : N on x axis, iterations to convergence on y axis, boxplots with the nbr of iterations across the experiments
for i in range(nbr_initial_states):
    for j in range(3):
        df_tpr = df[(df["InitialState"] == (i + 1)) & (df["RefPoint"] == (j + 1))]
        df_tpr_rrlb = df_tpr[df_tpr["Scheme"] == "rrlb"]
        df_tpr_reg = df_tpr[df_tpr["Scheme"] == "reg"]

        assert len(df_tpr_rrlb) == len(df_tpr_reg)
        for k in range(len(df_tpr_rrlb)):
            if (
                df_tpr_rrlb["NbrIterationsToConvergence"].iloc[k] != 350
                and df_tpr_reg["NbrIterationsToConvergence"].iloc[k] != 350
                and df_tpr_rrlb["NbrIterationsToConvergence"].iloc[k]
                > df_tpr_reg["NbrIterationsToConvergence"].iloc[k]
            ):
                print(
                    "reg better nbr of iterations than rrlb for initial state {} and ref point {} at {}".format(
                        i + 1, j + 1, k
                    )
                )
                break
        for k in range(len(df_tpr_rrlb)):
            if (
                df_tpr_rrlb["NbrIterationsToConvergence"].iloc[k] != 350
                and df_tpr_reg["NbrIterationsToConvergence"].iloc[k] != 350
                and df_tpr_rrlb["PerformanceMeasure"].iloc[k]
                > df_tpr_reg["PerformanceMeasure"].iloc[k]
            ):
                print(
                    "reg better perf measure than rrlb for initial state {} and ref point {} at {}".format(
                        i + 1, j + 1, k
                    )
                )
                break


# plot 2 : N on x axis, performance measure on y axis


# plot 3 : N on x axis, runtime on y axis
boxplot_across_exps(
    val="AverageSolvingTime",
    title="Solving Time as a function of horizon size",
    id="plot_3_1",
)
boxplot_across_exps(
    val="AverageCondensationTime",
    title="Condensation Time as a function of horizon size",
    id="plot_3_2",
)

boxplot_across_exps(
    val="AverageSensitivitiesComputationTime",
    title="Sensitivity Computation Time as a function of horizon size",
    id="plot_3_3",
)

plt.close("all")
plt.show()
