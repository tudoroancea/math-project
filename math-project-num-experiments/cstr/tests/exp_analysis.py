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

# analyse the times : compute the averages by scheme and horizon size
df_avg = (
    df[df["Scheme"] != "infhorz"]
    .drop(
        columns=[
            "InitialState",
            "RefPoint",
            "NbrIterationsToConvergence",
            "ConstraintsViolated",
            "PerformanceMeasure",
        ]
    )
    .groupby(["Scheme", "HorizonSize"])
    .mean()
)
df_avg.to_csv("exp_all_results_times_avg.csv")


# Boxplots for as described in the project report
vectorized_log = np.vectorize(np.log10)


def boxplot_across_exps(val: str, title: str, scheme: str):
    index = pd.MultiIndex.from_tuples(
        [t for t in itertools.product(range(1, 6), range(1, 4), [25, 50, 100, 250])],
        names=[
            "InitialState",
            "RefPoint",
            "HorizonSize",
        ],
    )
    df_plot = pd.DataFrame(
        vectorized_log(df[df["Scheme"] == scheme][val].to_numpy(dtype=float)),
        index=index,
    )
    group = df_plot.groupby(["HorizonSize"])
    plt.figure(figsize=(7, 5))
    group.boxplot(subplots=False)
    plt.suptitle(title)
    id = title[: title.find(":")].strip().replace(" ", "_").replace(".", "_").lower()
    plt.savefig("exp_all_{}.png".format(id), dpi=300)


# plot 1 : N on x axis, iterations to convergence on y axis, boxplots with the nbr of iterations across the experiments
boxplot_across_exps(
    val="NbrIterationsToConvergence",
    title="Plot 1 : Nbr iterations to convergence in function of horizon size",
    scheme="rrlb",
)
boxplot_across_exps(
    val="NbrIterationsToConvergence",
    title="Plot 1.2 : Nbr iterations to convergence in function of horizon size for reg",
    scheme="reg",
)

# plot 2 : N on x axis, performance measure on y axis
boxplot_across_exps(
    val="PerformanceMeasure",
    title="Plot 2.1 : Performance measure in function of horizon size for RRLB",
    scheme="rrlb",
)
boxplot_across_exps(
    val="PerformanceMeasure",
    title="Plot 2.2 : Performance measure in function of horizon size for Reg",
    scheme="reg",
)

# plot 3 : N on x axis, runtime on y axis
boxplot_across_exps(
    val="AverageSolvingTime",
    title="Plot 3.1.1 : Solving Time in function of horizon size for RRLB",
    scheme="rrlb",
)
boxplot_across_exps(
    val="AverageSolvingTime",
    title="Plot 3.1.2 : Solving Time in function of horizon size for Reg",
    scheme="reg",
)
boxplot_across_exps(
    val="AverageCondensationTime",
    title="Plot 3.2.1 : Condensation Time in function of horizon size for RRLB",
    scheme="rrlb",
)
boxplot_across_exps(
    val="AverageCondensationTime",
    title="Plot 3.2.2 : Condensation Time in function of horizon size for Reg",
    scheme="reg",
)
boxplot_across_exps(
    val="AverageSensitivitiesComputationTime",
    title="Plot 3.3.1 : Sensitivity Computation Time in function of horizon size for RRLB",
    scheme="rrlb",
)
boxplot_across_exps(
    val="AverageSensitivitiesComputationTime",
    title="Plot 3.3.2 : Sensitivity Computation Time in function of horizon size for Reg",
    scheme="reg",
)

plt.close("all")
plt.show()
