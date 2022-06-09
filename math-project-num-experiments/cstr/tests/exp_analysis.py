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
# ======================================================================================
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
# ======================================================================================
vectorized_log = np.vectorize(np.log10)


def boxplot_across_exps(val: str, id: str, title: str = None):
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
    if title is not None:
        plt.suptitle(title)
    plt.savefig("exp_all_{}.png".format(id), dpi=300)


# N on x axis, runtime on y axis
boxplot_across_exps(
    val="AverageSolvingTime",
    id="plot_runtimes_1",
    # title="Solving Time as a function of horizon size",
)
boxplot_across_exps(
    val="AverageCondensationTime",
    id="plot_runtimes_2",
    # title="Condensation Time as a function of horizon size",
)

boxplot_across_exps(
    val="AverageSensitivitiesComputationTime",
    id="plot_runtimes_3",
    # title="Sensitivity Computation Time as a function of horizon size",
)

# see if RRLB is indeed always better in nbr of iterations to convergence ad performance
# measure (when it does converge)
# ======================================================================================
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

# compute the percentage of amelioration for horizon sizes 100 and 250
# ignore the situations (1,1), (1,3), (2,1), (2,3), (3,1), (4,1), (5,1)
tuples = [
    t
    for t in itertools.product(range(1, 6), range(1, 4), [100, 250])
    if t[0:2] not in [(1, 1), (1, 3), (2, 1), (2, 3), (3, 1), (4, 1), (5, 1)]
]
index = pd.MultiIndex.from_tuples(
    tuples,
    names=[
        "InitialState",
        "RefPoint",
        "HorizonSize",
    ],
)
tpr_reg = (
    df[
        (df["Scheme"] == "reg")
        & ((df["HorizonSize"] == 100) | (df["HorizonSize"] == 250))
        & ~(
            (df["InitialState"] == 1) & (df["RefPoint"] == 1)
            | (df["InitialState"] == 1) & (df["RefPoint"] == 3)
            | (df["InitialState"] == 2) & (df["RefPoint"] == 1)
            | (df["InitialState"] == 2) & (df["RefPoint"] == 3)
            | (df["InitialState"] == 3) & (df["RefPoint"] == 1)
            | (df["InitialState"] == 4) & (df["RefPoint"] == 1)
            | (df["InitialState"] == 5) & (df["RefPoint"] == 1)
        )
    ]
    .drop(
        columns=[
            "InitialState",
            "RefPoint",
            "Scheme",
            "HorizonSize",
            "ConstraintsViolated",
            "AverageCondensationTime",
            "StdErrCondensationTime",
            "AverageSensitivitiesComputationTime",
            "StdErrSensitivitiesComputationTime",
            "AverageSolvingTime",
            "StdErrSolvingTime",
        ]
    )
    .to_numpy()
)
tpr_rrlb = (
    df[
        (df["Scheme"] == "rrlb")
        & ((df["HorizonSize"] == 100) | (df["HorizonSize"] == 250))
        & ~(
            (df["InitialState"] == 1) & (df["RefPoint"] == 1)
            | (df["InitialState"] == 1) & (df["RefPoint"] == 3)
            | (df["InitialState"] == 2) & (df["RefPoint"] == 1)
            | (df["InitialState"] == 2) & (df["RefPoint"] == 3)
            | (df["InitialState"] == 3) & (df["RefPoint"] == 1)
            | (df["InitialState"] == 4) & (df["RefPoint"] == 1)
            | (df["InitialState"] == 5) & (df["RefPoint"] == 1)
        )
    ]
    .drop(
        columns=[
            "InitialState",
            "RefPoint",
            "Scheme",
            "HorizonSize",
            "ConstraintsViolated",
            "AverageCondensationTime",
            "StdErrCondensationTime",
            "AverageSensitivitiesComputationTime",
            "StdErrSensitivitiesComputationTime",
            "AverageSolvingTime",
            "StdErrSolvingTime",
        ]
    )
    .to_numpy()
)
improvement_percentages = pd.DataFrame(
    100.0 * (tpr_reg - tpr_rrlb) / tpr_reg,
    index=index,
    columns=["NbrIterationsToConvergence", "PerformanceMeasure"],
)
grouped_improvement_percentages = improvement_percentages.groupby(
    ["HorizonSize"]
).mean()
print(grouped_improvement_percentages)
grouped_improvement_percentages.to_csv("exp_all_improvement_percentages.csv")
grouped_improvement_percentages.plot(kind="bar")
plt.ylim([0.0, 45.0])
plt.grid(True)
plt.savefig("exp_all_improvement_percentages.png", dpi=300)


plt.show()
