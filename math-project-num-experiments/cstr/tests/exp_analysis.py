import pandas as pd

from exp_all import nbr_initial_states

# load the csv file with the results
df = pd.read_csv("exp_all_results.csv")

# See if there are any aberrant data
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
