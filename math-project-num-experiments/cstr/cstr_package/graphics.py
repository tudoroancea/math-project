from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from . import CSTR


class CSTRAnimation:
    titles = [
        "c_A",
        "c_B",
        "theta",
        "theta_K",
        "Feed Inflow Rate",
        "Heat Removal Rate",
    ]
    xlabel = "Time [s]"
    ylabels = [
        "c_A [mol/L]",
        "c_B [mol/L]",
        "theta [°C]",
        "theta_K [°C]",
        "Feed Inflow rate [s^-1]",
        "Heat Removal rate [kJ/s]",
    ]
    axes = []

    def __init__(
        self,
        model: CSTR,
        reference_points: np.ndarray,
        initial_data: np.ndarray,
        initial_prediction: np.ndarray,
        times: np.ndarray,
    ):
        """
        params
        -------
        - reference_points : each row corresponds to a physical value (either state or \
            control) and each column to one reference point
        """
        # transform the initial prediction (in the form of a long vector with all the states and
        # contorls concatenated) into a matrix
        initial_prediction = np.concatenate(
            (
                np.reshape(
                    initial_prediction[: model.nx * (model.N + 1)],
                    (model.nx, model.N + 1),
                    order="F",
                ),
                np.concatenate(
                    (
                        np.reshape(
                            initial_prediction[-model.nu * model.N :],
                            (model.nu, model.N),
                            order="F",
                        ),
                        np.zeros((model.nu, 1)),
                    ),
                    axis=1,
                ),
            ),
            axis=0,
        )

        # Create empty plot
        fig = plt.gcf()
        plt.clf()
        gs = GridSpec(model.nx + model.nu, 1, figure=fig)

        self.axes = list(range(model.nx + model.nu))
        for i in range(model.nx + model.nu):
            self.axes[i] = fig.add_subplot(gs[i, 0])
            plt.title(self.titles[i])
            # plt.xlabel(self.xlabel)
            # plt.ylabel(self.ylabels[i])
            plt.grid("both")
            plt.xlim([0.0, np.sum(times) + model.T])

            if len(reference_points.shape) == 1 or reference_points.shape[1] == 1:
                plt.plot(
                    [0.0, 3000.0], [reference_points[i], reference_points[i]], "r-"
                )
            else:
                for j in range(reference_points.shape[1]):
                    plt.plot(
                        [3000.0 * j, 3000.0 * (j + 1)],
                        [reference_points[i, j], reference_points[i, j]],
                        "r-",
                    )

            if i == 2:
                plt.plot([0.0, np.sum(times) + model.T], [98.0, 98.0], "r:")
            elif i == 3:
                plt.plot([0.0, np.sum(times) + model.T], [92.0, 92.0], "r:")
            elif i == 4:
                plt.plot([0.0, np.sum(times) + model.T], [3.0, 3.0], "r:")
                plt.plot([0.0, np.sum(times) + model.T], [35.0, 35.0], "r:")
            elif i == 5:
                plt.plot([0.0, np.sum(times) + model.T], [-9000.0, -9000.0], "r:")
                plt.plot([0.0, np.sum(times) + model.T], [0.0, 0.0], "r:")

            if i < model.nx:
                plt.plot(np.cumsum(times), initial_data[i, :], "b-")

                plt.plot(
                    np.linspace(np.sum(times), np.sum(times) + model.T, model.N + 1),
                    initial_prediction[i, :],
                    "b--",
                )
            else:
                plt.step(np.cumsum(times), initial_data[i, :], "g-")

                plt.step(
                    np.linspace(np.sum(times), np.sum(times) + model.T, model.N),
                    initial_prediction[i, :-1],
                    "g--",
                )

        plt.tight_layout()

    def update_plot(self, data, prediction, times, iteration):
        """
        Update the plot with the new data and prediction.
        The data, prediction and times are all numpy arrays with pre-allocated space for the whole
        simulation, so not all the values are relevant at the call of this function, only those up
        until the index iteration.
        """
        for i in len(self.axes):
            # Delete old data in plots
            self.axes[i].get_lines().pop(-1).remove()
            self.axes[i].get_lines().pop(-1).remove()

            # plot new data
            if i < self.model.nx:
                self.axes[i].plot(np.cumsum(times), data[:, i], "b-")
                self.axes[i].plot(
                    np.linspace(
                        np.sum(times), np.sum(times) + self.model.T, self.model.N
                    ),
                    prediction[i, :],
                    "b--",
                )
            else:
                self.axes[i].step(np.cumsum(times), data[:, i], "g-")
                self.axes[i].step(
                    np.linspace(
                        np.sum(times), np.sum(times) + self.model.T, self.model.N
                    ),
                    prediction[i, :],
                    "g--",
                )