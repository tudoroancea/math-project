import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def createPlot(
    states,
    controls,
    state_prediction,
    control_prediction,
    simulation_length,
    xr,
    ur,
    N,
    final=False,
):
    """Creates a plot and adds the initial data provided by the arguments"""

    # Create empty plot
    fig = plt.figure(figsize=(15, 7))
    plt.clf()
    gs = GridSpec(4, 1, figure=fig)

    # Plot concentrations
    ax_concentration = fig.add_subplot(gs[0, 0])
    ax_concentration.grid("both")
    ax_concentration.set_title("evolution of concentrations over time")
    plt.ylim([0.0, 4.0])
    ax_concentration.set_xlabel("time [h]")
    ax_concentration.set_ylabel("concentration [mol/L]")
    ax_concentration.plot(
        [0, simulation_length - 1], [float(xr[0]), float(xr[0])], "r"
    )
    ax_concentration.plot(
        [0, simulation_length - 1], [float(xr[1]), float(xr[1])], "r"
    )
    if not final:
        (c_A,) = ax_concentration.plot(states[0, 0], "b-")
        (c_B,) = ax_concentration.plot(states[1, 0], "m-")
        ax_concentration.plot(state_prediction[0, :], "b--")
        ax_concentration.plot(state_prediction[1, :], "m--")
    else:
        (c_A,) = ax_concentration.plot(states[0, :], "b-")
        (c_B,) = ax_concentration.plot(states[1, :], "m-")
        ax_concentration.plot(
            range(states.shape[1] - 1, states.shape[1] + N),
            state_prediction[0, :],
            "b--",
        )
        ax_concentration.plot(
            range(states.shape[1] - 1, states.shape[1] + N),
            state_prediction[1, :],
            "m--",
        )
    ax_concentration.legend(
        [c_A, c_B],
        ["c_A", "c_B"],
        loc="upper right",
    )

    # Plot temperatures
    ax_temperatures = fig.add_subplot(gs[1, 0])
    ax_temperatures.grid("both")
    ax_temperatures.set_title("evolution of temperatures over time")
    ax_temperatures.set_xlabel("time [h]")
    ax_temperatures.set_ylabel("temperature [°C]")
    plt.ylim([95.0, 120.0])
    ax_temperatures.plot(
        [0, simulation_length - 1], [float(xr[2]), float(xr[2])], "r"
    )
    ax_temperatures.plot(
        [0, simulation_length - 1], [float(xr[3]), float(xr[3])], "r"
    )
    if not final:
        (theta,) = ax_temperatures.plot(states[2, 0], "b-")
        (theta_K,) = ax_temperatures.plot(states[3, 0], "m-")
        ax_temperatures.plot(state_prediction[2, :], "b--")
        ax_temperatures.plot(state_prediction[3, :], "m--")
    else:
        (theta,) = ax_temperatures.plot(states[2, :], "b-")
        (theta_K,) = ax_temperatures.plot(states[3, :], "m-")
        ax_temperatures.plot(
            range(states.shape[1] - 1, states.shape[1] + N),
            state_prediction[2, :],
            "b--",
        )
        ax_temperatures.plot(
            range(states.shape[1] - 1, states.shape[1] + N),
            state_prediction[3, :],
            "m--",
        )
    ax_temperatures.legend(
        [theta, theta_K],
        ["theta", "theta_K"],
        loc="upper right",
    )

    # Plot feed inflow
    ax_feed_inflow = fig.add_subplot(gs[2, 0])
    plt.ylim([0.0, 39.0])
    ax_feed_inflow.grid("both")
    ax_feed_inflow.set_title("evolution of feed inflow rate")
    ax_feed_inflow.set_xlabel("time [h]")
    ax_feed_inflow.set_ylabel("feed inflow rate [h^-1]")
    ax_feed_inflow.plot([0, simulation_length - 1], [float(ur[0]), float(ur[0])], "r")
    ax_feed_inflow.plot([0, simulation_length - 1], [3.0, 3.0], "r:")
    ax_feed_inflow.plot([0, simulation_length - 1], [35.0, 35.0], "r:")
    if not final:
        ax_feed_inflow.step(controls[0, 0], "g-")
        ax_feed_inflow.step(
            range(controls.shape[1] - 1, controls.shape[1] - 1 + N),
            control_prediction[0, :].T,
            "g--",
        )
    else:
        ax_feed_inflow.step(controls[0, :], "g-")
        ax_feed_inflow.step(
            range(controls.shape[1] - 1, controls.shape[1] - 1 + N),
            control_prediction[0, :].T,
            "g--",
        )

    # Plot heat removal
    ax_heat_removal = fig.add_subplot(gs[3, 0])
    ax_heat_removal.grid("both")
    plt.ylim([-9900.0, 900.0])
    ax_heat_removal.set_title("evolution of heat removal rate")
    ax_heat_removal.set_xlabel("time [h]")
    ax_heat_removal.set_ylabel("heat removal rate [kJ/h]")
    ax_heat_removal.plot(
        [0, simulation_length - 1], [float(ur[1]), float(ur[1])], "r"
    )
    ax_heat_removal.plot([0, simulation_length - 1], [-9000.0, -9000.0], "r:")
    ax_heat_removal.plot([0, simulation_length - 1], [0.0, 0.0], "r:")
    if not final:
        ax_heat_removal.step(controls[1, 0], "g-")
        ax_heat_removal.step(
            range(controls.shape[1] - 1, controls.shape[1] - 1 + N),
            control_prediction[1, :],
            "g--",
        )
    else:
        ax_heat_removal.step(controls[1, :], "g-")
        ax_heat_removal.step(
            range(controls.shape[1] - 1, controls.shape[1] - 1 + N),
            control_prediction[1, :],
            "g--",
        )

    plt.tight_layout()


def updatePlots(
    states,
    controls,
    state_predictions,
    control_predictions,
    iteration,
    pause_duration=0.05,
):
    fig = plt.gcf()
    [ax_concentration, ax_temperature, ax_feed_inflow, ax_heat_removal] = fig.axes

    # Delete old data in plots
    ax_concentration.get_lines().pop(-1).remove()
    ax_concentration.get_lines().pop(-1).remove()
    ax_concentration.get_lines().pop(-1).remove()
    ax_concentration.get_lines().pop(-1).remove()
    ax_temperature.get_lines().pop(-1).remove()
    ax_temperature.get_lines().pop(-1).remove()
    ax_temperature.get_lines().pop(-1).remove()
    ax_temperature.get_lines().pop(-1).remove()
    ax_feed_inflow.get_lines().pop(-1).remove()
    ax_feed_inflow.get_lines().pop(-1).remove()
    ax_heat_removal.get_lines().pop(-1).remove()
    ax_heat_removal.get_lines().pop(-1).remove()

    # Update plot with current simulation data
    ax_concentration.plot(range(0, iteration + 1), states[0, : iteration + 1], "b-")
    ax_concentration.plot(
        range(iteration, iteration + 1 + N), state_predictions[0, :].T, "b--"
    )
    ax_concentration.plot(range(0, iteration + 1), states[1, : iteration + 1], "m-")
    ax_concentration.plot(
        range(iteration, iteration + 1 + N), state_predictions[1, :].T, "m--"
    )

    ax_temperature.plot(range(0, iteration + 1), states[2, : iteration + 1], "b-")
    ax_temperature.plot(
        range(iteration, iteration + 1 + N), state_predictions[2, :].T, "b--"
    )
    ax_temperature.plot(range(0, iteration + 1), states[3, : iteration + 1], "m-")
    ax_temperature.plot(
        range(iteration, iteration + 1 + N), state_predictions[3, :].T, "m--"
    )

    ax_feed_inflow.step(range(0, iteration + 1), controls[0, : iteration + 1], "g-")
    ax_feed_inflow.step(
        range(iteration, iteration + N), control_predictions[0, :].T, "g--"
    )

    ax_heat_removal.step(range(0, iteration + 1), controls[1, : iteration + 1], "g-")
    ax_heat_removal.step(
        range(iteration, iteration + N), control_predictions[1, :].T, "g--"
    )

    plt.pause(pause_duration)