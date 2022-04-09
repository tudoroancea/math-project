# %% imports and global parameters
import os
import sys
from time import time
import casadi
import numpy as np
from math import inf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

T = 2000.0  # Time horizon
N = 100  # number of control intervals
M = 1  # RK4 integrations steps
delta = 10.0  # relaxation parameter
epsilon = 0.05  # barrier parameter
x_S = casadi.DM(
    [
        2.1402105301746182e00,
        1.0903043613077321e00,
        1.1419108442079495e02,
        1.1290659291045561e02,
    ]
)
u_S = casadi.DM([14.19, -1113.50])
simulation_length = 150  # iterations
x_init = [1.0, 0.5, 100.0, 100.0]


# %% Declare model
x = casadi.SX.sym("x", 4)
u = casadi.SX.sym("u", 2)
k_10 = 1.287e12  # [h^-1]
k_20 = 1.287e12  # [h^-1]
k_30 = 9.043e9  # [h^-1]
E_1 = -9758.3  # [1]
E_2 = -9758.3  # [1]
E_3 = -8560.0  # [1]
H_1 = 4.2  # [kJ/mol]
H_2 = -11.0  # [kJ/mol]
H_3 = -41.85  # [kJ/mol]
rho = 0.9342  # [kg/l]
C_p = 3.01  # [kJ/(kg.K)]
k_w = 4032.0  # [kJ/(h.m^2.K)]
A_R = 0.215  # [m^2]
V_R = 10.0  # [l]
m_K = 5.0  # [kg]
C_PK = 2.0  # [kJ/(kg.K)]
c_A0 = 5.1  # [mol/l]
theta_0 = 104.9  # [°C]

k_1 = k_10 * casadi.exp(E_1 / (x[2] + 273.15))
k_2 = k_20 * casadi.exp(E_2 / (x[2] + 273.15))
k_3 = k_30 * casadi.exp(E_3 / (x[2] + 273.15))

f_cont = casadi.Function(
    "f_cont",
    [x, u],
    [
        casadi.vertcat(
            u[0] * (c_A0 - x[0]) - k_1 * x[0] - k_3 * x[0] * x[0],
            -u[0] * x[1] + k_1 * x[0] - k_2 * x[1],
            u[0] * (theta_0 - x[2])
            + (x[3] - x[2]) * k_w * A_R / (rho * C_p * V_R)
            - (k_1 * x[0] * H_1 + k_2 * x[1] * H_2 + k_3 * x[0] * x[0] * H_3)
            / (rho * C_p),
            (u[1] + k_w * A_R * (x[2] - x[3])) / (m_K * C_PK),
        )
        / 3600.0
    ],
)

DT = T / N / M
new_x = x
for i in range(M):
    k1 = f_cont(new_x, u)
    k2 = f_cont(new_x + DT / 2 * k1, u)
    k3 = f_cont(new_x + DT / 2 * k2, u)
    k4 = f_cont(new_x + DT * k3, u)
    new_x = new_x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
f_discrete = casadi.Function("f_discrete", [x, u], [new_x])

f_discrete_linearized = casadi.Function(
    "f_discrete_linearized",
    [x, u],
    [casadi.jacobian(f_discrete(x, u), x), casadi.jacobian(f_discrete(x, u), u)],
)
(A, B) = f_discrete_linearized(x_S, u_S)
print("A = ", A)
print("B = ", B)


# %% barrier functions for state and control constraints
z = casadi.SX.sym("z", 1)
beta = casadi.Function(
    "beta", [z], [0.5 * (((z - 2 * delta) / delta) ** 2 - 1) - casadi.log(delta)]
)
B_x_1 = casadi.if_else(
    x[2] - 98.0 > delta,
    casadi.log(x_S[2] - 98.0) - casadi.log(x[2] - 98.0),
    beta(x[2] - 98.0),
)
B_x_2 = casadi.if_else(
    x[3] - 92.0 > delta,
    casadi.log(x_S[3] - 92.0) - casadi.log(x[3] - 92.0),
    beta(x[3] - 92.0),
)
grad_B_x_1 = casadi.Function("grad_B_x_1", [x], [casadi.jacobian(B_x_1, x)])
grad_B_x_2 = casadi.Function("grad_B_x_2", [x], [casadi.jacobian(B_x_2, x)])
w_x_1 = -1.0
w_x_2 = -1.0
B_x = casadi.Function(
    "B_x",
    [x],
    [(1.0 + w_x_1) * B_x_1 + (1.0 + w_x_2) * B_x_2],
)
grad_B_x = casadi.Function("grad_B_x", [x], [casadi.jacobian(B_x(x), x)])
hess_B_x = casadi.Function("hess_B_x", [x], [casadi.hessian(B_x(x), x)[0]])
print("grad_B_x = {}, norm = {}".format(grad_B_x(x_S), casadi.norm_2(grad_B_x(x_S))))
M_x = hess_B_x(x_S)


B_u_1 = casadi.if_else(
    35.0 - u[0] > delta, casadi.log(20.81) - casadi.log(35.0 - u[0]), beta(35.0 - u[0])
)
B_u_2 = casadi.if_else(
    u[0] - 3.0 > delta, casadi.log(11.19) - casadi.log(u[0] - 3.0), beta(u[0] - 3.0)
)
B_u_3 = casadi.if_else(
    -u[1] > delta, casadi.log(1113.5) - casadi.log(-u[1]), beta(-u[1])
)
B_u_4 = casadi.if_else(
    9000.0 + u[1] > delta,
    casadi.log(7886.5) - casadi.log(9000.0 + u[1]),
    beta(9000.0 + u[1]),
)
grad_B_u_1 = casadi.Function("grad_B_u_1", [u], [casadi.jacobian(B_u_1, u)])
grad_B_u_2 = casadi.Function("grad_B_u_2", [u], [casadi.jacobian(B_u_2, u)])
grad_B_u_3 = casadi.Function("grad_B_u_3", [u], [casadi.jacobian(B_u_3, u)])
grad_B_u_4 = casadi.Function("grad_B_u_4", [u], [casadi.jacobian(B_u_4, u)])
w_u_1 = 0.0
w_u_2 = -grad_B_u_1(u_S)[0] / grad_B_u_2(u_S)[0] - 1.0
w_u_3 = 0.0
w_u_4 = -grad_B_u_3(u_S)[1] / grad_B_u_4(u_S)[1] - 1.0

B_u = casadi.Function(
    "B_u",
    [u],
    [
        (1.0 + w_u_1) * B_u_1
        + (1.0 + w_u_2) * B_u_2
        + (1.0 + w_u_3) * B_u_3
        + (1.0 + w_u_4) * B_u_4
    ],
)
grad_B_u = casadi.Function("grad_B_u", [u], [casadi.jacobian(B_u(u), u)])
hess_B_u = casadi.Function("hess_B_u", [u], [casadi.hessian(B_u(u), u)[0]])
print("grad_B_u = {}, norm = {}".format(grad_B_u(u_S), casadi.norm_2(grad_B_u(u_S))))
M_u = hess_B_u(u_S)


# %% stage cost
l = casadi.Function(
    "l",
    [x, u],
    [
        0.2 * (x[0] - x_S[0]) ** 2
        + 1.0 * (x[1] - x_S[1]) ** 2
        + 0.5 * (x[2] - x_S[2]) ** 2
        + 0.2 * (x[3] - x_S[3]) ** 2
        + 0.5 * (u[0] - u_S[0]) ** 2
        + 5.0e-7 * (u[1] - u_S[1]) ** 2
    ],
)
l_tilde = casadi.Function(
    "l_tilde",
    [x, u],
    [l(x, u) + epsilon * B_x(x) + epsilon * B_u(u)],
)
Q = casadi.diag(casadi.DM([0.2, 1.0, 0.5, 0.2]))
R = casadi.diag(casadi.DM([0.5, 5.0e-7]))

# %% terminal cost (determine P)
P = casadi.MX.sym("P", 4, 4)
riccati = casadi.Function(
    "riccati",
    [P],
    [
        A.T * P * A
        - A.T @ P @ B @ casadi.inv(R + epsilon * M_u + B.T @ P @ B) @ B.T @ P @ A
        + Q
        + epsilon * M_x
    ],
)

P = casadi.DM.eye(4)
for i in range(10000):
    assert i < 9999, "P not converged : {}".format(casadi.norm_fro(new_P - P))
    new_P = riccati(P)
    if casadi.norm_fro(new_P - P) < 1.0e-6:
        print("P converged at iteration : ", i)
        break
    P = new_P
    if not P.is_regular():
        print("P contains Inf or NaN at iteration {} :".format(i))
        break

# P = casadi.DM(
#     [
#         [
#             1.4646778374584373,
#             0.6676889516721198,
#             0.35446715117028615,
#             0.10324422005086348,
#         ],
#         [
#             0.6676889516721198,
#             1.407812935783267,
#             0.17788030743777067,
#             0.050059833257226405,
#         ],
#         [
#             0.35446715117028615,
#             0.1778803074377706,
#             0.6336052592712396,
#             0.01110329497282364,
#         ],
#         [
#             0.10324422005086348,
#             0.05005983325722643,
#             0.011103294972823655,
#             0.229412393739723,
#         ],
#     ]
# )
print("P = ", P)


F = casadi.Function("F", [x], [casadi.bilin(P, (x - x_S), (x - x_S))])
K = -casadi.inv(R + epsilon * M_u + B.T @ P @ B) @ B.T @ P @ A

# %% Create compute_control function
def solve_rrlb_mpc(
    x_init: np.ndarray, x_start: list[casadi.DM], u_start: list[casadi.DM]
) -> casadi.DM:
    """
    returns the optimal solution with all the variables from all the stages
    """
    x_0 = casadi.MX.sym("x_0", 4)
    x_k = x_0

    J = 0  # objective function
    w = [x_0]  # list of all the variables x and u concatenated
    w_start = [x_start[0]]  # initial guess
    lbw = []
    lbw += x_init.tolist()  # lower bound for all the variables
    ubw = []
    ubw += x_init.tolist()  # upper bound for all the variables
    g = []  # equality constraints
    lbg = []  # lower bound for the equality constraints
    ubg = []  # upper bound for the equality constraints

    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        u_k = casadi.MX.sym("u_" + str(k), 2)
        w += [u_k]
        w_start += [u_start[k]]

        x_k_end = f_discrete(x_k, u_k)
        J = J + l_tilde(x_k, u_k)

        x_k = casadi.MX.sym(
            "x_" + str(k + 1), 4
        )  # WARNING : from here x_k represents x_{k+1}
        w += [x_k]
        w_start += [x_start[k + 1]]
        lbw += [-inf, -inf, -inf, -inf, -inf, -inf]
        ubw += [inf, inf, inf, inf, inf, inf]

        # Add equality constraint
        g += [x_k_end - x_k]
        lbg += [0.0, 0.0, 0.0, 0.0]
        ubg += [0.0, 0.0, 0.0, 0.0]
    J += F(x_k)  # here x_k represents x_N

    # Concatenate decision variables and constraint terms
    w = casadi.vertcat(*w)
    g = casadi.vertcat(*g)

    # Create an NLP solver
    nlp_prob = {"f": J, "x": w, "g": g}
    nlp_solver = casadi.nlpsol(
        "nlp_solver",
        "ipopt",
        nlp_prob,
        {
            "print_time": 0,
            "ipopt": {
                "print_level": 0,
                "max_iter": 1,
                "max_cpu_time": 10.0,
            },
        },
    )
    sol = nlp_solver(x0=casadi.vertcat(*w_start), lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    return sol["x"]


def solve_mpc(
    x_init: np.ndarray, x_start: list[casadi.DM], u_start: list[casadi.DM]
) -> casadi.DM:
    """
    returns the optimal solution with all the variables from all the stages
    """
    x_0 = casadi.MX.sym("x_0", 4)
    x_k = x_0

    J = 0  # objective function
    w = [x_0]  # list of all the variables x and u concatenated
    w_start = [x_start[0]]  # initial guess
    lbw = []
    lbw += x_init.tolist()  # lower bound for all the variables
    ubw = []
    ubw += x_init.tolist()  # upper bound for all the variables
    g = []  # equality constraints
    lbg = []  # lower bound for the equality constraints
    ubg = []  # upper bound for the equality constraints

    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        u_k = casadi.MX.sym("u_" + str(k), 2)
        w += [u_k]
        w_start += [u_start[k]]

        x_k_end = f_discrete(x_k, u_k)
        J = J + l(x_k, u_k)

        x_k = casadi.MX.sym(
            "x_" + str(k + 1), 4
        )  # WARNING : from here x_k represents x_{k+1}
        w += [x_k]
        w_start += [x_start[k + 1]]
        lbw += [3.0, -9000.0, -inf, -inf, 98.0, 92.0]
        ubw += [35.0, 0.0, inf, inf, inf, inf]

        # Add equality constraint
        g += [x_k_end - x_k]
        lbg += [0.0, 0.0, 0.0, 0.0]
        ubg += [0.0, 0.0, 0.0, 0.0]
    # J += F(x_k)  # here x_k represents x_N

    # Concatenate decision variables and constraint terms
    w = casadi.vertcat(*w)
    g = casadi.vertcat(*g)

    # Create an NLP solver
    nlp_prob = {"f": J, "x": w, "g": g}
    nlp_solver = casadi.nlpsol(
        "nlp_solver",
        "ipopt",
        nlp_prob,
        {
            "print_time": 0,
            "ipopt": {
                "print_level": 0,
                "max_iter": 1,
                "max_cpu_time": 10.0,
            },
        },
    )
    sol = nlp_solver(x0=casadi.vertcat(*w_start), lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    return sol["x"]


# %% solve the problem in a closed loop fashion
def createPlot(
    states,
    controls,
    state_prediction,
    control_prediction,
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
    ax_concentration.plot([0, simulation_length - 1], [x_S[0], x_S[0]], "b-.")
    ax_concentration.plot([0, simulation_length - 1], [x_S[1], x_S[1]], "m-.")
    if not final:
        (c_A,) = ax_concentration.plot(states[0, 0], "b-")
        (c_B,) = ax_concentration.plot(states[1, 0], "m-")
    else:
        (c_A,) = ax_concentration.plot(states[0, :], "b-")
        (c_B,) = ax_concentration.plot(states[1, :], "m-")
    ax_concentration.plot(state_prediction[0, :], "b--")
    ax_concentration.plot(state_prediction[1, :], "m--")
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
    ax_temperatures.plot([0, simulation_length - 1], [x_S[2], x_S[2]], "b-.")
    ax_temperatures.plot([0, simulation_length - 1], [x_S[3], x_S[3]], "m-.")
    if not final:
        (theta,) = ax_temperatures.plot(states[2, 0], "b-")
        (theta_K,) = ax_temperatures.plot(states[3, 0], "m-")
    else:
        (theta,) = ax_temperatures.plot(states[2, :], "b-")
        (theta_K,) = ax_temperatures.plot(states[3, :], "m-")
    ax_temperatures.plot(state_prediction[2, :], "b--")
    ax_temperatures.plot(state_prediction[3, :], "m--")
    ax_temperatures.legend(
        [theta, theta_K],
        ["theta", "theta_K"],
        loc="upper right",
    )

    # Plot feed inflow
    ax_feed_inflow = fig.add_subplot(gs[2, 0])
    ax_feed_inflow.grid("both")
    ax_feed_inflow.set_title("evolution of feed inflow rate")
    ax_feed_inflow.set_xlabel("time [h]")
    ax_feed_inflow.set_ylabel("feed inflow rate [h^-1]")
    ax_feed_inflow.plot([0, simulation_length - 1], [u_S[0], u_S[0]], "g-.")
    ax_feed_inflow.plot([0, simulation_length - 1], [3.0, 3.0], "r:")
    ax_feed_inflow.plot([0, simulation_length - 1], [35.0, 35.0], "r:")
    if not final:
        ax_feed_inflow.plot(controls[0, 0], "g-")
    else:
        ax_feed_inflow.plot(controls[0, :], "g-")
    ax_feed_inflow.plot(control_prediction[0, :], "g--")

    # Plot heat removal
    ax_heat_removal = fig.add_subplot(gs[3, 0])
    ax_heat_removal.grid("both")
    ax_heat_removal.set_title("evolution of heat removal rate")
    ax_heat_removal.set_xlabel("time [h]")
    ax_heat_removal.set_ylabel("heat removal rate [kJ/h]")
    ax_heat_removal.plot([0, simulation_length - 1], [u_S[1], u_S[1]], "g-.")
    ax_heat_removal.plot([0, simulation_length - 1], [-9000.0, -9000.0], "r:")
    ax_heat_removal.plot([0, simulation_length - 1], [0.0, 0.0], "r:")
    if not final:
        ax_heat_removal.plot(controls[1, 0], "g-")
    else:
        ax_heat_removal.plot(controls[1, :], "g-")
    ax_heat_removal.plot(control_prediction[1, :], "g--")

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


def run_closed_loop_simulation(
    simulation_length: int, method, step_by_step: bool, name: str = "rrlb_mpc"
):
    states = np.zeros((4, simulation_length + 1))
    controls = np.zeros((2, simulation_length))
    states[:, 0] = x_init

    # Get a feasible trajectory as an initial guess (for the plot creation)
    u_start = casadi.horzcat(*([u_S] * N))
    x_start = casadi.DM.zeros(4, N + 1)
    x_start[:, 0] = x_init
    for k in range(N):
        x_start[:, k + 1] = f_discrete(x_start[:, k], u_start[k])

    if step_by_step:
        createPlot(
            states,
            controls,
            x_start,
            u_start,
        )

    # transform the initial guess into a list of DMs for the solver
    u_start = [u_S] * N
    x_start_k = casadi.DM(x_init)
    x_start = [x_start_k]
    for k in range(N):
        x_start_k = f_discrete(x_start_k, u_start[k])
        x_start += [x_start_k]

    for i in range(simulation_length):
        # retrieve solution data
        start = time()
        sol = method(states[:, i], x_start=x_start, u_start=u_start)
        stop = time()
        print("time for iteration {} : {}".format(i, 1000*(stop - start)))
        c_A_pred = sol[0::6]
        c_B_pred = sol[1::6]
        theta_pred = sol[2::6]
        theta_K_pred = sol[3::6]
        u_1_pred = sol[4::6]
        u_2_pred = sol[5::6]

        # update the states and controls
        controls[:, i] = [u_1_pred[0], u_2_pred[0]]
        states[:, i + 1] = [c_A_pred[1], c_B_pred[1], theta_pred[1], theta_K_pred[1]]
        states_pred = casadi.horzcat(c_A_pred, c_B_pred, theta_pred, theta_K_pred).T
        controls_pred = casadi.horzcat(u_1_pred, u_2_pred).T

        # update the initial guess for next solve
        for k in range(N - 2):
            u_start[k] = casadi.DM([u_1_pred[k + 1], u_2_pred[k + 1]])
            x_start[k] = casadi.DM(
                [
                    c_A_pred[k + 1],
                    c_B_pred[k + 1],
                    theta_pred[k + 1],
                    theta_K_pred[k + 1],
                ]
            )
        u_start[N - 1] = -K @ (x_start[N - 1] - x_S)
        x_start[N] = f_discrete(x_start[N - 1], u_start[N - 1])

        # plot the simulation data if need be
        if step_by_step:
            updatePlots(
                states,
                controls,
                states_pred,
                controls_pred,
                i,
            )
            if i == simulation_length - 1:
                print("Final state: ", states[:, i + 1])
                print("Final control: ", controls[:, i])
                # print("all controls: ", controls)
                plt.show()
            else:
                plt.draw()
        else:
            if i == simulation_length - 1:
                print("Final state: ", states[:, i + 1])
                print("Final control: ", controls[:, i])
                createPlot(states, controls, states_pred, controls_pred, final=True)
                plt.savefig(os.path.join(os.path.dirname(__file__), name + ".png"), dpi=300, format="png")


if __name__ == "__main__":
    # run_closed_loop_simulation(
    #     simulation_length=simulation_length,
    #     method=solve_rrlb_mpc,
    #     step_by_step=True,
    # )
    sys.stderr.write("RRLB MPC : \n==========================")
    run_closed_loop_simulation(
        simulation_length=simulation_length,
        method=solve_rrlb_mpc,
        step_by_step=False,
        name="rrlb_mpc",
    )
    sys.stderr.write("MPC : \n==========================")
    run_closed_loop_simulation(
        simulation_length=simulation_length,
        method=solve_mpc,
        step_by_step=False,
        name="mpc",
    )

# %%
