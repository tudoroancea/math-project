# %% imports and global parameters
import casadi
import numpy as np
from math import inf

T = 12.0  # Time horizon
N = 40  # number of control intervals
M = 4  # RK4 integrations steps
delta = 0.33  # relaxation parameter
epsilon = 0.05  # barrier parameter

# %% Declare model
x = casadi.SX.sym("x", 2)
u = casadi.SX.sym("u")
c0 = 0.4
c1 = 0.2
f_cont = casadi.Function(
    "f_cont",
    [x, u],
    [
        casadi.vertcat(
            x[0] - x[0] * x[1] - c0 * x[0] * u, -x[1] + x[0] * x[1] - c1 * x[1] * u
        )
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
(A, B) = f_discrete_linearized(casadi.DM([1.0, 1.0]), casadi.DM(0.0))
print("A = ", A)
print("B = ", B)


# %% barrier functions for state and control constraints
z = casadi.SX.sym("z", 1)
beta = casadi.Function(
    "beta", [z], [0.5 * (((z - 2 * delta) / delta) ** 2 - 1) - casadi.log(delta)]
)
B_x_1 = casadi.if_else(2 - x[0] > delta, -casadi.log(2 - x[0]), beta(2 - x[0]))
B_x_2 = casadi.if_else(2 - x[1] > delta, -casadi.log(2 - x[1]), beta(2 - x[1]))
B_x_3 = casadi.if_else(x[0] > delta, -casadi.log(x[0]), beta(x[0]))
B_x_4 = casadi.if_else(x[1] > delta, -casadi.log(x[1]), beta(x[1]))
grad_B_x_1 = casadi.Function("grad_B_x_1", [x], [casadi.jacobian(B_x_1, x)])
grad_B_x_2 = casadi.Function("grad_B_x_2", [x], [casadi.jacobian(B_x_2, x)])
grad_B_x_3 = casadi.Function("grad_B_x_3", [x], [casadi.jacobian(B_x_3, x)])
grad_B_x_4 = casadi.Function("grad_B_x_4", [x], [casadi.jacobian(B_x_4, x)])
w_x_1 = 0.0
w_x_2 = 0.0
w_x_3 = (
    -grad_B_x_1(casadi.DM([1.0, 1.0]))[0] / grad_B_x_3(casadi.DM([1.0, 1.0]))[0] - 1.0
)
w_x_4 = (
    -grad_B_x_2(casadi.DM([1.0, 1.0]))[1] / grad_B_x_4(casadi.DM([1.0, 1.0]))[1] - 1.0
)
B_x = casadi.Function(
    "B_x",
    [x],
    [
        (1.0 + w_x_1) * B_x_1
        + (1.0 + w_x_2) * B_x_2
        + (1.0 + w_x_3) * B_x_3
        + (1.0 + w_x_4) * B_x_4
    ],
)
grad_B_x = casadi.Function("grad_B_x", [x], [casadi.jacobian(B_x(x), x)])
hess_B_x = casadi.Function("hess_B_x", [x], [casadi.hessian(B_x(x), x)[0]])
print("grad_B_x = ", grad_B_x(casadi.DM.ones(2)))
assert casadi.norm_inf(grad_B_x(casadi.DM([1.0, 1.0]))) < 1.0e-10, "grad_B_x(x) != 0"

M_x = casadi.DM(
    [[2.0 + float(w_x_1) + float(w_x_3), 0.0], [0.0, 2.0 + float(w_x_2) + float(w_x_4)]]
) / (2.0 * delta * delta)
M_tilde_x = hess_B_x(casadi.DM([1.0, 1.0]))


B_u_1 = casadi.if_else(1 - u > delta, -casadi.log(1 - u), beta(1 - u))
# B_u_2 = casadi.if_else(
#     1.0e-6 + u > delta,
#     casadi.log(casadi.DM(1.0e-6)) - casadi.log(1.0e-6 + u),
#     casadi.log(casadi.DM(1.0e-6)) - beta(1.0e-6 + u),
# )
B_u_2 = casadi.if_else(
    u > delta,
    -casadi.log(u),
    -beta(u),
)
grad_B_u_1 = casadi.Function("grad_B_u_1", [u], [casadi.jacobian(B_u_1, u)])
grad_B_u_2 = casadi.Function("grad_B_u_2", [u], [casadi.jacobian(B_u_2, u)])
w_u_1 = 0.0
w_u_2 = -grad_B_u_1(0.0) / grad_B_u_2(0.0) - 1.0

B_u = casadi.Function("B_u", [u], [(1.0 + w_u_1) * B_u_1 + (1.0 + w_u_2) * B_u_2])
grad_B_u = casadi.Function("grad_B_u", [u], [casadi.jacobian(B_u(u), u)])
hess_B_u = casadi.Function("hess_B_u", [u], [casadi.hessian(B_u(u), u)[0]])
print("grad_B_u = ", grad_B_u(0.0))
assert casadi.norm_inf(grad_B_u(0.0)) < 1.0e-10, "grad_B_u(u) != 0"
M_u = casadi.DM((2 + w_u_1 + w_u_2) / (2 * delta * delta))
M_tilde_u = hess_B_u(casadi.DM(0.0))


# %% stage cost
l = casadi.Function(
    "l",
    [x, u],
    [
        (x[0] - 1.0) ** 2
        + (x[1] - 1.0) ** 2
        + 1.0e-4 * u**2
        + epsilon * B_x(x)
        + epsilon * B_u(u)
    ],
)

# %% terminal cost (determine P)
P = casadi.SX.sym("P", 2, 2)
R = casadi.DM(1.0e-4)
riccati = casadi.Function(
    "riccati",
    [P],
    [
        # casadi.densify(
        A.T * P * A
        - A.T @ P @ B @ casadi.inv(R + epsilon * M_u + B.T @ P @ B) @ B.T @ P @ A
        + casadi.DM([[0.1, 0.0], [0.0, 0.1]])
        + epsilon * M_x
        # )
    ],
)

P = casadi.DM.eye(2)
for i in range(1000):
    new_P = riccati(P)
    if casadi.norm_fro(new_P - P) < 1.0e-6:
        print("done")
        break
    P = new_P
print("P = ", P)

F = casadi.Function(
    "F", [x], [casadi.bilin(P, (x - casadi.DM.ones(2)), (x - casadi.DM.ones(2)))]
)
K = -casadi.inv(R + epsilon * M_tilde_u + B.T @ P @ B) @ B.T @ P @ A

# %% Create compute_control function


def solve_mpc(
    x_init: np.ndarray, x_start: list[casadi.DM], u_start: list[casadi.DM]
) -> casadi.DM:
    """
    returns the optimal solution with all the variables from all the stages
    """
    x_0 = casadi.MX.sym("x_0", 2)
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
        u_k = casadi.MX.sym("u_" + str(k))
        w += [u_k]
        w_start += [u_start[k]]

        x_k_end = f_discrete(x_k, u_k)
        J = J + l(x_k, u_k)

        x_k = casadi.MX.sym(
            "x_" + str(k + 1), 2
        )  # WARNING : from here x_k represents x_{k+1}
        w += [x_k]
        w_start += [x_start[k + 1]]
        lbw += [-inf, -inf, -inf]
        ubw += [inf, inf, inf]

        # Add equality constraint
        g += [x_k_end - x_k]
        lbg += [0.0, 0.0]
        ubg += [0.0, 0.0]
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
            "ipopt": {
                "max_iter": 1,
                "max_cpu_time": 10.0,
            },
        },
    )
    sol = nlp_solver(x0=casadi.vertcat(*w_start), lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    return sol["x"]


# %% solve the problem in a closed loop fashion
simulation_length = 25
x_init = [1.2, 0.7]


def createPlot(
    initial_states,
    initial_controls,
    initial_state_prediction,
    initial_control_prediction,
):
    """Creates a plot and adds the initial data provided by the arguments"""

    # Create empty plot
    fig = plt.figure(figsize=(15, 7))
    plt.clf()
    gs = GridSpec(2, 1, figure=fig)

    # Plot trajectory
    ax_state = fig.add_subplot(gs[0, 0])
    ax_state.grid("both")
    ax_state.set_title("evolution of populations")
    ax_state.set_xlabel("time")
    ax_state.set_ylabel("population")
    ax_state.plot([0, simulation_length], [2.0, 2.0], "r:")
    ax_state.plot([0, simulation_length], [0.0, 0.0], "r:")
    ax_state.plot(initial_states[0, 0], "b-")
    ax_state.plot(initial_states[1, 0], "m-")
    ax_state.plot(initial_state_prediction[0, :], "b--")
    ax_state.plot(initial_state_prediction[1, :], "m--")

    # Plot control
    ax_control = fig.add_subplot(gs[1, 0])
    ax_control.grid("both")
    ax_control.set_title("evolution of controls")
    ax_control.set_xlabel("time")
    ax_control.set_ylabel("control")
    ax_control.plot([0, simulation_length - 1], [0.0, 0.0], "r:")
    ax_control.plot([0, simulation_length - 1], [1.0, 1.0], "r:")
    ax_control.plot(initial_controls[0], "g-")
    ax_control.plot(initial_control_prediction, "g--")

    plt.tight_layout()


def updatePlots(
    states,
    controls,
    state_predictions,
    control_predictions,
    iteration,
    pause_duration=T / N,
):
    fig = plt.gcf()
    [ax_states, ax_control] = fig.axes

    # Delete old data in plots
    ax_states.get_lines().pop(-1).remove()
    ax_states.get_lines().pop(-1).remove()
    ax_states.get_lines().pop(-1).remove()
    ax_states.get_lines().pop(-1).remove()
    ax_control.get_lines().pop(-1).remove()
    ax_control.get_lines().pop(-1).remove()

    # Update plot with current simulation data
    ax_states.plot(range(0, iteration + 1), states[0, : iteration + 1], "b-")
    ax_states.plot(
        range(iteration, iteration + 1 + N), state_predictions[0, :].T, "b--"
    )
    ax_states.plot(range(0, iteration + 1), states[1, : iteration + 1], "m-")
    ax_states.plot(
        range(iteration, iteration + 1 + N), state_predictions[1, :].T, "m--"
    )

    ax_control.step(range(0, iteration + 1), controls[: iteration + 1], "g-")
    ax_control.step(range(iteration, iteration + N), control_predictions, "g--")

    plt.pause(pause_duration)


# Plot the solution
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

states = np.zeros((2, simulation_length + 1))
controls = np.zeros(simulation_length)
states[:, 0] = x_init

# Get a feasible trajectory as an initial guess (for the plot creation)
u_start = casadi.DM.zeros(N)
x_start = casadi.DM.zeros(2, N + 1)
x_start[:, 0] = x_init
for k in range(N):
    x_start[:, k + 1] = f_discrete(x_start[:, k], u_start[k])


createPlot(
    states,
    controls,
    x_start,
    u_start,
)

# transform the initial guess into a list of DMs for the solver
u_start = [casadi.DM(0.0)] * N
x_start_k = casadi.DM(x_init)
x_start = [x_start_k]
for k in range(N):
    x_start_k = f_discrete(x_start_k, u_start[k])
    x_start += [x_start_k]

for i in range(simulation_length):
    # retrieve solution data
    sol = solve_mpc(states[:, i], x_start=x_start, u_start=u_start)
    x_1_pred = sol[0::3]
    x_2_pred = sol[1::3]
    u_pred = sol[2::3]

    # update the states and controls
    # u_pred = np.clip(u_pred, 0.0, 1.0)
    controls[i] = u_pred[0]
    states[:, i + 1] = [x_1_pred[1], x_2_pred[1]]
    updatePlots(states, controls, casadi.horzcat(x_1_pred, x_2_pred).T, u_pred, i)
    if i == simulation_length - 1:
        print("Final state: ", states[:, i + 1])
        print("Final control: ", controls[i])
        print("all controls: ", controls)
        plt.show()
    else:
        plt.draw()

    # update the initial guess
    for k in range(N - 2):
        u_start[k] = u_pred[k + 1]
        x_start[k] = casadi.DM([x_1_pred[k + 1], x_2_pred[k + 1]])
    u_start[N - 1] = -K @ (x_start[N - 1] - casadi.DM.ones(2))
    x_start[N] = f_discrete(x_start[N - 1], u_start[N - 1])


# %%
