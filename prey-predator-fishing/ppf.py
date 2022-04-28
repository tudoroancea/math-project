# %% imports and global parameters
import os
import sys
from time import time
from typing import Tuple

import casadi
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# general problem parameters
state_dim = 2
control_dim = 1
T = 12.0  # Time horizon
N = 40  # number of control intervals
M = 4  # RK4 integrations steps
delta = 0.33  # relaxation parameter
epsilon = 0.05  # barrier parameter
x_S = casadi.DM([1.0, 1.0])  # state target
u_S = casadi.DM(0.0)  # control target
simulation_length = 60
x_init = [1.2, 0.7]  # initial state

# %% Declare model
x = casadi.SX.sym("x", state_dim)
u = casadi.SX.sym("u", control_dim)
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
(A, B) = f_discrete_linearized(x_S, u_S)
print("A = ", A)
print("B = ", B)


# %% barrier functions for state and control constraints
z = casadi.SX.sym("z")
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
w_x_3 = -grad_B_x_1(x_S)[0] / grad_B_x_3(x_S)[0] - 1.0
w_x_4 = -grad_B_x_2(x_S)[1] / grad_B_x_4(x_S)[1] - 1.0
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
print("grad_B_x = ", grad_B_x(x_S))
assert casadi.norm_inf(grad_B_x(x_S)) < 1.0e-10, "grad_B_x(x) != 0"
M_x = hess_B_x(x_S)


B_u_1 = casadi.if_else(1 - u > delta, -casadi.log(1 - u), beta(1 - u))
B_u_2 = casadi.if_else(
    u > delta,
    -casadi.log(u),
    -beta(u),
)
grad_B_u_1 = casadi.Function("grad_B_u_1", [u], [casadi.jacobian(B_u_1, u)])
grad_B_u_2 = casadi.Function("grad_B_u_2", [u], [casadi.jacobian(B_u_2, u)])
w_u_1 = 0.0
w_u_2 = -grad_B_u_1(u_S) / grad_B_u_2(u_S) - 1.0

B_u = casadi.Function("B_u", [u], [(1.0 + w_u_1) * B_u_1 + (1.0 + w_u_2) * B_u_2])
grad_B_u = casadi.Function("grad_B_u", [u], [casadi.jacobian(B_u(u), u)])
hess_B_u = casadi.Function("hess_B_u", [u], [casadi.hessian(B_u(u), u)[0]])
print("grad_B_u = ", grad_B_u(u_S))
assert casadi.norm_inf(grad_B_u(u_S)) < 1.0e-10, "grad_B_u(u) != 0"
M_u = hess_B_u(u_S)


# %% stage cost
l = casadi.Function(
    "l",
    [x, u],
    [(x[0] - x_S[0]) ** 2 + (x[1] - x_S[1]) ** 2 + 1.0e-4 * (u - u_S) ** 2],
)
l_tilde = casadi.Function(
    "l_tilde", [x, u], [l(x, u) + epsilon * B_x(x) + epsilon * B_u(u)]
)
Q = casadi.DM([[0.1, 0.0], [0.0, 0.1]])
R = casadi.DM(1.0e-4)

# %% terminal cost (determine P)
P = casadi.SX.sym("P", 2, 2)
riccati = casadi.Function(
    "riccati",
    [P],
    [
        A.T * P * A
        - A.T @ P @ B @ casadi.inv(R + epsilon * M_u + B.T @ P @ B) @ B.T @ P @ A
        + +epsilon * M_x
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

F = casadi.Function("F", [x], [casadi.bilin(P, (x - x_S), (x - x_S))])
K = -casadi.inv(R + epsilon * M_u + B.T @ P @ B) @ B.T @ P @ A

# %% Check the assumptions of the nominal stability theorem
# stabilizability of (A,B)

# hessian of the objective is positive definite at the target (for Lipschitzianity)
x_0 = casadi.MX.sym("x_0", state_dim)
x_k = x_0
u = []

objective = 0
for k in range(N):
    u_k = casadi.MX.sym("u_" + str(k))
    u.append(u_k)
    x_k = f_discrete(x_k, u_k)
    objective += l(x_k, u_k)

objective += F(x_k)
u = casadi.vertcat(*u)
hess_at = casadi.Function("hess_at", [x_0, u], [casadi.hessian(objective, u)[0]])
print(
    "hessian of the objective wrt controls at the origin is positive definite : ",
    not np.any(
        np.linalg.eig(
            np.array(
                hess_at(casadi.DM.zeros(state_dim), casadi.DM.zeros(N * control_dim))
            )
        )[0]
        <= 0
    ),
)
stab_eig_values = np.linalg.eig(A + B @ K)
for val in stab_eig_values:
    if np.linalg.norm(val) >= 1.0:
        print("A+BK is not stable : ", np.linalg.norm(val))
        # break

print(
    "hessian of B_u wrt controls at the origin is positive definite : ",
    not np.any(np.linalg.eig(np.array(hess_B_u(casadi.DM.zeros(control_dim))))[0] <= 0),
)
# %% Create compute_control function
def solve_rrlb_mpc(
    x_init: np.ndarray,
    x_start: list[casadi.DM],
    u_start: list[casadi.DM],
    solver: str = "ipopt",
    opts: dict = {
        "print_time": 0,
        "ipopt": {"print_level": 0, "max_iter": 1, "max_cpu_time": 10.0},
    },
) -> casadi.DM:
    """
    returns the optimal solution with all the variables from all the stages
    """
    x_0 = casadi.MX.sym("x_0", state_dim)
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
        u_k = casadi.MX.sym("u_" + str(k), control_dim)
        w += [u_k]
        w_start += [u_start[k]]

        x_k_end = f_discrete(x_k, u_k)
        J = J + l_tilde(x_k, u_k)

        x_k = casadi.MX.sym(
            "x_" + str(k + 1), 2
        )  # WARNING : from here x_k represents x_{k+1}
        w += [x_k]
        w_start += [x_start[k + 1]]
        lbw += [-np.inf] * (state_dim + control_dim)
        ubw += [np.inf] * (state_dim + control_dim)

        # Add equality constraint
        g += [x_k_end - x_k]
        lbg += [0.0] * state_dim
        ubg += [0.0] * state_dim
    J += F(x_k)  # here x_k represents x_N

    # Concatenate decision variables and constraint terms
    w = casadi.vertcat(*w)
    g = casadi.vertcat(*g)

    # Create an NLP solver
    nlp_prob = {"f": J, "x": w, "g": g}
    nlp_solver = casadi.nlpsol(
        "nlp_solver",
        solver,
        nlp_prob,
        opts,
    )
    sol = nlp_solver(x0=casadi.vertcat(*w_start), lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    return sol["x"]


def solve_mpc(
    x_init: np.ndarray,
    x_start: list[casadi.DM],
    u_start: list[casadi.DM],
    solver: str = "ipopt",
    opts: dict = {
        "print_time": 0,
        "ipopt": {"print_level": 0, "max_cpu_time": 10.0},
    },
) -> casadi.DM:
    """
    returns the optimal solution with all the variables from all the stages
    """
    x_0 = casadi.MX.sym("x_0", state_dim)
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
        u_k = casadi.MX.sym("u_" + str(k), control_dim)
        w += [u_k]
        w_start += [u_start[k]]
        lbw += [0.0]
        ubw += [1.0]

        x_k_end = f_discrete(x_k, u_k)
        J = J + l(x_k, u_k)

        x_k = casadi.MX.sym(
            "x_" + str(k + 1), state_dim
        )  # WARNING : from here x_k represents x_{k+1}
        w += [x_k]
        w_start += [x_start[k + 1]]
        lbw += [0.0, 0.0]
        ubw += [2.0, 2.0]

        # Add equality constraint
        g += [x_k_end - x_k]
        lbg += [0.0] * state_dim
        ubg += [0.0] * state_dim
    J += F(x_k)  # here x_k represents x_N

    # Concatenate decision variables and constraint terms
    w = casadi.vertcat(*w)
    g = casadi.vertcat(*g)

    # Create an NLP solver
    nlp_prob = {"f": J, "x": w, "g": g}
    nlp_solver = casadi.nlpsol(
        "nlp_solver",
        solver,
        nlp_prob,
        opts,
    )
    sol = nlp_solver(x0=casadi.vertcat(*w_start), lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    return sol["x"]


# %% solve the problem in a closed loop fashion
def createPlot(
    states,
    controls,
    state_prediction,
    control_prediction,
    simulation_length,
    final=False,
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
    if not final:
        ax_state.plot(states[0, 0], "b-")
        ax_state.plot(states[1, 0], "m-")
    else:
        ax_state.plot(states[0, :], "b-")
        ax_state.plot(states[1, :], "m-")
    ax_state.plot(state_prediction[0, :], "b--")
    ax_state.plot(state_prediction[1, :], "m--")

    # Plot control
    ax_control = fig.add_subplot(gs[1, 0])
    ax_control.grid("both")
    ax_control.set_title("evolution of controls")
    ax_control.set_xlabel("time")
    ax_control.set_ylabel("control")
    ax_control.plot([0, simulation_length - 1], [0.0, 0.0], "r:")
    ax_control.plot([0, simulation_length - 1], [1.0, 1.0], "r:")
    if not final:
        ax_control.plot(controls[0], "g-")
    else:
        ax_control.step(controls, "g-")

    ax_control.step(
        range(controls.shape[0] - 1, controls.shape[0] - 1 + N),
        control_prediction,
        "g--",
    )

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


def run_closed_loop_simulation(
    custom_x_init=x_init,
    stop_tol=1.0e-3,
    max_simulation_length: int = 100,
    method=solve_rrlb_mpc,
    name: str = None,
    step_by_step: bool = True,
    solver: str = "ipopt",
    opts: dict = {
        "print_time": 0,
        "ipopt": {"print_level": 0, "max_iter": 1, "max_cpu_time": 10.0},
    },
) -> Tuple[float, int, bool]:
    """
    Returns the total cost of the run until convergence, the number of iterations taken to reach
    convergence, and if there was any constraint violation
    """
    constraints_violated = False
    states = np.zeros((state_dim, max_simulation_length + 1))
    controls = np.zeros(max_simulation_length)
    states[:, 0] = custom_x_init

    # Get a feasible trajectory as an initial guess (for the plot creation)
    u_start = casadi.horzcat(*([u_S] * N))
    x_start = casadi.DM.zeros(state_dim, N + 1)
    x_start[:, 0] = custom_x_init
    for k in range(N):
        x_start[:, k + 1] = f_discrete(x_start[:, k], u_start[k])

    if step_by_step:
        createPlot(
            states,
            controls,
            x_start,
            u_start,
            simulation_length=max_simulation_length,
        )

    # transform the initial guess into a list of DMs for the solver
    u_start = [u_S] * N
    x_start_k = casadi.DM(custom_x_init)
    x_start = [x_start_k]
    for k in range(N):
        x_start_k = f_discrete(x_start_k, u_start[k])
        x_start += [x_start_k]

    i = 0

    def converged(i: int) -> bool:
        return (
            np.sqrt(np.sum(np.square(states[:, i] - np.array(x_S).ravel()))) < stop_tol
        )

    while not converged(i) and i < max_simulation_length:
        # retrieve solution data
        start = time()
        sol = method(
            x_init=states[:, i],
            x_start=x_start,
            u_start=u_start,
            solver=solver,
            opts=opts,
        )
        stop = time()
        # print("time for iteration {} : {} ms".format(i, 1000 * (stop - start)))
        x_1_pred = sol[0::3]
        x_2_pred = sol[1::3]
        u_pred = sol[2::3]

        # update the states and controls
        controls[i] = u_pred[0]
        states[:, i + 1] = np.array([float(x_1_pred[1]), float(x_2_pred[1])])
        states_pred = casadi.horzcat(x_1_pred, x_2_pred).T

        # check if there was any constraint violation
        if controls[i] < 0.0 or controls[i] > 1.0:
            constraints_violated = True

        # update the initial guess for next solve
        for k in range(N - 2):
            u_start[k] = u_pred[k + 1]
            x_start[k] = states_pred[:, k + 1]
        u_start[N - 1] = -K @ (x_start[N - 1] - x_S)
        x_start[N] = f_discrete(x_start[N - 1], u_start[N - 1])

        # if the simulation will stop at next iteration (either because it has converged or because we
        #  have reached the maximum number of iterations), we say what happened
        if converged(i + 1):
            print("Converged at iteration {}".format(i))
        elif i == max_simulation_length - 1:
            sys.stderr.write("Not converged in {max_simulation_length} iterations")

        # plot the simulation data if need be
        if step_by_step:
            updatePlots(
                states,
                controls,
                states_pred,
                u_pred,
                i,
            )
            plt.draw()

        i += 1

    print("Final state: ", states[:, i + 1])
    print("Final control: ", controls[i])
    states = np.delete(states, list(range(i + 1, max_simulation_length + 1)), axis=1)
    controls = np.delete(controls, list(range(i, max_simulation_length)), axis=0)

    if step_by_step:
        plt.show()
    elif name != None:
        createPlot(
            states, controls, states_pred, u_pred, simulation_length=i, final=True
        )
        if not converged(i):
            plt.title("NOT CONVERGED")
        plt.savefig(
            os.path.join(os.path.dirname(__file__), name + ".png"),
            dpi=300,
            format="png",
        )

    # compute total sum of controls
    total_cost = 0.0
    np.sum(np.sqrt(np.sum((np.square(states - np.array(x_S))), axis=0)))
    for k in range(i):
        total_cost += l(states[:, k], controls[k])
    total_cost += F(states[:, i])

    return float(total_cost), i, constraints_violated
