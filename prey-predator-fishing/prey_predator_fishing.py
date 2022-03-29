# %% imports and global parameters
import casadi
from math import inf

T = 12.0  # Time horizon
N = 40  # number of control intervals
M = 4  # RK4 integrations steps
delta = 0.5  # relaxation parameter
epsilon = 0.01  # barrier parameter

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
print("w_x_3 = ", w_x_3)
print("w_x_4 = ", w_x_4)
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
# assert grad_B_x(casadi.DM([1.0, 1.0])) == casadi.DM.zeros(1,2), "grad_B_x(x) != 0"

M_x = casadi.DM(
    [[2.0 + float(w_x_1) + float(w_x_3), 0.0], [0.0, 2.0 + float(w_x_2) + float(w_x_4)]]
) / (2.0 * delta * delta)
M_tilde_x = hess_B_x(casadi.DM([1.0, 1.0]))


B_u_1 = casadi.if_else(1 - u > delta, -casadi.log(1 - u), beta(1 - u))
B_u_2 = casadi.if_else(
    1.0e-6 + u > delta,
    casadi.log(casadi.DM(1.0e-6)) - casadi.log(1.0e-6 + u),
    casadi.log(casadi.DM(1.0e-6)) - beta(1.0e-6 + u),
)
grad_B_u_1 = casadi.Function("grad_B_u_1", [u], [casadi.jacobian(B_u_1, u)])
grad_B_u_2 = casadi.Function("grad_B_u_2", [u], [casadi.jacobian(B_u_2, u)])
w_u_1 = 0.0
w_u_2 = -grad_B_u_1(0.0) / grad_B_u_2(0.0) - 1.0

B_u = casadi.Function("B_u", [u], [(1.0 + w_u_1) * B_u_1 + (1.0 + w_u_2) * B_u_2])
grad_B_u = casadi.Function("grad_B_u", [u], [casadi.jacobian(B_u(u), u)])
hess_B_u = casadi.Function("hess_B_u", [u], [casadi.hessian(B_u(u), u)[0]])
print("grad_B_u = ", grad_B_u(0.0))
# assert grad_B_u(0.0) == casadi.DM([0.0]), "grad_B_u(u) != 0"
M_u = casadi.DM((2 + w_u_1 + w_u_2) / (2 * delta * delta))
M_tilde_u = hess_B_u(casadi.DM(0.0))


# %% stage cost
l = casadi.Function(
    "l",
    [x, u],
    [
        (x[0] - 1) ** 2
        + (x[1] - 1) ** 2
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

F = casadi.Function("F", [x], [casadi.bilin(P, x, x)])


# %% Initial conditions and guesses
x_init = [0.5, 0.7]
# Initial guess for u
u_start = [casadi.DM(0.0)] * N
# Get a feasible trajectory as an initial guess
x_k = casadi.DM(x_init)
x_start = [x_k]
for k in range(N):
    x_k = f_discrete(x_k, u_start[k])
    x_start += [x_k]

# %% Formulate the NLP

x_0 = casadi.MX.sym("x_0", 2)
x_k = x_0

J = 0  # objective function
w = [x_0]  # list of all the variables x and u concatenated
w_start = [x_start[0]]  # initial guess
lbw = []
lbw += x_init  # lower bound for all the variables
ubw = []
ubw += x_init  # upper bound for all the variables
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
    lbg += [0, 0]
    ubg += [0, 0]
J += F(x_k)  # here x_k represents x_N

# Concatenate decision variables and constraint terms
w = casadi.vertcat(*w)
g = casadi.vertcat(*g)

# Create an NLP solver
nlp_prob = {"f": J, "x": w, "g": g}
nlp_solver = casadi.nlpsol("nlp_solver", "ipopt", nlp_prob)

# %% solve the NLP
sol = nlp_solver(x0=casadi.vertcat(*w_start), lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
# print(nlp_solver.stats())

# Plot the solution
tgrid = [T / N * k for k in range(N + 1)]
import matplotlib.pyplot as plt

plt.figure(1)
plt.clf()


def plot_sol(w_opt):
    w_opt = w_opt.full().flatten()
    x0_opt = w_opt[0::3]
    x1_opt = w_opt[1::3]
    u_opt = w_opt[2::3]
    plt.plot(tgrid, x0_opt, "--")
    plt.plot(tgrid, x1_opt, "-")
    plt.step(tgrid, casadi.vertcat(casadi.DM.nan(1), u_opt), "-.")
    plt.xlabel("t")
    plt.legend(["x0", "x1", "u"])
    plt.grid(True)


plot_sol(sol["x"])
plt.show()

# %%
