from operator import inv
from ppf import *
from scipy.linalg import solve_discrete_are, solve_continuous_are, solve_continuous_lyapunov, solve_discrete_lyapunov, inv
import numpy as np
import casadi as ca

x = ca.MX.sym('x', state_dim)
u = ca.MX.sym('u', control_dim)
A_func = ca.Function("A_func", [x,u], [ca.jacobian(f_cont(x,u), x)])
B_func = ca.Function("B_func", [x,u], [ca.jacobian(f_cont(x,u), u)])
A = A_func(x_S, u_S)
B = B_func(x_S, u_S)
print("A: ", A)
print("B: ", B)
P = solve_continuous_are(A, B, Q+epsilon*M_x, R+epsilon*M_u)
print("P:", P)
K = -inv(R) @ B.T @ P
print("K:", K)

stab_eig_values = np.linalg.eig(A + B @ K)
for val in stab_eig_values:
    if np.linalg.norm(val) >= 1.0:
        print("A+BK is not stable : ", np.linalg.norm(val))

controllability_matrix = np.tile(B, (1, N))
for i in range(1, N):
    controllability_matrix[:, i * control_dim : (i + 1) * control_dim] = (
        A @ controllability_matrix[:, (i - 1) * control_dim : i * control_dim]
    )
print("rank of controllability matrix = ", np.linalg.matrix_rank(controllability_matrix))