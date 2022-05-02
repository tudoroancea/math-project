from cstr import *  # also imports numpy, scipy and casadi

# ====================================================================================================
# Check that gradients of barrier functions are 0 at target state/control
# ====================================================================================================
if ca.norm_2(grad_B_x(x_S)) <= 1e-6:
    print("OK : grad_B_x(x_S) is 0")
else:
    sys.stderr.write(
        "ERROR : grad_B_x(x_S) = {}, norm = {}\n".format(grad_B_x(x_S)),
        ca.norm_2(grad_B_x(x_S)),
    )
    sys.stderr.flush()
if ca.norm_2(grad_B_u(u_S)) <= 1e-6:
    print("OK : grad_B_u(u_S) is 0")
else:
    sys.stderr.write(
        "ERROR : grad_B_u(u_S) = {}, norm = {}\n".format(grad_B_u(u_S)),
        ca.norm_2(grad_B_u(u_S)),
    )
    sys.stderr.flush()


# ====================================================================================================
# Check controllability/stabilizability
# ====================================================================================================
def test_controllability(my_A, my_B):
    controllability_matrix = np.tile(my_B, (1, N))
    for i in range(1, N):
        controllability_matrix[:, i * control_dim : (i + 1) * control_dim] = (
            my_A @ controllability_matrix[:, (i - 1) * control_dim : i * control_dim]
        )
    rank = np.linalg.matrix_rank(controllability_matrix)
    if rank == state_dim:
        print("\tOK : (A,B) is controllable")
    else:
        sys.stderr.write(
            "\tERROR : (A,B) is not controllable, rank = {}\n".format(rank)
        )

def test_stabilizability(my_A, my_B, my_K):
    discrete_eig = np.linalg.eig(my_A + my_B @ my_K)[0]
    not_in_unit_disk = []
    for val in discrete_eig:
        if np.linalg.norm(val) >= 1.0:
            not_in_unit_disk.append(np.linalg.norm(val))

    if len(not_in_unit_disk) == 0:
        print("\tOK : (A,B) is stabilizable")
    else:
        sys.stderr.write(
            "\tERROR : (A,B) is not stabilizable, eigenvalues = {}\n".format(
                not_in_unit_disk
            )
        )
        sys.stderr.flush()


# Discrete time
print("Discrete time stabilizability / controllability:")
test_controllability(A, B)
test_stabilizability(A, B, K)

# Continuous time
print("Continuous time stabilizability / controllability:")
A_cont_func = ca.Function("A_cont_func", [x, u], [ca.jacobian(f_cont(x, u), x)])
B_cont_func = ca.Function("B_cont_func", [x, u], [ca.jacobian(f_cont(x, u), u)])
A_cont = np.array(A_cont_func(x_S, u_S))
B_cont = np.array(B_cont_func(x_S, u_S))
test_controllability(A_cont, B_cont)

P_cont = la.solve_continuous_are(A_cont, B_cont, Q + epsilon * M_x, R + epsilon * M_u)
K_cont = -la.inv(R + epsilon * M_u) @ B_cont.T @ P_cont
test_stabilizability(A_cont, B_cont, K_cont)

# ====================================================================================================
# Check the positive definiteness of the hessian of the objective at the target state/control
# (for Lipschitzianity)
# ====================================================================================================
x_0 = ca.MX.sym("x_0", state_dim)
x_k = x_0
u = []

objective = 0
for k in range(N):
    u_k = ca.MX.sym("u_" + str(k), control_dim)
    u.append(u_k)
    x_k = f_discrete(x_k, u_k)
    objective += l(x_k, u_k)

objective += F(x_k)
u = ca.vertcat(*u)
hess_at = ca.Function("hess_at", [x_0, u], [ca.hessian(objective, u)[0]])
eigvalues = np.linalg.eig(np.array(hess_at(x_S, ca.vertcat(*([u_S] * N)))))[0]
if np.any(eigvalues <= 0):
    sys.stderr.write(
        "ERROR : hessian of the objective wrt controls at the origin is not positive \
        definite. eigen values are : {}\n".format(
            eigvalues
        )
    )
else:
    print(
        "OK hessian of the objective wrt controls at the origin is positive definite : "
    )

# ====================================================================================================
# Check positive definiteness of hessian of B_u at the target controls
# ====================================================================================================

eigvalues = np.linalg.eig(np.array(hess_B_u(ca.DM.zeros(control_dim))))[0]
if np.any(eigvalues <= 0):
    sys.stderr.write(
        "ERROR: hessian of B_u wrt controls at the origin is not positive definite. \
        eigen values are : {}\n".format(
            eigvalues
        )
    )
else:
    print("OK : hessian of B_u wrt controls at the origin is positive definite")
