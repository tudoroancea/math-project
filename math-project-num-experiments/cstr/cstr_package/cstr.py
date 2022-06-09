import os
from enum import Enum
from time import time
from typing import Tuple

import casadi as ca
import numpy as np
import scipy.linalg as la
from scipy import sparse

global_epsilon = 1e-3


class Scheme(Enum):
    RRLB = 1
    REGULAR = 2
    INFINITE_HORIZON = 3

    @staticmethod
    def from_str(scheme_str: str):
        if scheme_str == "rrlb":
            return Scheme.RRLB
        elif scheme_str == "reg":
            return Scheme.REGULAR
        elif scheme_str == "infhorz":
            return Scheme.INFINITE_HORIZON
        else:
            raise ValueError("Unknown scheme: {}".format(scheme_str))


INF_HORIZON_SIZE = 5000

# Reference points computed via a steady state analysis
xr1 = np.array(
    [
        2.1402105301746182e00,
        1.0903043613077321e00,
        1.1419108442079495e02,
        1.1290659291045561e02,
    ]
)
ur1 = np.array([14.19, -1113.50])
xr2 = np.array([2.7151681, 1.02349152, 105.50585058, 100.8920758])
ur2 = np.array([13.66640639, -3999.58908628])
xr3 = np.array([2.97496989, 0.95384459, 101.14965441, 95.19386292])
ur3 = np.array([12.980148, -5162.95653026])


class CSTR:
    """
    Class created to encapsulate the dynamical model of the CSTR system, all the cost functions and
    parameters of the MPC, and some auxiliary functions used for preparing an RTI iteration.
    """

    # misc
    nx = 4  # dimension of state
    nu = 2  # dimension of control
    states_idx = list(range(nx))  # indices of states in a stage vector (x_k, u_k)
    controls_idx = list(
        range(nx, nx + nu)
    )  # indices of controls in a stage vector (x_k, u_k)
    T: float  # time horizon
    N: int  # number of control intervals
    M = 6  # RK4 integration steps
    epsilon: float  # barrier parameter
    is_RRLB: bool  # whether to compute the RRLB functions or not

    # constraints (see the paper for notations)
    C_x = sparse.kron(sparse.eye(nx), sparse.csc_matrix([[1.0], [-1.0]]), format="csc")
    q_x: int = 8
    C_u = sparse.csc_matrix(
        np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    )
    q_u: int = 4
    d_x = np.array([10.0, 0.0, 10.0, 0.0, 150.0, -98.0, 150.0, -92.0])
    d_u = np.array([35.0, -3.0, 0.0, 9000.0])

    # reference points
    xr: np.ndarray  # target / reference state
    ur: np.ndarray  # target / reference control

    # RRLB functions (see paper for notations)
    B_x: ca.Function
    B_u: ca.Function
    grad_B_x: ca.Function
    grad_B_u: ca.Function
    hess_B_x: ca.Function
    hess_B_u: ca.Function
    M_x: np.ndarray
    M_u: np.ndarray
    delta: float  # relaxation parameter

    # dynamics (see paper for notations)
    f: ca.Function  # discrete dynamics (for the fixed sampling time)
    f_cont: ca.Function  # continuous dynamics
    f_special: ca.Function  # discrete dynamics (for an arbitrary sampling time)
    jac_f_x: ca.Function
    jac_f_u: ca.Function
    A: np.ndarray
    B: np.ndarray

    # costs (see paper for notations)
    Q = sparse.diags([[0.2, 1.0, 0.5, 0.2]], [0], format="csc")
    R = sparse.diags([[0.5, 5.0e-7]], [0], format="csc")
    P: np.ndarray
    K: np.ndarray
    l: ca.Function
    l_tilde: ca.Function
    F: ca.Function

    def __init__(
        self,
        xr: np.ndarray = xr1,
        ur: np.ndarray = ur1,
        epsilon: float = global_epsilon,
        scheme: Scheme = Scheme.RRLB,
        N: int = 100,
    ) -> None:
        # Choosing the MPC scheme ===================================================
        if scheme == Scheme.RRLB or scheme == Scheme.REGULAR:
            self.N = N
            self.T = N * 20.0
        else:
            self.N = INF_HORIZON_SIZE
            self.T = INF_HORIZON_SIZE * 20.0

        self.epsilon = epsilon
        if scheme == Scheme.RRLB:
            self.is_RRLB = True
        else:
            self.is_RRLB = False

        # Defining model dynamics =========================================================
        x = ca.SX.sym("x", self.nx)
        u = ca.SX.sym("u", self.nu)
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

        k_1 = k_10 * ca.exp(E_1 / (x[2] + 273.15))
        k_2 = k_20 * ca.exp(E_2 / (x[2] + 273.15))
        k_3 = k_30 * ca.exp(E_3 / (x[2] + 273.15))

        self.f_cont = ca.Function(
            "f_cont",
            [x, u],
            [
                ca.vertcat(
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

        DT = self.T / self.N / self.M
        new_x = x
        for i in range(self.M):
            k1 = self.f_cont(new_x, u)
            k2 = self.f_cont(new_x + DT / 2 * k1, u)
            k3 = self.f_cont(new_x + DT / 2 * k2, u)
            k4 = self.f_cont(new_x + DT * k3, u)
            new_x = new_x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.f = ca.Function("f", [x, u], [new_x])
        self.jac_f_x = ca.Function("jac_f_x", [x, u], [ca.jacobian(new_x, x)])
        self.jac_f_u = ca.Function("jac_f_u", [x, u], [ca.jacobian(new_x, u)])

        h = ca.SX.sym("h")
        DT = h / self.M
        new_x = x
        for i in range(self.M):
            k1 = self.f_cont(new_x, u)
            k2 = self.f_cont(new_x + DT / 2 * k1, u)
            k3 = self.f_cont(new_x + DT / 2 * k2, u)
            k4 = self.f_cont(new_x + DT * k3, u)
            new_x = new_x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.f_special = ca.Function("f", [x, u, h], [new_x])

        # Setting reference point ===========================================================
        self.xr = xr
        self.ur = ur

        # Computing RRLB functions ==========================================================
        if self.is_RRLB:
            # first compute the translated d_x, i.e. d_x_tilde
            d_x_tilde = self.d_x - self.C_x @ self.xr
            delta_x = np.min(np.abs(d_x_tilde))
            # first compute the translated d_u, i.e. d_u_tilde
            d_u_tilde = self.d_u - self.C_u @ self.ur
            delta_u = np.min(np.abs(d_u_tilde))
            # choose delta
            self.delta = 0.5 * np.min([delta_u, delta_x])
            # define the function beta
            z = ca.SX.sym("z")
            beta = ca.Function(
                "beta",
                [z],
                [
                    0.5 * (((z - 2 * self.delta) / self.delta) ** 2 - 1)
                    - ca.log(self.delta)
                ],
            )

            # compute the individual functions B_x_i
            tpr = []
            full_C_x = self.C_x.toarray()
            for i in range(self.q_x):
                z_i = self.d_x[i] - ca.dot(full_C_x[i, :], x)
                tpr.append(
                    ca.if_else(
                        z_i > self.delta, ca.log(d_x_tilde[i]) - ca.log(z_i), beta(z_i)
                    )
                )
            tpr2 = d_x_tilde[1::2]
            tpr3 = d_x_tilde[0::2]
            # compute the weight vector w_x
            w_x = np.ones(self.q_x)
            w_x[1::2] = tpr2 / tpr3
            # assemble the RRLB function B_x
            self.B_x = ca.Function("B_x", [x], [ca.dot(w_x, ca.vertcat(*tpr))])

            self.grad_B_x = ca.Function("grad_B_x", [x], [ca.jacobian(self.B_x(x), x)])
            self.hess_B_x = ca.Function(
                "hess_B_x", [x], [ca.hessian(self.B_x(x), x)[0]]
            )
            self.M_x = self.hess_B_x(self.xr).full()

            # compute the individual functions B_u_i
            tpr = []
            full_C_u = self.C_u.toarray()
            for i in range(self.q_u):
                z_i = self.d_u[i] - ca.dot(full_C_u[i, :], u)
                tpr.append(
                    ca.if_else(
                        z_i > self.delta, ca.log(d_u_tilde[i]) - ca.log(z_i), beta(z_i)
                    )
                )
            tpr2 = d_u_tilde[1::2]
            tpr3 = d_u_tilde[0::2]
            # compute the weight vector w_u
            w_u = np.ones(self.q_u)
            w_u[1::2] = tpr2 / tpr3
            # assemble the RRLB function B_u
            self.B_u = ca.Function("B_u", [u], [ca.dot(w_u, ca.vertcat(*tpr))])

            self.grad_B_u = ca.Function("grad_B_u", [u], [ca.jacobian(self.B_u(u), u)])
            self.hess_B_u = ca.Function(
                "hess_B_u", [u], [ca.hessian(self.B_u(u), u)[0]]
            )
            self.M_u = self.hess_B_u(self.ur).full()

        # Defining costs functions ==========================================================
        self.l = ca.Function(
            "l",
            [x, u],
            [
                0.2 * (x[0] - self.xr[0]) ** 2
                + 1.0 * (x[1] - self.xr[1]) ** 2
                + 0.5 * (x[2] - self.xr[2]) ** 2
                + 0.2 * (x[3] - self.xr[3]) ** 2
                + 0.5 * (u[0] - self.ur[0]) ** 2
                + 5.0e-7 * (u[1] - self.ur[1]) ** 2
            ],
        )
        self.A = np.array(self.jac_f_x(self.xr, self.ur))
        self.B = np.array(self.jac_f_u(self.xr, self.ur))
        if self.is_RRLB:
            self.P = np.array(
                la.solve_discrete_are(
                    self.A,
                    self.B,
                    self.Q + self.epsilon * self.M_x,
                    self.R + self.epsilon * self.M_u,
                )
            )
            self.K = (
                -la.inv(self.R + self.epsilon * self.M_u + self.B.T @ self.P @ self.B)
                @ self.B.T
                @ self.P
                @ self.A
            )
            self.l_tilde = ca.Function(
                "l_tilde",
                [x, u],
                [
                    self.l(x, u)
                    + self.epsilon * self.B_x(x)
                    + self.epsilon * self.B_u(u)
                ],
            )
        else:
            self.P = np.array(
                la.solve_discrete_are(
                    self.A,
                    self.B,
                    self.Q.toarray(),
                    self.R.toarray(),
                )
            )
            self.K = (
                -la.inv(self.R.toarray() + self.B.T @ self.P @ self.B)
                @ self.B.T
                @ self.P
                @ self.A
            )

        self.F = ca.Function("F", [x], [ca.bilin(self.P, (x - self.xr), (x - self.xr))])

        # Verify all assumptions ============================================================
        if self.N != INF_HORIZON_SIZE:
            assumptions_verified, err_msg = self.check_stability_assumptions()
            assert assumptions_verified, err_msg

    def check_stability_assumptions(self):
        err = False
        err_msg = ""
        # ====================================================================================================
        # Check that the given reference is indeed a steady state
        # ====================================================================================================
        if np.linalg.norm(self.xr - self.f(self.xr, self.ur)) > 1e-8:
            err = True
            err_msg += "ERROR : The given reference xr={},ur={} is not a steady state.\n".format(
                self.xr, self.ur
            )

        # ====================================================================================================
        # Check recentering of B_x and B_u, i.e. that they are 0 at xr and ur resp. and
        # that their gradients 0 at xr and ur resp.
        # ====================================================================================================
        if self.is_RRLB:
            if ca.fabs(self.B_x(self.xr)) >= 1e-6:
                err = True
                err_msg += "ERROR : |B_x(xr)| = {}\n".format(ca.fabs(self.B_x(self.xr)))

            if ca.fabs(self.B_u(self.ur)) >= 1e-6:
                err = True
                err_msg += "ERROR : |B_u(ur)| = {}\n".format(ca.fabs(self.B_u(self.ur)))

            if ca.norm_inf(self.grad_B_x(self.xr)) >= 1e-6:
                err = True
                err_msg += "ERROR : grad_B_x(xr) = {}, norm = {}\n".format(
                    self.grad_B_x(self.xr), ca.norm_inf(self.grad_B_x(self.xr))
                )

            if (
                ca.norm_inf(self.grad_B_u(self.ur)) >= 1e-6
                or ca.fabs(self.B_u(self.ur)) >= 1e-6
            ):
                err = True
                err_msg += "ERROR : grad_B_u(ur) = {}, norm = {}\n".format(
                    self.grad_B_u(self.ur), ca.norm_inf(self.grad_B_u(self.ur))
                )

        # ====================================================================================================
        # Check positive definiteness of hessian of B_u at the target controls
        # ====================================================================================================
        if self.is_RRLB:
            eigvalues = np.linalg.eigvalsh(
                np.array(self.hess_B_u(ca.DM.zeros(self.nu)))
            )
            if np.any(eigvalues <= 0):
                err = True
                err_msg += "ERROR: hessian of B_u wrt controls at the origin is not \
                    positive definite. Eigen values are : {}\n".format(
                    eigvalues
                )

            eigvalues = np.linalg.eigvalsh(
                np.array(self.hess_B_x(ca.DM.zeros(self.nx)))
            )
            if np.any(eigvalues <= 0):
                err = True
                err_msg += "ERROR: hessian of B_x wrt controls at the origin is not \
                                positive definite. Eigen values are : {}\n".format(
                    eigvalues
                )

        # ====================================================================================================
        # Check controllability/stabilizability
        # ====================================================================================================
        def check_controllability(my_A, my_B) -> Tuple[bool, str]:
            controllability_matrix = np.tile(my_B, (1, self.nx))
            for i in range(1, self.nx):
                controllability_matrix[:, i * self.nu : (i + 1) * self.nu] = (
                    my_A @ controllability_matrix[:, (i - 1) * self.nu : i * self.nu]
                )
            rank = np.linalg.matrix_rank(controllability_matrix)
            if rank != self.nx:
                return True, "\tERROR : (A,B) is not controllable, rank = {}\n".format(
                    rank
                )
            else:
                return False, ""

        def check_stabilizability(my_A, my_B, my_K) -> Tuple[bool, str]:
            discrete_eig = np.linalg.eig(my_A + my_B @ my_K)[0]
            not_in_unit_disk = []
            for val in discrete_eig:
                if np.linalg.norm(val) >= 1.0:
                    not_in_unit_disk.append(np.linalg.norm(val))

            if len(not_in_unit_disk) != 0:
                return (
                    True,
                    "\tERROR : (A,B) is not stabilizable, eigenvalues = {}\n".format(
                        not_in_unit_disk
                    ),
                )
            else:
                return False, ""

        # Discrete time
        err_msg += "Discrete time stabilizability / controllability:\n"
        err_tpr, err_msg_tpr = check_controllability(self.A, self.B)
        if err_tpr:
            err = True
            err_msg += err_msg_tpr
        err_tpr, err_msg_tpr = check_stabilizability(self.A, self.B, self.K)
        if err_tpr:
            err = True
            err_msg += err_msg_tpr

        # Continuous time
        err_msg += "Continuous time stabilizability / controllability:\n"
        x = ca.SX.sym("x", self.nx)
        u = ca.SX.sym("u", self.nu)
        A_cont_func = ca.Function(
            "A_cont_func", [x, u], [ca.jacobian(self.f_cont(x, u), x)]
        )
        B_cont_func = ca.Function(
            "B_cont_func", [x, u], [ca.jacobian(self.f_cont(x, u), u)]
        )
        A_cont = np.array(A_cont_func(self.xr, self.ur))
        B_cont = np.array(B_cont_func(self.xr, self.ur))

        err_tpr, err_msg_tpr = check_controllability(A_cont, B_cont)
        if err_tpr:
            err = True
            err_msg += err_msg_tpr

        if self.is_RRLB:
            P_cont = la.solve_continuous_are(
                A_cont,
                B_cont,
                self.Q + self.epsilon * self.M_x,
                self.R + self.epsilon * self.M_u,
            )
            K_cont = -la.inv(self.R + self.epsilon * self.M_u) @ B_cont.T @ P_cont
        else:
            P_cont = la.solve_continuous_are(
                A_cont,
                B_cont,
                self.Q.toarray(),
                self.R.toarray(),
            )
            K_cont = -la.inv(self.R.toarray()) @ B_cont.T @ P_cont

        err_tpr, err_msg_tpr = check_stabilizability(A_cont, B_cont, K_cont)
        if err_tpr:
            err = True
            err_msg += err_msg_tpr

        # ====================================================================================================
        # Check the positive definiteness of the hessian of the objective at the target state/control
        # (for Lipschitzianity)
        # ====================================================================================================
        if self.N != INF_HORIZON_SIZE:
            x_0 = ca.MX.sym("x_0", self.nx)
            x_k = x_0
            u = []

            objective = 0
            for k in range(self.N):
                u_k = ca.MX.sym("u_" + str(k), self.nu)
                u.append(u_k)
                x_k = self.f(x_k, u_k)
                objective += self.l(x_k, u_k)

            objective += self.F(x_k)
            u = ca.vertcat(*u)
            hess_at = ca.Function("hess_at", [x_0, u], [ca.hessian(objective, u)[0]])
            eigvalues = np.linalg.eigvalsh(
                np.array(hess_at(self.xr, ca.vertcat(*([self.ur] * self.N))))
            )
            if np.any(eigvalues <= 0):
                err = True
                err_msg += "ERROR : hessian of the objective wrt controls at the origin is \
                    not positive definite. Eigen values are : {}\n".format(
                    eigvalues
                )

        return not err, err_msg

    def initial_prediction(self, initial_state: np.ndarray) -> np.ndarray:
        """
        Compute the initial prediction of the state and the control input in the format
        (x_0, x_1, ...,x_N, u_0, u_1, ..., u_{N-1})
        """
        x_k = ca.MX.sym("x_0", self.nx)

        J = 0  # objective function
        w_x = [x_k]  # list of all state variables
        w_u = []  # list of all control variables
        g = []  # equality constraints

        w_start_x = [initial_state]  # initial guess
        w_start_u = [self.ur] * self.N

        # Formulate the NLP
        for k in range(self.N):
            # New NLP variable for the control
            u_k = ca.MX.sym("u_" + str(k), self.nu)
            w_u += [u_k]

            x_k_end = self.f(x_k, u_k)

            # increment cost by one stage
            J += self.l_tilde(x_k, u_k) if self.is_RRLB else self.l(x_k, u_k)

            # new NLP variable for the state
            x_k = ca.MX.sym(
                "x_" + str(k + 1), self.nx
            )  # WARNING : from here x_k represents x_{k+1}
            w_x += [x_k]

            # Add equality constraint
            g += [x_k_end - x_k]

            # compute initial guess
            w_start_x += [self.f(w_start_x[-1], w_start_u[-1])]

        if self.is_RRLB:
            J += self.F(x_k)  # here x_k represents x_N

        # Concatenate stuff
        w = w_x + w_u
        w = ca.vertcat(*w)
        g = ca.vertcat(*g)
        w_start = w_start_x + w_start_u
        w_start = ca.vertcat(*w_start)

        # bounds on NLP variables
        if self.is_RRLB:
            lbw = (
                initial_state.tolist() + [-np.inf] * (self.nx + self.nu) * self.N
            )  # lower bound for all the variables
            ubw = (
                initial_state.tolist() + [np.inf] * (self.nx + self.nu) * self.N
            )  # upper bound for all the variables
        else:
            lbw = (
                initial_state.tolist()
                + [-np.inf, -np.inf, 98.0, 92.0] * self.N
                + [3.0, -9000.0] * self.N
            )
            ubw = (
                initial_state.tolist()
                + [np.inf, np.inf, np.inf, np.inf] * self.N
                + [35.0, 0.0] * self.N
            )

        lbg = [0.0] * self.nx * self.N  # lower bound for the equality constraints
        ubg = [0.0] * self.nx * self.N  # upper bound for the equality constraints

        # Create an NLP solver
        nlp_prob = {"f": J, "x": w, "g": g}
        nlp_solver = ca.nlpsol(
            "nlp_solver",
            "ipopt",
            nlp_prob,
            {
                "print_time": 0,
                "ipopt": {
                    "sb": "yes",
                    "print_level": 1,
                    "warm_start_init_point": "yes",
                },
            },
        )

        # Solve the NLP
        sol = nlp_solver(x0=w_start, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        # retrieve the solution
        result = sol["x"].full().ravel()
        if np.isnan(result).any():
            raise ValueError("NaN in the initial prediction")

        return result

    def shift_prediction(self, prediction: np.ndarray) -> np.ndarray:
        """
        Shift the prediction of the form (x_0, x_1, ...,x_N, u_0, u_1, ..., u_{N-1}) by one stage
        """
        result = np.zeros(self.nx * (self.N + 1) + self.nu * self.N)
        # shifted old predictions
        result[0 : self.nx * self.N] = prediction[self.nx : self.nx * (self.N + 1)]
        result[
            self.nx * (self.N + 1) : self.nx * (self.N + 1) + self.nu * (self.N - 1)
        ] = prediction[self.nx * (self.N + 1) + self.nu :]
        # append new control and state
        result[self.nx * (self.N + 1) + self.nu * (self.N - 1) :] = (
            self.K @ prediction[self.nx * self.N : self.nx * (self.N + 1)]
        )
        result[self.nx * self.N : self.nx * (self.N + 1)] = (
            self.f(
                result[self.nx * (self.N - 1) : self.nx * self.N],
                result[self.nx * (self.N + 1) + self.nu * (self.N - 1) :],
            )
            .full()
            .ravel()
        )
        return result

    def preparation_phase(
        self, prediction
    ) -> Tuple[
        sparse.csc_array,
        np.ndarray,
        sparse.csc_array,
        np.ndarray,
        np.ndarray,
        float,
        float,
    ]:
        """
        Note : x_0 in l and u is to be updated at the beginning of the feedback phase
        """
        sensitivity_computation_time, condensing_time = 0.0, 0.0
        tpr = []

        # compute q =============================================================
        if self.is_RRLB:
            start = time()
            tpr = []
            for k in range(self.N + 1):
                tpr += [self.grad_B_x(prediction[self.get_state_idx(k)]).full().ravel()]
            for k in range(self.N):
                tpr += [
                    self.grad_B_u(prediction[self.get_control_idx(k)]).full().ravel()
                ]
            stop = time()
            sensitivity_computation_time += 1000.0 * (stop - start)

        start = time()

        q = (
            2
            * sparse.block_diag(
                [self.Q] * self.N + [self.P] + [self.R] * self.N, format="csc"
            )
            @ (
                prediction
                - np.concatenate(
                    (np.tile(self.xr, self.N + 1), np.tile(self.ur, self.N))
                )
            )
        )
        if self.is_RRLB:
            q += self.epsilon * np.concatenate(tpr)

        stop = time()
        condensing_time += 1000.0 * (stop - start)

        # compute P =======================================================
        if self.is_RRLB:
            start = time()
            tpr = []
            for k in range(self.N + 1):
                tpr += [self.hess_B_x(prediction[self.get_state_idx(k)]).sparse()]
            for k in range(self.N):
                tpr += [self.hess_B_u(prediction[self.get_control_idx(k)]).sparse()]
            stop = time()
            sensitivity_computation_time += 1000.0 * (stop - start)

        start = time()
        P = 4 * sparse.block_diag(
            [self.Q] * self.N + [self.P] + [self.R] * self.N, format="csc"
        )
        if self.is_RRLB:
            P += 2 * self.epsilon * sparse.block_diag(tpr, format="csc")
        stop = time()
        condensing_time += 1000.0 * (stop - start)

        # compute A =======================================================
        start = time()
        tpr = []
        tpr2 = []
        for k in range(self.N):
            tpr += [
                self.jac_f_x(
                    prediction[self.get_state_idx(k)],
                    prediction[self.get_control_idx(k)],
                ).sparse()
            ]
            tpr2 += [
                self.jac_f_u(
                    prediction[self.get_state_idx(k)],
                    prediction[self.get_control_idx(k)],
                ).sparse()
            ]
        stop = time()
        sensitivity_computation_time += 1000.0 * (stop - start)

        start = time()

        Ax = sparse.diags(
            [[1.0] * self.nx + [-1.0] * self.nx * self.N], [0], format="csc"
        ) + sparse.bmat(
            [
                [None, sparse.csc_matrix((self.nx, self.nx))],
                [sparse.block_diag(mats=tpr, format="csc"), None],
            ],
            format="csc",
        )
        Bu = sparse.bmat(
            [
                [sparse.csc_matrix((self.nx, self.nu * self.N))],
                [sparse.block_diag(mats=tpr2, format="csc")],
            ],
            format="csc",
        )
        A = sparse.hstack([Ax, Bu])
        if not self.is_RRLB:
            A = sparse.vstack(
                [
                    A,
                    sparse.bmat(
                        [
                            [
                                sparse.block_diag(
                                    mats=[self.C_x] * (self.N + 1), format="csc"
                                ),
                                None,
                            ],
                            [
                                None,
                                sparse.block_diag(
                                    mats=[self.C_u] * self.N, format="csc"
                                ),
                            ],
                        ],
                        format="csc",
                    ),
                ]
            )
        stop = time()
        condensing_time += 1000.0 * (stop - start)

        # compute l and u =======================================================
        start = time()
        tpr = [-prediction[self.get_state_idx(0)]]
        for k in range(self.N):
            tpr += [
                prediction[self.get_state_idx(k + 1)]
                - self.f(
                    prediction[self.get_state_idx(k)],
                    prediction[self.get_control_idx(k)],
                )
                .full()
                .ravel()
            ]
        stop = time()
        sensitivity_computation_time += 1000.0 * (stop - start)

        start = time()
        l = np.concatenate(tpr)
        u = l
        stop = time()
        condensing_time += 1000.0 * (stop - start)
        if not self.is_RRLB:
            start = time()

            tpr = []
            for k in range(self.N + 1):
                tpr.append(self.d_x - self.C_x @ prediction[self.get_state_idx(k)])
            for k in range(self.N):
                tpr.append(self.d_u - self.C_u @ prediction[self.get_control_idx(k)])

            stop = time()

            sensitivity_computation_time += 1000.0 * (stop - start)

            start = time()
            l = np.concatenate(
                [
                    l,
                    np.full(
                        self.C_x.shape[0] * (self.N + 1) + self.C_u.shape[0] * self.N,
                        -np.inf,
                    ),
                ]
            )
            u = np.concatenate([u, np.concatenate(tpr)])

            stop = time()

            condensing_time += 1000.0 * (stop - start)

        return P, q, A, l, u, sensitivity_computation_time, condensing_time

    def get_control_idx(self, k: int) -> range:
        return range(
            self.nx * (self.N + 1) + self.nu * k,
            self.nx * (self.N + 1) + self.nu * (k + 1),
        )

    def get_state_idx(self, k: int) -> range:
        return range(self.nx * k, self.nx * (k + 1))

    def constraints_violated(self, x: np.ndarray, u: np.ndarray):
        assert x.shape == (self.nx,) and u.shape == (self.nu,)
        return np.any(self.C_x @ x > self.d_x) or np.any(self.C_u @ u > self.d_u)
