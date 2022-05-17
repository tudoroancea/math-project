# %% imports and global parameters
import sys
from time import time
from typing import Tuple

import casadi as ca
import numpy as np
import scipy.linalg as la
from scipy import sparse


class CSTR:
    # contains dynamics, constraints, RRLB functions,

    # misc
    nx = 4  # dimension of state
    nu = 2  # dimension of control
    T = 1500.0  # time horizon
    N = 70  # number of control intervals
    M = 6  # RK4 integration steps
    epsilon = 0.1  # barrier parameter
    nz = (
        nx * (N + 1) + nu * N
    )  # dimension of the vector with all states and control variables concatenaed
    states_idx = list(range(nx))
    controls_idx = list(range(nx, nx + nu))

    # constraints
    C_x = sparse.hstack([sparse.csc_matrix((2, 2)), -sparse.eye(2)])
    C_u = sparse.vstack(
        (sparse.diags([[1.0, -1.0]], [0]), sparse.diags([[1.0, -1.0]], [0]))
    )
    d_x = np.array([-98.0, -92.0])
    d_u = np.array([35.0, -3.0, 0.0, 9000.0])

    # reference points
    xr: np.ndarray
    ur: np.ndarray

    # RRLB functions
    B_x: ca.Function
    B_u: ca.Function
    grad_B_x: ca.Function
    grad_B_u: ca.Function
    hess_B_x: ca.Function
    hess_B_u: ca.Function
    M_x: np.ndarray
    M_u: np.ndarray
    delta = 10.0  # relaxation parameter
    # delta = np.min(np.abs(np.concatenate((d_x, d_u), axis=0)))  # relaxation parameter

    # dynamics
    f: ca.Function
    f_cont: ca.Function
    jac_f_x: ca.Function
    jac_f_u: ca.Function
    A: np.ndarray
    B: np.ndarray

    # costs
    Q = sparse.diags([[0.2, 1.0, 0.5, 0.2]], [0])
    R = sparse.diags([[0.5, 5.0e-7]], [0])
    P: np.ndarray
    K: np.ndarray
    l: ca.Function
    l_tilde: ca.Function
    F: ca.Function

    def __init__(
        self,
        initial_xr: np.ndarray = np.array(
            [
                2.1402105301746182e00,
                1.0903043613077321e00,
                1.1419108442079495e02,
                1.1290659291045561e02,
            ]
        ),
        initial_ur: np.ndarray = np.array([14.19, -1113.50]),
    ) -> None:
        # dynamics
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
        self.jac_f_x = ca.Function("f_jac_x", [x, u], [ca.jacobian(new_x, x)])
        self.jac_f_u = ca.Function("f_jac_u", [x, u], [ca.jacobian(new_x, u)])

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

        self.set_reference(initial_xr, initial_ur)

    def compute_RRLB(self) -> None:
        # reimplement everything in a more generic way
        x = ca.SX.sym("x", self.nx)
        u = ca.SX.sym("u", self.nu)
        z = ca.SX.sym("z")
        beta = ca.Function(
            "beta",
            [z],
            [0.5 * (((z - 2 * self.delta) / self.delta) ** 2 - 1) - ca.log(self.delta)],
        )
        B_x_1 = ca.if_else(
            x[2] - 98.0 > self.delta,
            ca.log(self.xr[2] - 98.0) - ca.log(x[2] - 98.0),
            beta(x[2] - 98.0),
        )
        B_x_2 = ca.if_else(
            x[3] - 92.0 > self.delta,
            ca.log(self.xr[3] - 92.0) - ca.log(x[3] - 92.0),
            beta(x[3] - 92.0),
        )
        grad_B_x_1 = ca.Function("grad_B_x_1", [x], [ca.jacobian(B_x_1, x)])
        grad_B_x_2 = ca.Function("grad_B_x_2", [x], [ca.jacobian(B_x_2, x)])
        w_x_1 = -1.0
        w_x_2 = -1.0
        self.B_x = ca.Function(
            "B_x",
            [x],
            [(1.0 + w_x_1) * B_x_1 + (1.0 + w_x_2) * B_x_2],
        )
        self.grad_B_x = ca.Function("grad_B_x", [x], [ca.jacobian(self.B_x(x), x)])
        self.hess_B_x = ca.Function("hess_B_x", [x], [ca.hessian(self.B_x(x), x)[0]])
        self.M_x = self.hess_B_x(self.xr).full()

        B_u_1 = ca.if_else(
            35.0 - u[0] > self.delta,
            ca.log(35.0 - self.ur[0]) - ca.log(35.0 - u[0]),
            beta(35.0 - u[0]),
        )
        B_u_2 = ca.if_else(
            u[0] - 3.0 > self.delta,
            ca.log(self.ur[0] - 3.0) - ca.log(u[0] - 3.0),
            beta(u[0] - 3.0),
        )
        B_u_3 = ca.if_else(
            -u[1] > self.delta, ca.log(-self.ur[1]) - ca.log(-u[1]), beta(-u[1])
        )
        B_u_4 = ca.if_else(
            9000.0 + u[1] > self.delta,
            ca.log(9000.0 + self.ur[1]) - ca.log(9000.0 + u[1]),
            beta(9000.0 + u[1]),
        )
        grad_B_u_1 = ca.Function("grad_B_u_1", [u], [ca.jacobian(B_u_1, u)])
        grad_B_u_2 = ca.Function("grad_B_u_2", [u], [ca.jacobian(B_u_2, u)])
        grad_B_u_3 = ca.Function("grad_B_u_3", [u], [ca.jacobian(B_u_3, u)])
        grad_B_u_4 = ca.Function("grad_B_u_4", [u], [ca.jacobian(B_u_4, u)])
        w_u_1 = 0.0
        w_u_2 = -grad_B_u_1(self.ur)[0] / grad_B_u_2(self.ur)[0] - 1.0
        w_u_3 = 0.0
        w_u_4 = -grad_B_u_3(self.ur)[1] / grad_B_u_4(self.ur)[1] - 1.0

        self.B_u = ca.Function(
            "B_u",
            [u],
            [
                (1.0 + w_u_1) * B_u_1
                + (1.0 + w_u_2) * B_u_2
                + (1.0 + w_u_3) * B_u_3
                + (1.0 + w_u_4) * B_u_4
            ],
        )
        self.grad_B_u = ca.Function("grad_B_u", [u], [ca.jacobian(self.B_u(u), u)])
        self.hess_B_u = ca.Function("hess_B_u", [u], [ca.hessian(self.B_u(u), u)[0]])
        self.M_u = self.hess_B_u(self.ur).full()

    def compute_terminal_cost(self) -> None:
        self.A = np.array(self.jac_f_x(self.xr, self.ur))
        self.B = np.array(self.jac_f_u(self.xr, self.ur))
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

    def set_reference(self, xr: np.ndarray, ur: np.ndarray) -> None:
        self.xr = xr
        self.ur = ur
        self.compute_RRLB()
        self.compute_terminal_cost()

        x = ca.SX.sym("x", self.nx)
        u = ca.SX.sym("u", self.nu)
        self.l = ca.Function(
            "l",
            [x, u],
            [
                # ca.bilin(self.Q, x - self.xr, x - self.xr)
                # + ca.bilin(self.R, u - self.ur, u - self.ur)
                0.2 * (x[0] - self.xr[0]) ** 2
                + 1.0 * (x[1] - self.xr[1]) ** 2
                + 0.5 * (x[2] - self.xr[2]) ** 2
                + 0.2 * (x[3] - self.xr[3]) ** 2
                + 0.5 * (u[0] - self.ur[0]) ** 2
                + 5.0e-7 * (u[1] - self.ur[1]) ** 2
            ],
        )
        self.l_tilde = ca.Function(
            "l_tilde",
            [x, u],
            [self.l(x, u) + self.epsilon * self.B_x(x) + self.epsilon * self.B_u(u)],
        )
        self.F = ca.Function("F", [x], [ca.bilin(self.P, (x - self.xr), (x - self.xr))])

        assumptions_verified, err_msg = self.check_stability_assumptions()
        assert assumptions_verified, err_msg

    def check_stability_assumptions(self):
        err = (False,)
        err_msg = ""
        # ====================================================================================================
        # Check that gradients of barrier functions are 0 at target state/control
        # ====================================================================================================
        if ca.norm_2(self.grad_B_x(self.xr)) >= 1e-6:
            err = True
            err_msg += "ERROR : grad_B_x(xr) = {}, norm = {}\n".format(
                self.grad_B_x(self.xr), ca.norm_2(self.grad_B_x(self.xr))
            )

        if ca.norm_2(self.grad_B_u(self.ur)) >= 1e-6:
            err = True
            err_msg += "ERROR : grad_B_u(ur) = {}, norm = {}\n".format(
                self.grad_B_u(self.ur), ca.norm_2(self.grad_B_u(self.ur))
            )

        # ====================================================================================================
        # Check controllability/stabilizability
        # ====================================================================================================
        def check_controllability(my_A, my_B):
            controllability_matrix = np.tile(my_B, (1, self.N))
            for i in range(1, self.N):
                controllability_matrix[:, i * self.nu : (i + 1) * self.nu] = (
                    my_A @ controllability_matrix[:, (i - 1) * self.nu : i * self.nu]
                )
            rank = np.linalg.matrix_rank(controllability_matrix)
            if rank != self.nx:
                err = True
                err_msg += "\tERROR : (A,B) is not controllable, rank = {}\n".format(
                    rank
                )

        def check_stabilizability(my_A, my_B, my_K):
            discrete_eig = np.linalg.eig(my_A + my_B @ my_K)[0]
            not_in_unit_disk = []
            for val in discrete_eig:
                if np.linalg.norm(val) >= 1.0:
                    not_in_unit_disk.append(np.linalg.norm(val))

            if len(not_in_unit_disk) != 0:
                err = True
                err_msg += (
                    "\tERROR : (A,B) is not stabilizable, eigenvalues = {}\n".format(
                        not_in_unit_disk
                    )
                )

        # Discrete time
        err_msg += "Discrete time stabilizability / controllability:\n"
        check_controllability(self.A, self.B)
        check_stabilizability(self.A, self.B, self.K)

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
        check_controllability(A_cont, B_cont)

        P_cont = la.solve_continuous_are(
            A_cont,
            B_cont,
            self.Q + self.epsilon * self.M_x,
            self.R + self.epsilon * self.M_u,
        )
        K_cont = -la.inv(self.R + self.epsilon * self.M_u) @ B_cont.T @ P_cont
        check_stabilizability(A_cont, B_cont, K_cont)

        # ====================================================================================================
        # Check the positive definiteness of the hessian of the objective at the target state/control
        # (for Lipschitzianity)
        # ====================================================================================================
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
        eigvalues = np.linalg.eig(
            np.array(hess_at(self.xr, ca.vertcat(*([self.ur] * self.N))))
        )[0]
        if np.any(eigvalues <= 0):
            err = True
            err_msg += "ERROR : hessian of the objective wrt controls at the origin is \
                not positive definite. Eigen values are : {}\n".format(
                eigvalues
            )

        # ====================================================================================================
        # Check positive definiteness of hessian of B_u at the target controls
        # ====================================================================================================

        eigvalues = np.linalg.eig(np.array(self.hess_B_u(ca.DM.zeros(self.nu))))[0]
        if np.any(eigvalues <= 0):
            err = True
            err_msg += "ERROR: hessian of B_u wrt controls at the origin is not \
                positive definite. Eigen values are : {}\n".format(
                eigvalues
            )

        return err, err_msg

    def initial_prediction(
        self, initial_state: np.ndarray, RRLB: bool = True
    ) -> np.ndarray:
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
        w_start_u = []

        # Formulate the NLP
        for k in range(self.N):
            # New NLP variable for the control
            u_k = ca.MX.sym("u_" + str(k), self.nu)
            w_u += [u_k]

            x_k_end = self.f(x_k, u_k)

            # increment cost by one stage
            J += self.l_tilde(x_k, u_k) if RRLB else self.l(x_k, u_k)

            # new NLP variable for the state
            x_k = ca.MX.sym(
                "x_" + str(k + 1), self.nx
            )  # WARNING : from here x_k represents x_{k+1}
            w_x += [x_k]

            # Add equality constraint
            g += [x_k_end - x_k]

            # compute initial guess
            w_start_u += [self.ur]  # [-self.K @ w_start_x[-1]]
            w_start_x += [self.f(w_start_x[-1], w_start_u[-1])]

        J += self.F(x_k)  # here x_k represents x_N

        # Concatenate stuff
        w = w_x + w_u
        w = ca.vertcat(*w)
        g = ca.vertcat(*g)
        w_start = w_start_x + w_start_u
        w_start = ca.vertcat(*w_start)

        # bounds on NLP variables
        if RRLB:
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
                    # "max_iter": 1 if RRLB else 1000,
                    "print_level": 1,
                },
            },
        )

        # Solve the NLP
        start = time()
        sol = nlp_solver(x0=w_start, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        stop = time()
        print(f"Initial prediction computed in {1000.0*(stop - start):.5f} ms")

        result = sol["x"].full().ravel()
        assert not np.isnan(result).any(), "NaNs found in initial prediction"
        return result

    def shift_prediction(self, prediction: np.ndarray) -> np.ndarray:
        """
        Shift the prediction by one stage
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
        self, prediction, RRLB: bool = True
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

        # compute q =============================================================
        if RRLB:
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
        # TODO : try multiplying and the condensing
        q = sparse.block_diag(
            [self.Q] * self.N + [self.P] + [self.R] * self.N, format="csc"
        ) @ (
            prediction
            - np.concatenate((np.tile(self.xr, self.N + 1), np.tile(self.ur, self.N)))
        ) + (
            self.epsilon * np.concatenate(tpr) if RRLB else 0.0
        )
        stop = time()
        condensing_time += 1000.0 * (stop - start)

        # compute P =======================================================
        if RRLB:
            start = time()
            tpr = []
            for k in range(self.N + 1):
                tpr += [self.hess_B_x(prediction[self.get_state_idx(k)]).sparse()]
            for k in range(self.N):
                tpr += [self.hess_B_u(prediction[self.get_control_idx(k)]).sparse()]
            stop = time()
            sensitivity_computation_time += 1000.0 * (stop - start)

        start = time()
        P = 2 * sparse.block_diag(
            [self.Q] * self.N + [self.P] + [self.R] * self.N, format="csc"
        )
        if RRLB:
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
        if not RRLB:
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
        if not RRLB:
            start = time()
            tpr = []
            for k in range(self.N + 1):
                tpr += [self.d_x - self.C_x @ prediction[self.get_state_idx(k)]]
            for k in range(self.N):
                tpr += [self.d_u - self.C_u @ prediction[self.get_control_idx(k)]]
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

    def get_control_idx(self, k: int) -> np.ndarray:
        return range(
            self.nx * (self.N + 1) + self.nu * k,
            self.nx * (self.N + 1) + self.nu * (k + 1),
        )

    def get_state_idx(self, k: int) -> np.ndarray:
        return range(self.nx * k, self.nx * (k + 1))
