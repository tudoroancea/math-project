# %% imports and global parameters
from time import time
from typing import Tuple

import casadi as ca
import numpy as np
import scipy.linalg as la
from scipy import sparse

x_init = np.array([1.0, 0.5, 100.0, 100.0])  # initial state


class CSTR:
    # contains dynamics, constraints, RRLB functions,

    # misc
    nx = 4  # dimension of state
    nu = 2  # dimension of control
    T = 2000.0  # time horizon
    N = 100  # number of control intervals
    M = 6  # RK4 integration steps
    epsilon = 0.05  # barrier parameter
    nz = (
        nx * (N + 1) + nu * N
    )  # dimension of the vector with all states and control variables concatenaed
    states_idx = list(range(nx))
    controls_idx = list(range(nx, nx + nu))

    # constraints
    C_x = -sparse.eye(2, nx)
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
    # delta = np.min(np.concatenate((d_x, d_u), axis=0))  # relaxation parameter

    # dynamics
    f: ca.Function
    f_jac_x: ca.Function
    f_jac_u: ca.Function

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

        f_cont = ca.Function(
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
            k1 = f_cont(new_x, u)
            k2 = f_cont(new_x + DT / 2 * k1, u)
            k3 = f_cont(new_x + DT / 2 * k2, u)
            k4 = f_cont(new_x + DT * k3, u)
            new_x = new_x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.f = ca.Function("f", [x, u], [new_x])
        self.f_jac_x = ca.Function("f_jac_x", [x, u], [ca.jacobian(new_x, x)])
        self.f_jac_u = ca.Function("f_jac_u", [x, u], [ca.jacobian(new_x, u)])

        h = ca.SX.sym("h")
        DT = h / self.M
        new_x = x
        for i in range(self.M):
            k1 = f_cont(new_x, u)
            k2 = f_cont(new_x + DT / 2 * k1, u)
            k3 = f_cont(new_x + DT / 2 * k2, u)
            k4 = f_cont(new_x + DT * k3, u)
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
        A = np.array(self.f_jac_x(self.xr, self.ur))
        B = np.array(self.f_jac_u(self.xr, self.ur))
        self.P = np.array(
            la.solve_discrete_are(
                A, B, self.Q + self.epsilon * self.M_x, self.R + self.epsilon * self.M_u
            )
        )
        self.K = (
            -la.inv(self.R + self.epsilon * self.M_u + B.T @ self.P @ B)
            @ B.T
            @ self.P
            @ A
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
        # TODO : implement check for stability assumptions
        return True, "bruh"

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
            J += self.l_tilde(x_k, u_k)

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
                "ipopt": {"sb": "yes", "max_iter": 1, "print_level": 1},
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
        bruh1 = sparse.hstack(
            [
                sparse.csc_matrix((self.nx * self.N, self.nx)),
                sparse.eye(self.nx * self.N),
                sparse.csc_matrix((self.nx * self.N, self.nu * self.N)),
            ],
            format="csc",
        )

        bruh2 = sparse.vstack(
            [
                sparse.hstack(
                    [
                        sparse.csc_matrix(
                            (self.nu * (self.N - 1), self.nx * (self.N + 1) + self.nu)
                        ),
                        sparse.eye(self.nu * (self.N - 1)),
                    ],
                    format="csc",
                ),
                sparse.hstack(
                    [
                        sparse.csc_matrix((self.nu, self.nx * self.N)),
                        self.K,
                        sparse.csc_matrix((self.nu, self.nu * self.N)),
                    ],
                    format="csc",
                ),
            ],
            format="csc",
        )

        tpr1 = bruh1.dot(prediction)
        tpr2 = bruh2.dot(prediction)
        result = np.concatenate(
            (
                tpr1,
                self.f(
                    prediction[self.get_state_idx(self.N)],
                    tpr2[-self.nu :],
                )
                .full()
                .ravel(),
                tpr2,
            ),
            axis=0,
        )

        # result = np.zeros(self.nx * (self.N + 1) + self.nu * self.N)
        # # shifted old predictions
        # result[0 : self.nx * self.N] = prediction[self.nx : self.nx * (self.N + 1)]
        # result[
        #     self.nx * (self.N + 1) : self.nx * (self.N + 1) + self.nu * (self.N - 1)
        # ] = prediction[self.nx * (self.N + 1) + self.nu :]
        # # append new control and state
        # result[self.nx * (self.N + 1) + self.nu * (self.N - 1) :] = (
        #     self.K @ prediction[self.nx * self.N : self.nx * (self.N + 1)]
        # )
        # result[self.nx * self.N : self.nx * (self.N + 1)] = (
        #     self.f(
        #         result[self.nx * (self.N - 1) : self.nx * self.N],
        #         result[self.nx * (self.N + 1) + self.nu * (self.N - 1) :],
        #     )
        #     .full()
        #     .ravel()
        # )
        return result

    def preparation_phase(
        self, prediction, RRLB=True
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
        # TODO : add different preparation phase for regular MPC
        sensitivity_computation_time, condensing_time = 0.0, 0.0

        # compute q =============================================================
        start = time()
        tpr = []
        for k in range(self.N + 1):
            tpr += [self.grad_B_x(prediction[self.get_state_idx(k)]).full().ravel()]
        for k in range(self.N):
            tpr += [self.grad_B_u(prediction[self.get_control_idx(k)]).full().ravel()]
        stop = time()
        sensitivity_computation_time += 1000.0 * (stop - start)

        start = time()
        q = sparse.block_diag(
            [self.Q] * self.N + [self.P] + [self.R] * self.N, format="csc"
        ) @ (
            prediction
            - np.concatenate((np.tile(self.xr, self.N + 1), np.tile(self.ur, self.N)))
        ) + self.epsilon * np.concatenate(
            tpr
        )
        stop = time()
        condensing_time += 1000.0 * (stop - start)

        # compute P =======================================================
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
        ) + 2 * self.epsilon * sparse.block_diag(tpr, format="csc")
        stop = time()
        condensing_time += 1000.0 * (stop - start)

        # compute A =======================================================
        start = time()
        tpr = []
        tpr2 = []
        for k in range(self.N):
            tpr += [
                self.f_jac_x(
                    prediction[self.get_state_idx(k)],
                    prediction[self.get_control_idx(k)],
                ).sparse()
            ]
            tpr2 += [
                self.f_jac_u(
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

        return P, q, A, l, u, sensitivity_computation_time, condensing_time

    def get_control_idx(self, k: int) -> np.ndarray:
        return range(
            self.nx * (self.N + 1) + self.nu * k,
            self.nx * (self.N + 1) + self.nu * (k + 1),
        )

    def get_state_idx(self, k: int) -> np.ndarray:
        return range(self.nx * k, self.nx * (k + 1))
