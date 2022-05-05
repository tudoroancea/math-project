import numpy as np
import casadi as ca


class RRLB:
    # user defined
    N: int
    n_x: int
    n_u: int
    x_S: np.ndarray
    u_S: np.ndarray
    Q: np.array
    R: np.array
    f: ca.Function
    epsilon: float
    C_x: np.ndarray
    C_u: np.ndarray
    d_x: np.ndarray
    d_u: np.ndarray

    # internally determined
    q_x: int
    q_u: int
    A: np.ndarray
    B: np.ndarray
    P: np.ndarray
    K: np.ndarray
    l: ca.Function
    F: ca.Function
    B_x: ca.function
    B_u: ca.function
    M_x: np.ndarray
    M_u: np.ndarray
    delta: float

    def __init__(
        self,
        N: int,
        n_x: int,
        n_u: int,
        x_S: np.ndarray,
        u_S: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        f: ca.Function,
        epsilon: float,
        C_x: np.ndarray,
        C_u: np.ndarray,
        d_x: np.ndarray,
        d_u: np.ndarray,
    ):
        assert N >= 1
        assert n_x >= 1
        assert n_u >= 1
        assert x_S.shape == (n_x, 1)
        assert u_S.shape == (n_u, 1)
        assert Q.ndim == 2 and Q.shape == (
            n_x,
            n_x,
        ), "Q does not have shape ({},{}) but {}".format(n_x, n_x, Q.shape)
        assert not np.any(np.linalg.eigvals(Q) <= 0.0), "Q is not positive definite"
        assert R.ndim == 2 and R.shape == (
            n_u,
            n_u,
        ), "R does not have shape ({},{}) but {}".format(n_u, n_u, R.shape)
        assert not np.any(np.linalg.eigvals(R) <= 0.0), "R is not positive definite"
        assert epsilon > 0.0, "barrier parameter epsilon must be positive"
        assert C_x.ndim == 2 and C_x.shape[1] == n_x
        self.q_x = C_x.shape[0]
        assert d_x.ndim == 2 and d_x.shape == (n_x, 1)
        assert C_u.ndim == 2 and C_u.shape[1] == n_u
        self.q_u = C_u.shape[0]
        assert d_u.ndim == 2 and d_u.shape == (n_u, 1)
        self.N = N
        self.n_x = n_x
        self.n_u = n_u
        self.Q = Q
        self.R = R
        self.f = f
        self.epsilon = epsilon
        self.C_x = C_x
        self.C_u = C_u
        self.d_x = d_x
        self.d_u = d_u

        self.delta = 2

        # determine barrier functions


    def construct_rrlb_functions(self):
        z = ca.SX.sym("z")
        beta = beta = ca.Function(
            "beta", [z], [0.5 * (((z - 2 * self.delta) / self.delta) ** 2 - 1) - ca.log(self.delta)]
        )
        