import sys
from time import time

import numpy as np
from osqp import OSQP

from .cstr import *
from .graphics import *


def run_open_loop_simulation(
    custom_x_init: np.ndarray = np.array([1.0, 0.5, 100.0, 100.0]),
    xr: np.ndarray = xr1,
    ur: np.ndarray = ur1,
    scheme: Scheme = Scheme.INFINITE_HORIZON,
    N: int = 100,
    stop_tol: float = 1.0e-3,
    verbose: bool = False,
) -> Tuple[float, int, bool]:
    """
    Runs a open-loop simulation (i.e. only computes the initial prediction) and returns
    the performance measure, the number of iterations taken to reach convergence within
    tolerance stop_tol (returns the horizon size if it has not converged), and if the
    constraints were violated or not.
    """
    # declare the CSTR instance
    cstr = CSTR(
        xr=xr,
        ur=ur,
        scheme=scheme,
        N=N,
    )

    # Compute initial prediction off-line with great precision (use fully converged IP
    # instead of RTI)
    start = time()
    try:
        prediction = cstr.initial_prediction(custom_x_init)
    except ValueError:
        return 0.0, 0, False
    stop = time()
    if verbose:
        print(f"Initial prediction computed in {1000.0 * (stop - start):.5f} ms")

    # compute the return values
    prediction = np.concatenate(
        (
            np.reshape(
                prediction[: (cstr.N + 1) * cstr.nx], (cstr.nx, cstr.N + 1), order="F"
            ),
            np.concatenate(
                (
                    np.reshape(
                        prediction[-cstr.N * cstr.nu :], (cstr.nu, cstr.N), order="F"
                    ),
                    np.zeros((cstr.nu, 1)),
                ),
                axis=1,
            ),
        ),
        axis=0,
    )
    k = 0
    constraints_violated = False
    total_cost = 0.0
    while (
        np.max(np.abs((prediction[cstr.states_idx, k] - xr) / xr)) >= stop_tol
        and k <= cstr.N
    ):
        total_cost += cstr.l(
            prediction[cstr.states_idx, k], prediction[cstr.controls_idx, k]
        )
        if not constraints_violated and cstr.constraints_violated(
            prediction[cstr.states_idx, k], prediction[cstr.controls_idx, k]
        ):
            constraints_violated = True
        k += 1

    return float(total_cost), k, constraints_violated


def run_closed_loop_simulation(
    custom_x_init: np.ndarray = np.array([1.0, 0.5, 100.0, 100.0]),
    xr: np.ndarray = xr1,
    ur: np.ndarray = ur1,
    N: int = 100,
    scheme: Scheme = Scheme.RRLB,
    stop_tol: float = 1.0e-3,
    max_nbr_feedbacks: int = 250,
    graphics: bool = True,
    verbose: bool = False,
):
    """
    Runs a closed-loop simulation of CSTR by applying the RTI method to solve the MPC.

    Returns the total cost of the run until convergence, the number of iterations taken
    to reach convergence, if there was any constraint violation, and the runtimes of
    each step of RTI : sensitivity computation, condensation, and QP solving.
    """
    if scheme == Scheme.INFINITE_HORIZON:
        raise ValueError(
            "Infinite horizon scheme not supported for closed-loop simulation"
        )

    # declare the CSTR instance
    cstr = CSTR(
        xr=xr,
        ur=ur,
        scheme=scheme,
        N=N,
    )

    # Compute initial prediction off-line with great precision (use fully converged IP
    # instead of RTI)
    start = time()
    try:
        prediction = cstr.initial_prediction(custom_x_init)
    except ValueError:
        return 0.0, 0, False, np.empty(0), np.empty(0), np.empty(0)
    stop = time()
    if verbose:
        print(f"Initial prediction computed in {1000.0 * (stop - start):.5f} ms")

    # data for the closed loop simulation
    data = np.zeros((cstr.nx + cstr.nu, 2 * max_nbr_feedbacks + 1))
    data[cstr.states_idx, 0] = custom_x_init
    data[cstr.controls_idx, 0] = prediction[cstr.get_control_idx(0)]
    times = np.zeros(
        2 * max_nbr_feedbacks + 1
    )  # time elapsed between two time points : times[i] = t_i-t_{i-1} and times[0]=0.0

    sensitivities_computation_times = np.zeros(max_nbr_feedbacks)
    condensation_times = np.zeros(max_nbr_feedbacks)
    solve_times = np.zeros(max_nbr_feedbacks)
    current_reference_warm_start = np.concatenate(
        (np.tile(xr, cstr.N + 1), np.tile(ur, cstr.N))
    )

    # generate C-code since now we don't need the symbolic expressions anymore
    # cstr.gencode()

    # first preparation phase =================================================================
    (
        P,
        q,
        A,
        l,
        u,
        sensitivities_computation_times[0],
        condensation_times[0],
    ) = cstr.preparation_phase(prediction)

    prob = OSQP()
    prob.setup(P, q, A, l, u, warm_start=True, verbose=False)
    prob.warm_start(current_reference_warm_start - prediction)

    times[1] = cstr.T / cstr.N / 1000.0

    # first feedback phase =================================================================
    # get current data
    data[cstr.controls_idx, 1] = data[cstr.controls_idx, 0]
    data[cstr.states_idx, 1] = (
        cstr.f_special(data[cstr.states_idx, 0], data[cstr.controls_idx, 0], times[1])
        .full()
        .ravel()
    )

    start = time()

    # solve the QP problem
    l[: cstr.nx] = data[cstr.states_idx, 1] - prediction[cstr.get_state_idx(0)]
    u[: cstr.nx] = data[cstr.states_idx, 1] - prediction[cstr.get_state_idx(0)]
    prob.update(l=l, u=u)
    res = prob.solve()
    if res.info.status != "solved":
        raise ValueError("OSQP did not solve the problem! exitflag ", res.info.status)
    prediction += res.x

    stop = time()

    times[2] = 1000.0 * (stop - start)
    solve_times[0] = 1000.0 * res.info.solve_time

    # update the states and constraints
    data[cstr.controls_idx, 2] = prediction[cstr.get_control_idx(0)]
    data[cstr.states_idx, 2] = (
        cstr.f_special(data[cstr.states_idx, 1], data[cstr.controls_idx, 2], times[2])
        .full()
        .ravel()
    )

    def converged(index: int) -> bool:
        return np.max(np.abs((data[cstr.states_idx, 2 * index] - xr) / xr)) < stop_tol

    i = 1
    while not converged(i) and i < max_nbr_feedbacks:
        # preparation phase ========================================================
        prediction = cstr.shift_prediction(prediction)
        (
            P,
            q,
            A,
            l,
            u,
            sensitivities_computation_times[i],
            condensation_times[i],
        ) = cstr.preparation_phase(prediction)

        prob = OSQP()
        prob.setup(P, q, A, l, u, warm_start=True, verbose=False)
        prob.warm_start(current_reference_warm_start - prediction)

        times[2 * i + 1] = cstr.T / cstr.N - times[2 * i]

        # feedback phase ===========================================================
        data[cstr.controls_idx, 2 * i + 1] = data[cstr.controls_idx, 2 * i]
        data[cstr.states_idx, 2 * i + 1] = (
            cstr.f_special(
                data[cstr.states_idx, 2 * i],
                data[cstr.controls_idx, 2 * i],
                times[2 * i + 1],
            )
            .full()
            .ravel()
        )

        start = time()

        l[: cstr.nx] = (
            data[cstr.states_idx, 2 * i + 1] - prediction[cstr.get_state_idx(0)]
        )
        u[: cstr.nx] = (
            data[cstr.states_idx, 2 * i + 1] - prediction[cstr.get_state_idx(0)]
        )
        prob.update(l=l, u=u)
        res = prob.solve()
        if res.info.status != "solved":
            raise ValueError(
                "OSQP did not solve the problem! exitflag ", res.info.status
            )
        prediction += res.x

        stop = time()

        times[2 * i + 2] = 1000.0 * (stop - start)
        solve_times[i] = 1000.0 * res.info.solve_time

        data[cstr.controls_idx, 2 * i + 2] = prediction[cstr.get_control_idx(0)]
        data[cstr.states_idx, 2 * i + 2] = (
            cstr.f_special(
                data[cstr.states_idx, 2 * i + 1],
                data[cstr.controls_idx, 2 * i + 1],
                times[2 * i + 2],
            )
            .full()
            .ravel()
        )

        # if the simulation will stop at next iteration (either because it has converged or because we
        #  have reached the maximum number of iterations), we say what happened
        if verbose:
            if converged(i + 1):
                print("Converged at iteration {}".format(i))
            elif i == max_nbr_feedbacks - 1:
                sys.stderr.write(
                    "Not converged after {} feedbacks.\n\t(relative) state discrepancy : {}\n\t(relative) control discrepancy : {}\n".format(
                        max_nbr_feedbacks,
                        np.linalg.norm((data[cstr.states_idx, -1] - cstr.xr) / cstr.xr),
                        np.linalg.norm(
                            (data[cstr.controls_idx, -1] - cstr.ur) / cstr.ur
                        ),
                    )
                )

        i += 1

    data = np.delete(data, list(range(2 * i + 1, data.shape[1])), axis=1)
    times = np.delete(times, list(range(2 * i + 1, times.shape[0])), axis=0)
    solve_times = np.delete(solve_times, list(range(i, solve_times.shape[0])), axis=0)
    sensitivities_computation_times = np.delete(
        sensitivities_computation_times,
        list(range(i, sensitivities_computation_times.shape[0])),
        axis=0,
    )
    condensation_times = np.delete(
        condensation_times, list(range(i, condensation_times.shape[0])), axis=0
    )

    if verbose:
        print("Final state: ", data[cstr.states_idx, -1])
        print("Final control: ", data[cstr.controls_idx, -1])

    # create the simulation but don't show it yet, show it instead after all the
    # treatments outside this function are done
    if graphics:
        CSTRAnimation(cstr, np.concatenate((xr, ur)), data, prediction, times)

    # compute the performance measure (total cost along the closed loop trajectory)
    constraints_violated = False
    total_cost = cstr.l(data[cstr.states_idx, 0], data[cstr.controls_idx, 0])
    k = 1
    while (
        k < data.shape[1]
        and np.max(np.abs((data[cstr.states_idx, k] - xr) / xr)) >= stop_tol
    ):
        total_cost += cstr.l(data[cstr.states_idx, k], data[cstr.controls_idx, k])
        if not constraints_violated and cstr.constraints_violated(
            data[cstr.states_idx, k], data[cstr.controls_idx, k]
        ):
            constraints_violated = True

        k += 2

    return (
        float(total_cost),
        i,
        constraints_violated,
        sensitivities_computation_times,
        condensation_times,
        solve_times,
    )
