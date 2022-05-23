import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as sla
from osqp import OSQP

from .cstr import *
from .graphics import *


def run_closed_loop_simulation(
    custom_x_init=np.array([1.0, 0.5, 100.0, 100.0]),
    xr: np.ndarray = xr1,
    ur: np.ndarray = ur1,
    stop_tol=1.0e-3,
    RRLB=True,
    graphics: bool = True,
):
    """
    Returns the total cost of the run until convergence, the number of iterations taken to reach
    convergence, and if there was any constraint violation
    """
    cstr = CSTR(
        xr=xr,
        ur=ur,
    )
    reference_points = np.array(
        [
            np.concatenate((xr, ur)),
        ]
    ).T
    current_reference_idx = 0
    print("started with reference point n°1")
    current_reference = np.concatenate(
        (
            np.tile(
                reference_points[cstr.states_idx, current_reference_idx], cstr.N + 1
            ),
            np.tile(reference_points[cstr.controls_idx, current_reference_idx], cstr.N),
        )
    )
    max_nbr_feedbacks = int(5000.0 * reference_points.shape[1] / (cstr.T / cstr.N)) + 1

    # data for the closed loop simulation
    constraints_violated = False
    data = np.zeros((cstr.nx + cstr.nu, 2 * max_nbr_feedbacks + 1))
    data[cstr.states_idx, 0] = custom_x_init
    times = np.zeros(
        2 * max_nbr_feedbacks + 1
    )  # time elapsed between two time points : times[i] = t_i-t_{i-1} and times[0]=0.0

    sensitivites_computation_times = np.zeros(max_nbr_feedbacks)
    condensation_times = np.zeros(max_nbr_feedbacks)
    solve_times = np.zeros(max_nbr_feedbacks)

    # Get initial prediction off-line with great precision (use fully converged IP instead of RTI)
    prediction = cstr.initial_prediction(custom_x_init, RRLB)

    data[cstr.controls_idx, 0] = prediction[cstr.get_control_idx(0)]

    # generate C-code since now we don't need the symbolic expressions anymore
    cstr.gencode()

    # first preparation phase =================================================================
    (
        P,
        q,
        A,
        l,
        u,
        sensitivites_computation_times[0],
        condensation_times[0],
    ) = cstr.preparation_phase(prediction, RRLB)

    prob = OSQP()
    prob.setup(P, q, A, l, u, warm_start=True, verbose=False)
    prob.warm_start(current_reference - prediction)

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

    def converged(i: int) -> bool:
        if len(reference_points.shape) == 1 or reference_points.shape[1] == 1:
            return (
                np.sqrt(np.sum(np.square(data[cstr.states_idx, 2 * i] - cstr.xr)))
                < stop_tol
            )
        else:
            return False

    i = 1
    while not converged(i) and i < max_nbr_feedbacks:
        # change current reference points if necessary
        if int(np.sum(times[: 2 * i]) / 5000.0) > current_reference_idx:
            current_reference_idx += 1
            current_reference = np.concatenate(
                (
                    np.tile(
                        reference_points[cstr.states_idx, current_reference_idx],
                        cstr.N + 1,
                    ),
                    np.tile(
                        reference_points[cstr.controls_idx, current_reference_idx],
                        cstr.N,
                    ),
                )
            )
            cstr.set_reference(
                xr=reference_points[cstr.states_idx, current_reference_idx],
                ur=reference_points[cstr.controls_idx, current_reference_idx],
            )
            cstr.gencode()
            print("changed reference point to point n°", current_reference_idx + 1)

        # preparation phase ========================================================
        prediction = cstr.shift_prediction(prediction)
        (
            P,
            q,
            A,
            l,
            u,
            sensitivites_computation_times[i],
            condensation_times[i],
        ) = cstr.preparation_phase(prediction, RRLB)

        prob = OSQP()
        prob.setup(P, q, A, l, u, warm_start=True, verbose=False)
        prob.warm_start(current_reference - prediction)

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
        if len(reference_points.shape) == 1 or reference_points.shape[1] == 1:
            if converged(i + 1):
                print("Converged at iteration {}".format(i))
            elif i == max_nbr_feedbacks - 1:
                sys.stderr.write(
                    "Not converged after {} feedbacks.\n\tstate discrepancy : {}\n\tcontrol discrepancy : {}\n".format(
                        max_nbr_feedbacks,
                        np.linalg.norm(data[cstr.states_idx, 2 * i + 2] - cstr.xr),
                        np.linalg.norm(data[cstr.controls_idx, 2 * i + 2] - cstr.ur),
                    )
                )

        i += 1

    data = np.delete(data, list(range(2 * i + 1, data.shape[1])), axis=1)
    times = np.delete(times, list(range(2 * i + 1, times.shape[0])), axis=0)
    solve_times = np.delete(solve_times, list(range(i, solve_times.shape[0])), axis=0)
    sensitivites_computation_times = np.delete(
        sensitivites_computation_times,
        list(range(i, sensitivites_computation_times.shape[0])),
        axis=0,
    )
    condensation_times = np.delete(
        condensation_times, list(range(i, condensation_times.shape[0])), axis=0
    )

    print("Final state: ", data[cstr.states_idx, -1])
    print("Final control: ", data[cstr.controls_idx, -1])

    # create the simulation but don't show it yet, show it instead after all the treatments outside
    # this function are done
    if graphics:
        CSTRAnimation(cstr, reference_points, data, prediction, times)

    # compute the performance measure (total cost along the closed loop trajectory)
    total_cost = 0.0
    for k in range(0, 2 * i, 2):
        total_cost += cstr.l(data[cstr.states_idx, k], data[cstr.controls_idx, k])
        if not constraints_violated:
            for j in range(cstr.N):
                if not (
                    3.0 <= data[4 + 0, k] <= 35.0
                    and -9000.0 <= data[4 + 1, k] <= 0.0
                    and 98.0 <= data[2, k]
                    and 92.0 <= data[3, k]
                ):
                    constraints_violated = True
    # total_cost += data[cstr.states_idx, 2 * i] @ cstr.P @ data[cstr.states_idx, 2 * i]
    if not (98.0 <= data[2, 2 * i] and 92.0 <= data[3, 2 * i]):
        constraints_violated = True

    return (
        float(total_cost),
        i,
        constraints_violated,
        sensitivites_computation_times,
        condensation_times,
        solve_times,
    )
