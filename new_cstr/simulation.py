from cstr import *
from graphics import *
import osqp


def run_closed_loop_simulation(
    custom_x_init=x_init,
    stop_tol=1.0e-3,
    max_nbr_feedbacks: int = 200,
    RRLB=True,
    step_by_step: bool = True,
    name: str = None,
):
    """
    Returns the total cost of the run until convergence, the number of iterations taken to reach
    convergence, and if there was any constraint violation
    """
    cstr = CSTR()

    # data for the closed loop simulation
    constraints_violated = False
    states = np.zeros((cstr.nx, 2 * max_nbr_feedbacks + 1))
    controls = np.zeros((cstr.nu, 2 * max_nbr_feedbacks + 1))
    states[:, 0] = custom_x_init
    times = np.zeros(2 * max_nbr_feedbacks + 1)
    sensitivites_computation_times = np.zeros(max_nbr_feedbacks + 1)
    condensation_times = np.zeros(max_nbr_feedbacks + 1)
    feedback_times = np.zeros(max_nbr_feedbacks)

    # Get initial prediction off-line
    prediction = cstr.initial_prediction(custom_x_init, RRLB)
    controls[:, 0] = prediction[cstr.get_control_idx(0)]

    # first preparation phase
    (
        P,
        q,
        A,
        l,
        u,
        sensitivites_computation_times[0],
        condensation_times[0],
    ) = cstr.preparation_phase(prediction)

    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, warm_start=True)

    times[1] = cstr.T / cstr.N / 1000.0

    # first feedback phase
    start = time()

    controls[:, 1] = controls[:, 0]
    states[:, 1] = cstr.f_special(states[:, 0], controls[:, 0], times[1])
    l[:cstr.nx] = states[:, 1]-prediction[cstr.get_state_idx(0)]
    u[:cstr.nx] = states[:, 1]-prediction[cstr.get_state_idx(0)]
    prob.update(l=l, u=u)
    res = prob.solve()
    if res.info.status != "solved":
        raise ValueError("OSQP did not solve the problem!")
    prediction = res.x

    stop = time()
    times[2] = 1000.0*(stop - start)
    feedback_times[0] = 1000.0*res.info.solve_time

    controls[:,2] = prediction[cstr.get_control_idx(0)]
    states[:,2] = cstr.f_special(states[:,1], controls[:,2], times[2])

    prediction = cstr.shift_prediction(prediction)
    
    
    def converged(i: int) -> bool:
        return np.sqrt(np.sum(np.square(states[:, 2*i] - cstr.xr))) < stop_tol

    i = 1
    while not converged(i) and i < max_nbr_feedbacks:
        # preparation phase
        start = time()
        (
            P,
            q,
            A,
            l,
            u,
            sensitivites_computation_times[i],
            condensation_times[i],
        ) = cstr.preparation_phase(prediction)
        prob.update(Px=P, q=q, Ax=A, l=l, u=u)
        stop = time()

        times[2*i+1] = 1000.0*(stop - start)

        # feedback phase
        start = time()
        controls[:, 2*i+1] = controls[:, 2*i]
        states[:, 2*i+1] = cstr.f_special(states[:, 2*i], controls[:, 2*i+1], times[2*i+1])
        l[:cstr.nx] = states[:, 2*i+1]-prediction[cstr.get_state_idx(0)]
        u[:cstr.nx] = states[:, 2*i+1]-prediction[cstr.get_state_idx(0)]
        prob.update(l=l, u=u)
        res = prob.solve()
        if res.info.status != "solved":
            raise ValueError("OSQP did not solve the problem!")
        prediction = res.x

        stop = time()


        # if the simulation will stop at next iteration (either because it has converged or because we
        #  have reached the maximum number of iterations), we say what happened
        # if converged(i + 1):
        #     print("Converged at iteration {}".format(i))
        # elif i == max_simulation_length - 1:
        #     sys.stderr.write("Not converged in {max_simulation_length} iterations")

        # # plot the simulation data if need be
        # if step_by_step:
        #     updatePlots(
        #         states,
        #         controls,
        #         states_pred,
        #         controls_pred,
        #         i,
        #     )
        #     plt.draw()

        i += 1

    # print("Final state: ", states[:, i])
    # print("Final control: ", controls[:, i - 1])
    # states = np.delete(states, list(range(i + 1, max_simulation_length + 1)), axis=1)
    # controls = np.delete(controls, list(range(i, max_simulation_length)), axis=1)
    # times = np.delete(times, list(range(i, max_simulation_length)), axis=0)

    # if step_by_step:
    #     plt.show()
    # elif name != None:
    #     createPlot(
    #         states,
    #         controls,
    #         states_pred,
    #         controls_pred,
    #         simulation_length=i,
    #         final=True,
    #     )
    #     if not converged(i):
    #         plt.gcf()
    #         plt.title(
    #             "NOT CONVERGED, discrepancy = {}, control discrepancy = {}".format(
    #                 np.sqrt(np.sum(np.square(states[:, i] - x_S))),
    #                 np.sqrt(np.sum(np.square(controls[:, i - 1] - cstr.ur))),
    #             )
    #         )
    #         print(
    #             "NOT CONVERGED, state discrepancy = {}, control discrepancy = {}".format(
    #                 np.sqrt(np.sum(np.square(states[:, i] - x_S))),
    #                 np.sqrt(np.sum(np.square(controls[:, i - 1] - cstr.ur))),
    #             ),
    #         )
    #     plt.savefig(
    #         os.path.join(os.path.dirname(__file__), name + ".png"),
    #         dpi=300,
    #         format="png",
    #     )

    # # compute total sum of controls
    # total_cost = 0.0
    # np.sum(np.sqrt(np.sum((np.square(states - np.array(x_S))), axis=0)))
    # for k in range(i):
    #     total_cost += l(states[:, k], controls[:, k])
    #     if not constraints_violated:
    #         for j in range(cstr.N):
    #             if not (
    #                 3.0 <= controls[0, k] <= 35.0
    #                 and -9000.0 <= controls[1, k] <= 0.0
    #                 and 98.0 <= states[2, k]
    #                 and 92.0 <= states[3, k]
    #             ):
    #                 constraints_violated = True
    # total_cost += F(states[:, i])
    # if not (98.0 <= states[2, i] and 92.0 <= states[3, i]):
    #     constraints_violated = True

    # return float(total_cost), i, constraints_violated, times
