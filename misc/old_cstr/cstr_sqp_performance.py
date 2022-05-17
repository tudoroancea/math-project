from cstr import *

max_sqp_iter = 100
costs = []
for max_iter in range(20, max_sqp_iter+1):
    total_cost, iterations, constraints_violated, times = run_closed_loop_simulation(
        custom_x_init=x_init,
        max_simulation_length=100,
        method=solve_rrlb_mpc,
        step_by_step=False,
        name=None,
        solver="sqpmethod",
        opts={
            "print_time": False,
            "print_header": False,
            "qpsol": "qpoases",
            "qpsol_options": {
                "printLevel": "none",
                "CPUtime": 5.0,
            },
            "print_iteration": False,
            "max_iter": max_iter,
        },
    )
    print(
        "total_cost = {}, iterations = {}, constraints_violated = {}".format(
            total_cost, iterations, constraints_violated
        )
    )
    costs.append(total_cost)

plt.figure()
plt.plot(costs)
plt.show()
