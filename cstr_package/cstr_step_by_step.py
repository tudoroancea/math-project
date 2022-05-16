from cstr import *

# Run closed loop simulation of RRLB MPC with live plot ==============================================
total_cost, iterations, constraints_violated, times = run_closed_loop_simulation(
    custom_x_init=x_init,
    max_simulation_length=150,
    method=solve_rrlb_mpc,
    step_by_step=False,
    solver="ipopt",
    opts={
        "print_time": 0,
        "ipopt": {"sb": "yes", "print_level": 0, "max_iter": 1, "max_cpu_time": 10.0},
        # "ipopt": {"sb": "yes", "print_level": 0, "max_cpu_time": 10.0},
    },
    # solver="sqpmethod",
    # opts={
    #     "print_time": False,
    #     "print_header": False,
    #     "qpsol": "qpoases",
    #     "qpsol_options": {
    #         "printLevel": "none",
    #         "CPUtime": 5.0,
    #         "sparse": True,
    #     },
    #     "print_iteration": False,
    #     "max_iter": 1,
    # },
)
print(
    "total_cost = {}, iterations = {}, constraints_violated = {}".format(
        total_cost, iterations, constraints_violated
    )
)

# Run closed loop simulation of Regular MPC with live plot ==============================================
# total_cost, iterations, constraints_violated = run_closed_loop_simulation(
#     # custom_x_init=[1.5, 0.7],
#     max_simulation_length=100,
#     method=solve_mpc,
#     step_by_step=True,
#     solver="ipopt",
#     opts={
#         "print_time": 0,
#         "ipopt": {"print_level": 0, "max_iter": 1000, "max_cpu_time": 10.0},
#     },
# )
# print(
#     "total_cost = {}, iterations = {}, constraints_violated = {}".format(
#         total_cost, iterations, constraints_violated
#     )
# )
