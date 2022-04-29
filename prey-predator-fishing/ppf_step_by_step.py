from ppf import *

# Run closed loop simulation of RRLB MPC with live plot ==============================================
total_cost, iterations, constraints_violated = run_closed_loop_simulation(
    # custom_x_init=[1.5, 0.7],
    max_simulation_length=100,
    method=solve_rrlb_mpc,
    step_by_step=True,
    solver="ipopt",
    opts={
        "print_time": 0,
        "ipopt": {"print_level": 0, "max_iter": 1, "max_cpu_time": 10.0},
    },
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
