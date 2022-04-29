from cstr import *

# run closed loop simulation of RRLB and regular NMPC to compare behaviors =======================
print("RRLB MPC : \n==========================")
run_closed_loop_simulation(
    max_simulation_length=simulation_length,
    method=solve_rrlb_mpc,
    step_by_step=False,
    name="rrlb_mpc",
    solver="ipopt",
    opts={
        "print_time": 0,
        "ipopt": {"sb": "yes", "print_level": 0, "max_iter": 1, "max_cpu_time": 10.0},
    },
)
print("MPC : \n==========================")
run_closed_loop_simulation(
    max_simulation_length=simulation_length,
    method=solve_mpc,
    step_by_step=False,
    name="mpc",
    solver="ipopt",
    opts={
        "print_time": 0,
        "ipopt": {"sb": "yes", "print_level": 0, "max_cpu_time": 10.0},
    },
)
