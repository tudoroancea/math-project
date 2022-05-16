from cstr import *

# run closed loop simulation of RRLB and regular NMPC to compare behaviors =======================
start = time()
print("RRLB MPC : \n==========================")
total_cost, iterations, constraints_violated, times_rrlb = run_closed_loop_simulation(
    max_simulation_length=1,
    method=solve_rrlb_mpc,
    step_by_step=False,
    name="rrlb_mpc",
    solver="ipopt",
    opts={
        "print_time": 0,
        "ipopt": {"sb": "yes", "print_level": 0, "max_cpu_time": 10.0},
    },
)
print(
    "total_cost = {}, iterations = {}, constraints_violated = {}".format(
        total_cost, iterations, constraints_violated
    )
)


print("MPC : \n==========================")
total_cost, iterations, constraints_violated, times = run_closed_loop_simulation(
    max_simulation_length=1,
    method=solve_mpc,
    step_by_step=False,
    name="mpc",
    solver="ipopt",
    opts={
        "print_time": 0,
        "ipopt": {"sb": "yes", "print_level": 0, "max_cpu_time": 10.0},
    },
)
print(
    "total_cost = {}, iterations = {}, constraints_violated = {}".format(
        total_cost, iterations, constraints_violated
    )
)
stop = time()
print("\n\nTime taken: {} seconds\n\n".format(stop - start))


print("Comparison : \n==========================")
print(
    "solve times RRLB MPC :\n\tmean={}\n\tstd={}".format(
        np.mean(times_rrlb[1:]), np.std(times_rrlb[1:])
    )
)
print(
    "solve times regular MPC :\n\tmean={}\n\tstd={}".format(
        np.mean(times), np.std(times)
    )
)
plt.figure()
plt.plot(times_rrlb, label="solve times RRLB MPC")
plt.plot(times, label="solve times regular MPC")
plt.legend()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "solve_times.png"),
    dpi=300,
    format="png",
)
plt.show()
