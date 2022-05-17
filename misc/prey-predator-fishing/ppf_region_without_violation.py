from ppf import *

# Find region without constraint violation ========================================================
x_sample = np.linspace(0.7, 1.3, 10)
y_sample = np.linspace(0.7, 1.3, 10)
constraints_violated_table = np.zeros((10, 10), dtype=bool)

for i in range(10):
    for j in range(10):
        start = time()
        (
            total_cost,
            iterations,
            constraints_violated_table[i, j],
        ) = run_closed_loop_simulation(
            custom_x_init=np.array([x_sample[i], y_sample[i]]),
            stop_tol=1.0e-3,
            max_simulation_length=1000,
            method=solve_rrlb_mpc,
            step_by_step=False,
        )
        stop = time()
        print(
            "for i={},j={} : total_cost = {}, iterations = {}, time = {} s".format(
                i, j, total_cost, iterations, stop - start
            )
        )

np.savetxt(
    "constraints_violated_ppf.csv", constraints_violated_table, delimiter=","
)