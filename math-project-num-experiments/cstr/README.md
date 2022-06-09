# CSTR experiments

## Setup

The simplest way to get the code ready to run is to install all the dependencies
using a virtual environment at the root of the `cstr` directory :

```bash
python3 -m venv venv
```

Then you can install all the dependencies listed in the `requirements.txt` file with
the command :

```bash
source venv/bin/activate
pip install -r requirements.txt
```

> Note : CasADi cannot be installed through `pip` on Macs with Mac Silicon CPUs and
> have to be installed from sources by following these [instructions](https://github.com/casadi/casadi/wiki/InstallationMac)

## Code structure

The `cstr` directory is organized in the following way :

```bash
cstr
├── README.md
├── cstr_package
│   ├── __init__.py
│   ├── cstr.py
│   ├── graphics.py
│   └── simulation.py
├── requirements.txt
└── tests
    ├── exp_all.py
    ├── exp_analysis.py
    └── plot_closed_loop_traj.py
```

- The `cstr_package` directory contains a Python package (indicated by the presence
  of an `__init__.py` file) that contains 3 modules :

  - `cstr.py` : contains a class `CSTR` encapsulating the ODE, the constraints, the
    MPC costs, the RRLB functions, and auxiliary functions to compute an initial
    guess, the sensitivities, etc.
    This class supports three types of MPC schemes (RRLB MPC, regular MPC and
    infinite horizon MPC) indicated by the enum class `Scheme`.
  - `simulation.py` : contains two functions `run_open_loop_simulation` and `run_closed_loop_simulation`
    that both run a simulation and return the performance measure, the number of
    iterations taken to converge, and whether the constraints were violated or not
  - `graphics.py` : contains a class `CSTRAnimation` capable of plotting the
    closed-loop trajectories and MPC predictions in live.
    The live animations are not used in the current version of the tests and were created for debugging purposes at the beginning of the project.

  All the classes and functions in these modules are all parametric and are able to
  support different MPC schemes, different initial states, different reference
  points, different horizon sizes, etc.
- The `tests` directory contains a set of scripts for all the experiments presented in
  the report.

  - `exp_all.py` : runs all the experiments and saves the results in a CSV file.
  - `exp_analysis.py` : runs the analysis of the results of the experiments and
    exports the different plots.
  - `plot_closed_loop_traj.py` : plots the closed-loop trajectories of the experiments
    and saves them in a png file.
