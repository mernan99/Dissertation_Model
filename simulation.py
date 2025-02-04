# simulation.py
import numpy as np
from assimulo.problem import Implicit_Problem
from assimulo.solvers.sundials import IDA
from multiprocessing import Pool, cpu_count
from scipy.interpolate import interp1d  # Required for interpolation
from cardio_model import CardiovascularModel

def simulate_single_trajectory(args):
    index, param_values, t_τL, x = args
    u0 = np.array([8.0, 8.0, 8.0, 265.0, 0.0, 0.0, 0.0])
    params = param_values[index, :]

    # Compute τ_ep from τ_es and Δτ
    τ_es = params[0]
    Δτ = params[1]
    τ_ep = τ_es + Δτ

    # Update params array
    params = np.insert(params, 1, τ_ep)
    params = np.delete(params, 2)  # Remove Δτ

    if τ_ep <= τ_es:
        print(f"Invalid parameters: τ_ep ({τ_ep}) should be greater than τ_es ({τ_es})")
        return None

    model = CardiovascularModel(u0, np.zeros(7), params, t_τL)
    problem = Implicit_Problem(model.res, u0, np.zeros(7), 0.0)  # Start from t=0
    problem.root = model.root
    problem.handle_event = model.handle_event
    problem.nroots = 1

    solver = IDA(problem)
    solver.report_continuously = False
    solver.atol = 1e-6
    solver.rtol = 1e-6
    solver.maxord = 5
    solver.maxh = 0.05
    solver.maxsteps = 20000

    try:
        # Simulate the model
        t, y, _ = solver.simulate(tfinal=x[-1], ncp_list=x)
    except Exception as e:
        print(f"Simulation failed for index {index} at time {x[-1]} with error: {e}")
        return None

    # Exclude the initial time point at t=0.0
    t = t[1:]
    y = y[1:, :]

    # Verify that lengths match
    if len(y) != len(x):
        print(f"After discarding initial point, length of y ({len(y)}) does not match length of x ({len(x)})")
        return None

    # Extract outputs
    pLV = y[:, 0]
    psa = y[:, 1]
    Vlv = y[:, 3]

    return pLV, psa, Vlv

def simulate_ensemble(param_values, t_τL, x, trajectories=None):
    if trajectories is None:
        trajectories = param_values.shape[0]
    num_time_points = len(x)
    num_trajectories = trajectories

    # Prepare arguments for each process
    args = [(i, param_values, t_τL, x) for i in range(num_trajectories)]

    # Use multiprocessing pool to parallelize simulations
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(simulate_single_trajectory, args)

    # Filter out invalid results
    valid_results = [(i, res) for i, res in enumerate(results) if res is not None]

    if len(valid_results) == 0:
        raise RuntimeError("No valid simulations available for Sobol analysis.")

    num_valid = len(valid_results)
    outputs = np.zeros((3 * num_time_points, num_valid))

    # Combine results into outputs array
    for idx, (i, (pLV, psa, Vlv)) in enumerate(valid_results):
        outputs[:num_time_points, idx] = pLV
        outputs[num_time_points:2 * num_time_points, idx] = psa
        outputs[2 * num_time_points:, idx] = Vlv

    return outputs

