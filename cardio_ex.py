import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
from assimulo.problem import Implicit_Problem
from assimulo.solvers.sundials import IDA
import time

# Import the cardiovascular model functions and classes
from cardio_model import CardiovascularModel, HRV

# =============================================================================
# 1. Define Uncertain Parameter Distributions and Sample
# =============================================================================
dist_p1 = cp.Uniform(0.21, 0.34)    # τ_es
dist_p2 = cp.Uniform(0.15, 0.205)   # τ_ep
dist_p3 = cp.Uniform(0.042, 0.078)    # Rmv
dist_p4 = cp.Uniform(0.0231, 0.0429)  # Zao
dist_p5 = cp.Uniform(0.777, 1.443)    # Rs
dist_p6 = cp.Uniform(0.791, 1.469)    # Csa
dist_p7 = cp.Uniform(7.7, 14.3)       # Csv
dist_p8 = cp.Uniform(1.05, 1.95)      # E_max
dist_p9 = cp.Uniform(0.021, 0.039)    # E_min

joint_dist = cp.J(dist_p1, dist_p2, dist_p3, dist_p4, dist_p5,
                  dist_p6, dist_p7, dist_p8, dist_p9)

# Set sample size (you might increase N for improved training)
N = 32
samples = joint_dist.sample(N, rule="sobol")  # shape: (9, N)

# Get a global time grid from HRV (we use it as cycle boundaries)
simulation_end_time = 43
global_time_grid = HRV(simulation_end_time)  # 1D array of event times
# Use the event times as cycle boundaries
cycle_boundaries = global_time_grid  # e.g., [t0, t1, t2, ..., t_M]
n_cycles = len(cycle_boundaries) - 1  # number of cycles available

# =============================================================================
# 2. Define Normalization Functions for Parameters and Time
# =============================================================================
lower_bounds = np.array([0.21, 0.15, 0.042, 0.0231, 0.777, 0.791, 7.7, 1.05, 0.021])
upper_bounds = np.array([0.34, 0.205, 0.078, 0.0429, 1.443, 1.469, 14.3, 1.95, 0.039])

def normalize_params(params, lower, upper):
    return (params - lower.reshape(-1, 1)) / (upper - lower).reshape(-1, 1)

def normalize_time(t, t_min, t_max):
    return (t - t_min) / (t_max - t_min)

# =============================================================================
# 3. Simulation Function for a Given Parameter Sample
# =============================================================================
def run_simulation_for_params(p_sample, time_grid):
    """
    Run the cardiovascular model simulation for a given parameter sample and time grid.
    Returns:
      t_values: simulation time grid (1D array)
      outputs: list of output arrays [pLV, psa, psv, Vlv]
    """
    u0 = np.array([8.0, 8.0, 8.0, 265.0, 0.0, 0.0, 0.0])
    udot0 = np.zeros(7)
    model = CardiovascularModel(u0, udot0, p_sample, time_grid)
    problem = Implicit_Problem(model.res, u0, udot0, 0.0)
    problem.root = model.root
    problem.handle_event = model.handle_event
    problem.nroots = 1

    solver = IDA(problem)
    solver.report_continuously = True
    solver.atol = 1e-6
    solver.rtol = 1e-6
    solver.maxord = 5
    solver.maxh = 0.01
    solver.maxsteps = 20000

    tfinal = time_grid[-1] + 0.1
    plot_saveat = np.arange(0, tfinal + 0.002, 0.002)
    t_values, y, yd = solver.simulate(tfinal, ncp_list=plot_saveat)
    t_values = np.array(t_values)
    
    # Return four outputs: pLV, psa, psv, Vlv
    outputs = [y[:, 0], y[:, 1], y[:, 2], y[:, 3]]
    return t_values, outputs

# =============================================================================
# 4. Extract Cycles and Build Multi-Cycle Training Data
# =============================================================================
# For each simulation, extract each cycle from cycle_boundaries,
# compute normalized phase and also store a normalized cycle index.
X_all = []   # to hold input columns (11 rows: 9 params, 1 cycle index, 1 phase)
Y_all_pLV = []
Y_all_psa = []
Y_all_psv = []
Y_all_Vlv = []

# Loop over simulations
for i in range(N):
    p_sample = samples[:, i]
    t_vals, outputs = run_simulation_for_params(p_sample, global_time_grid)
    
    # Loop over cycles using the common cycle boundaries
    for j in range(n_cycles):
        start = cycle_boundaries[j]
        end = cycle_boundaries[j+1]
        idx = np.where((t_vals >= start) & (t_vals < end))[0]
        if len(idx)==0:
            continue  # skip if no data in this cycle
        t_cycle = t_vals[idx]
        # Normalized phase for this cycle
        phase = (t_cycle - start) / (end - start)
        # Also compute a normalized cycle index (e.g., j normalized by n_cycles-1)
        cycle_index = j / (n_cycles - 1) if n_cycles > 1 else 0.0
        
        # Build input: repeat p_sample (9 parameters) for each time point,
        # add the cycle index (as a constant) and the phase (varying)
        n_pts = len(phase)
        X_params = np.tile(p_sample.reshape(-1,1), (1, n_pts))
        X_cycle_index = cycle_index * np.ones((1, n_pts))
        X_phase = phase.reshape(1, n_pts)
        X_cycle = np.vstack([X_params, X_cycle_index, X_phase])  # shape: (11, n_pts)
        X_all.append(X_cycle)
        
        # Append corresponding outputs for each state
        Y_all_pLV.append(outputs[0][idx])
        Y_all_psa.append(outputs[1][idx])
        Y_all_psv.append(outputs[2][idx])
        Y_all_Vlv.append(outputs[3][idx])

# Stack all training data from all cycles and all simulations
X_train = np.hstack(X_all)  # shape: (11, total_points)
Y_train_pLV = np.concatenate(Y_all_pLV)
Y_train_psa = np.concatenate(Y_all_psa)
Y_train_psv = np.concatenate(Y_all_psv)
Y_train_Vlv = np.concatenate(Y_all_Vlv)

# =============================================================================
# 5. Normalize the Parameter Inputs
# =============================================================================
# Only the first 9 rows of X_train (parameters) need normalization.
X_params_train = X_train[:9, :]
X_params_train_norm = (X_params_train - lower_bounds.reshape(-1,1)) / (upper_bounds - lower_bounds).reshape(-1,1)
# The cycle index (row 10) and phase (row 11) are already in [0,1].
X_train_norm = np.vstack([X_params_train_norm, X_train[9:11, :]])  # final input shape: (11, total_points)

# =============================================================================
# 6. Build a Surrogate Model on the Multi-Cycle Phase Domain
# =============================================================================
# The input space is now 11D (9 normalized params + normalized cycle index + phase)
joint_dist_multi = cp.J(*(cp.Uniform(0, 1) for _ in range(11)))
order = 2  # Increase order as needed
poly_basis_multi = cp.orth_ttr(order, joint_dist_multi)

# Fit surrogate models for each output variable using the normalized training inputs
pce_multi_pLV = cp.fit_regression(poly_basis_multi, X_train_norm, Y_train_pLV)
pce_multi_psa = cp.fit_regression(poly_basis_multi, X_train_norm, Y_train_psa)
pce_multi_psv = cp.fit_regression(poly_basis_multi, X_train_norm, Y_train_psv)
pce_multi_Vlv = cp.fit_regression(poly_basis_multi, X_train_norm, Y_train_Vlv)

# =============================================================================
# 7. Evaluate the Surrogate for a Test Parameter Over Multiple Cycles
# =============================================================================
# Define a test parameter sample (unnormalized) and normalize it
p_test = np.array([0.3, 0.45, 0.06, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03])
p_test_norm = (p_test - lower_bounds) / (upper_bounds - lower_bounds)

# For evaluation, we simulate a full set of cycles using the same cycle boundaries.
# We'll loop over each cycle and compute surrogate predictions.
surrogate_predictions = {"pLV":[], "psa":[], "psv":[], "Vlv":[]}
phase_predictions = []  # to record the phase grid for each cycle
for j in range(n_cycles):
    # For cycle j, define a phase grid (e.g., 100 points)
    phase_grid = np.linspace(0, 1, 100)
    cycle_index = j / (n_cycles - 1) if n_cycles > 1 else 0.0
    # Build test input: parameters (repeated), cycle index, and phase grid.
    X_test = np.vstack([np.tile(p_test_norm.reshape(-1,1), (1, 100)),
                        cycle_index * np.ones((1,100)),
                        phase_grid.reshape(1,100)])  # shape: (11, 100)
    # Evaluate the surrogate models
    pLV_pred = pce_multi_pLV(*X_test)
    psa_pred = pce_multi_psa(*X_test)
    psv_pred = pce_multi_psv(*X_test)
    Vlv_pred = pce_multi_Vlv(*X_test)
    
    surrogate_predictions["pLV"].append(pLV_pred)
    surrogate_predictions["psa"].append(psa_pred)
    surrogate_predictions["psv"].append(psv_pred)
    surrogate_predictions["Vlv"].append(Vlv_pred)
    phase_predictions.append(phase_grid)

# =============================================================================
# 8. Plot the Surrogate Predictions for Each Cycle
# =============================================================================
# For demonstration, we plot the surrogate prediction for each cycle separately.
num_cycles_to_plot = min(n_cycles, 4)  # e.g., plot up to 4 cycles

for j in range(num_cycles_to_plot):
    plt.figure(figsize=(12,6))
    plt.plot(phase_predictions[j], surrogate_predictions["pLV"][j], '--', label='Surrogate $P_{LV}$')
    plt.plot(phase_predictions[j], surrogate_predictions["psa"][j], '--', label='Surrogate $P_{SA}$')
    plt.plot(phase_predictions[j], surrogate_predictions["psv"][j], '--', label='Surrogate $P_{SV}$')
    plt.xlabel('Cycle Phase (normalized)')
    plt.ylabel('Pressure')
    plt.title(f'Surrogate Predictions for Cycle {j+1} (Pressure)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(12,6))
    plt.plot(phase_predictions[j], surrogate_predictions["Vlv"][j], '--', label='Surrogate $V_{LV}$')
    plt.xlabel('Cycle Phase (normalized)')
    plt.ylabel('Volume')
    plt.title(f'Surrogate Predictions for Cycle {j+1} (Volume)')
    plt.legend()
    plt.grid(True)
    plt.show()
