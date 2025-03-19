import numpy as np
import matplotlib.pyplot as plt
import openturns as ot

from cardio_model import baseline_run_and_plot
from surrogate import build_pce_surrogates_openturns, evaluate_pce_model_openturns

###################################
# Approach A:
# 1) Keep the full model outputs for 0–35s.
# 2) Downsample the training data for building the surrogate.
# 3) Filter the original data only at plotting time.
###################################

# Step 1: Generate simulation data
p = [0.3, 0.45, 0.06, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]  # Baseline parameters
simulation_end_time = 35  # Simulation duration

# Generate full data (0–35s)
X_train_full, Y_train_list_full, t_values_full = baseline_run_and_plot()

###################################
# Step 2: Downsample the data to create training data for the surrogate
###################################

num_samples = 500  # Choose how many samples for training

# Create random parameter sets for X_train
# (these do NOT depend on t-values, they represent different param variations)
# If you are only using baseline, replace with a tile of the baseline p.

bounds_list = [
    (0.2, 0.5),  # param_0
    (0.3, 0.6),  # param_1
    (0.05, 0.1), # param_2
    (0.02, 0.05),# param_3
    (0.8, 1.5),  # param_4
    (0.9, 1.3),  # param_5
    (8, 15),     # param_6
    (1.0, 2.0),  # param_7
    (0.01, 0.05) # param_8
]

X_train = np.random.uniform(
    low=[b[0] for b in bounds_list],
    high=[b[1] for b in bounds_list],
    size=(num_samples, len(bounds_list))
)

# For demonstration, we also downsample the original Y data by selecting num_samples time points
indices = np.linspace(0, len(t_values_full) - 1, num_samples, dtype=int)

# Downsample the Y data accordingly (just an example)
Y_train_list = [
    Y_train_list_full[0][indices],  # pLV downsampled
    Y_train_list_full[1][indices],  # e.g. pSA
    Y_train_list_full[2][indices],  # Vlv
]

###################################
# Step 3: Build the PCE surrogate with the training data
###################################

polynomial_degree = 3  # Adjust as needed
meta_models, distribution = build_pce_surrogates_openturns(
    X_train, Y_train_list, bounds_list,
    polynomial_degree=polynomial_degree
)

###################################
# Step 4: Evaluate surrogate model predictions on new param sets (demo)
###################################

X_test = X_train[:100, :]  # Or generate new random param sets
predictions = evaluate_pce_model_openturns(meta_models, X_test)

###################################
# Step 5: Filter the ORIGINAL data (not the downsampled) for plotting (33-35s)
###################################

time_mask = (t_values_full >= 33) & (t_values_full <= 35)
filtered_t = t_values_full[time_mask]
filtered_pLV = Y_train_list_full[0][time_mask]
filtered_Vlv = Y_train_list_full[2][time_mask]

###################################
# Step 6: Plot results in the 33–35s range
###################################

# a) pLV
plt.figure(figsize=(10, 5))
plt.plot(filtered_t, filtered_pLV, label='True pLV', color='blue')
plt.plot(
    np.linspace(33, 35, len(predictions[:, 0])),
    predictions[:, 0],
    label='Surrogate pLV', linestyle='dashed', color='red'
)
plt.xlabel("Time (s)")
plt.ylabel("Left Ventricular Pressure (pLV)")
plt.title("Surrogate Model vs. True Model for pLV (33-35s)")
plt.legend()
plt.grid(True)
plt.xlim(33, 35)
plt.show()

# b) Vlv
plt.figure(figsize=(10, 5))
plt.plot(filtered_t, filtered_Vlv, label='True Vlv', color='blue')
plt.plot(
    np.linspace(33, 35, len(predictions[:, 2])),
    predictions[:, 2],
    label='Surrogate Vlv', linestyle='dashed', color='red'
)
plt.xlabel("Time (s)")
plt.ylabel("Left Ventricular Volume (Vlv)")
plt.title("Surrogate Model vs. True Model for Vlv (33-35s)")
plt.legend()
plt.grid(True)
plt.xlim(33, 35)
plt.show()
