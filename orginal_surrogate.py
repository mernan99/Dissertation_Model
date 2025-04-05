import numpy as np
import matplotlib.pyplot as plt
import openturns as ot

from cardio_model import baseline_run_and_plot
from surrogate import build_pce_surrogates_openturns, evaluate_pce_model_openturns



p = [0.3, 0.45, 0.06, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]  # Baseline parameters
simulation_end_time = 35  # Simulation duration

# Generate full data (0–35s)
X_train_full, Y_train_list_full, t_values_full = baseline_run_and_plot()


num_samples = 2000  # Choose how many samples for training

# Create random parameter sets for X_train
# bounds_list = [
#     (0.21, 0.34),  # param_0
#     (0.15, 0.6),  # param_1
#     (0.05, 0.1), # param_2
#     (0.02, 0.05),# param_3
#     (0.8, 1.5),  # param_4
#     (0.9, 1.3),  # param_5
#     (8, 15),     # param_6
#     (1.0, 2.0),  # param_7
#     (0.01, 0.05) # param_8
# ]

bounds_list = [
    (0.21, 0.34),
    (0.15, 0.205),
    (0.042, 0.078),
    (0.0231, 0.0429),
    (0.777, 1.443),
    (0.791, 1.469),
    (7.7, 14.3),
    (1.05, 1.95),
    (0.021, 0.039),
]

X_train = np.random.uniform(
    low=[b[0] for b in bounds_list],
    high=[b[1] for b in bounds_list],
    size=(num_samples, len(bounds_list))
)

# For demonstration, we also downsample the original Y data by selecting num_samples time points
indices = np.linspace(0, len(t_values_full) - 1, num_samples, dtype=int)

Y_train_list = [
    Y_train_list_full[0][indices],  
    Y_train_list_full[1][indices],  
    Y_train_list_full[2][indices],  
]


# Build the PCE surrogate with the training data


polynomial_degree = 5  # Adjust as needed
meta_models, distribution = build_pce_surrogates_openturns(
    X_train, Y_train_list, bounds_list,
    polynomial_degree=polynomial_degree
)
X_test = X_train[:2000, :]  # Or generate new random param sets
predictions = evaluate_pce_model_openturns(meta_models, X_test)


# Filter the ORIGINAL data (not the downsampled) for plotting (33-35s)


time_mask = (t_values_full >= 33) & (t_values_full <= 43)
filtered_t = t_values_full[time_mask]
filtered_pLV = Y_train_list_full[0][time_mask]
filtered_Vlv = Y_train_list_full[2][time_mask]


# Plot results in the 33–35s range


# a) pLV
plt.figure(figsize=(10, 5))
plt.plot(filtered_t, filtered_pLV, label='True pLV', color='blue')
plt.plot(
    np.linspace(33, 43, len(predictions[:, 0])),
    predictions[:, 0],
    label='Surrogate pLV', linestyle='dashed', color='red'
)
plt.xlabel("Time (s)")
plt.ylabel("Left Ventricular Pressure (pLV)")
plt.title("Surrogate Model vs. True Model for pLV (33-35s)")
plt.legend()
plt.grid(True)
plt.xlim(33, 43)
plt.show()

# b) Vlv
plt.figure(figsize=(10, 5))
plt.plot(filtered_t, filtered_Vlv, label='True Vlv', color='blue')
plt.plot(
    np.linspace(33, 43, len(predictions[:, 2])),
    predictions[:, 2],
    label='Surrogate Vlv', linestyle='dashed', color='red'
)
plt.xlabel("Time (s)")
plt.ylabel("Left Ventricular Volume (Vlv)")
plt.title("Surrogate Model vs. True Model for Vlv (33-35s)")
plt.legend()
plt.grid(True)
plt.xlim(33, 43)
plt.show()

# print(ot.__version__)
# print(dir(ot))
