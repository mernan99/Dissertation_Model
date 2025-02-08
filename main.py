# # main.py
# import numpy as np
# import matplotlib.pyplot as plt

# from SALib.sample import saltelli
# from SALib.analyze import sobol

# # Import the model and HRV function from cardio_model.py
# from cardio_model import CardiovascularModel, HRV

# # Import the parallel ensemble simulation from simulation.py
# from simulation import simulate_ensemble

# # Import the PCE-building functions from surrogate.py
# from surrogate import (
#     build_pce_surrogates_openturns,
#     evaluate_pce_model_openturns,
#     sobol_analysis_with_surrogate
# )


# def baseline_run_and_plot():
#     """
#     Run the baseline (single) simulation of the cardiovascular model
#     and plot the results.
#     """
#     # Baseline parameter set: [τ_es, τ_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min]
#     p = [0.3, 0.45, 0.06, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]

#     # Simulation end time and HRV generation
#     simulation_end_time = 35
#     t_τL = HRV(simulation_end_time)

#     # Initial states
#     u0 = np.array([8.0, 8.0, 8.0, 265.0, 0.0, 0.0, 0.0])
#     udot0 = np.zeros(7)

#     # Instantiate and wrap in an Implicit_Problem for Assimulo
#     from assimulo.problem import Implicit_Problem
#     from assimulo.solvers.sundials import IDA

#     model = CardiovascularModel(u0, udot0, p, t_τL)
#     problem = Implicit_Problem(model.res, u0, udot0, 0.0)
#     problem.root = model.root
#     problem.handle_event = model.handle_event
#     problem.nroots = 1

#     # Configure the solver
#     solver = IDA(problem)
#     solver.report_continuously = True
#     solver.atol = 1e-6
#     solver.rtol = 1e-6
#     solver.maxord = 5
#     solver.maxh = 0.05
#     solver.maxsteps = 20000

#     # Simulate slightly beyond the last HRV event
#     tfinal = t_τL[-1] + 0.1
#     plot_saveat = np.arange(0, tfinal + 0.002, 0.002)

#     # Run the simulation
#     t, y, yd = solver.simulate(tfinal, ncp_list=plot_saveat)

#     # Extract outputs
#     pLV = y[:, 0]
#     psa = y[:, 1]
#     psv = y[:, 2]
#     Vlv = y[:, 3]

#     # Plot pressures between 33 and 35s
#     plt.figure(figsize=(12, 6))
#     plt.plot(t, pLV, label='P_LV')
#     plt.plot(t, psa, label='P_SA')
#     plt.plot(t, psv, label='P_SV')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Pressure')
#     plt.title('Pressures Over Time (33 - 35 s)')
#     plt.legend()
#     plt.grid(True)
#     plt.xlim(33, 35)
#     plt.show()

#     # Plot left ventricular volume between 33 and 35s
#     plt.figure(figsize=(12, 6))
#     plt.plot(t, Vlv, label='V_LV')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Volume')
#     plt.title('Left Ventricular Volume (33 - 35 s)')
#     plt.legend()
#     plt.grid(True)
#     plt.xlim(33, 35)
#     plt.show()


# def plot_sobol_heatmap(S_matrix, parameter_names, output_names, title, colormap="plasma"):
#     """
#     Utility function to plot a heatmap of Sobol indices.
#     """
#     fig, ax = plt.subplots(figsize=(8, 6))
#     heatmap = ax.imshow(S_matrix, cmap=colormap, aspect="auto", vmin=0, vmax=1)

#     ax.set_xticks(np.arange(len(output_names)))
#     ax.set_xticklabels(output_names, rotation=45, ha="right", fontsize=10)
#     ax.set_yticks(np.arange(len(parameter_names)))
#     ax.set_yticklabels(parameter_names, fontsize=10)

#     ax.set_title(title, fontsize=12)
#     ax.set_xlabel("Outputs", fontsize=10)
#     ax.set_ylabel("Parameters", fontsize=10)

#     cbar = fig.colorbar(heatmap, ax=ax, orientation="vertical", shrink=0.8, pad=0.02)
#     cbar.set_label("Sobol Index", fontsize=10)

#     # Annotate each cell with the numeric value
#     for i in range(len(parameter_names)):
#         for j in range(len(output_names)):
#             val = S_matrix[i, j]
#             ax.text(j, i, f"{val:.2f}", ha="center", va="center",
#                     color="white" if val > 0.5 else "black", fontsize=8)

#     plt.tight_layout()
#     plt.show()


# def main():
#     """
#     1) Runs a single baseline simulation + plotting.
#     2) Performs an ensemble simulation + Sobol analysis, 
#        using param sets from Saltelli's method.
#     3) Builds PCE Surrogates from the ensemble outputs 
#        and runs a surrogate-based Sobol analysis (optional).
#     """
#     # A) Baseline run and plots
#     print("\n=== Running Baseline Simulation ===")
#     baseline_run_and_plot()

#     # B) Sobol / ensemble simulation
#     print("\n=== Running Ensemble Simulations for Sensitivity ===")

#     # SALib problem definition
#     problem = {
#         'num_vars': 9,
#         'names': ['τ_es', 'Δτ', 'Rmv', 'Zao', 'Rs', 'Csa', 'Csv', 'E_max', 'E_min'],
#         'bounds': [
#             [0.21, 0.34],
#             [0.15, 0.205],
#             [0.042, 0.078],
#             [0.0231, 0.0429],
#             [0.777, 1.443],
#             [0.791, 1.469],
#             [7.7, 14.3],
#             [1.05, 1.95],
#             [0.021, 0.039],
#         ]
#     }

#     # Generate Saltelli param sets
#     N = 32
#     param_values = saltelli.sample(problem, N, calc_second_order=True)
#     print("param_values.shape:", param_values.shape)

#     # Time array from 33..35 seconds (we only care about last 2 seconds)
#     x = np.arange(33, 35, 0.002)

#     # HRV times up to 35 s
#     t_τL = HRV(35)

#     # Perform ensemble runs using the imported function
#     ensemble_data = simulate_ensemble(param_values, t_τL, x)
#     # If simulate_ensemble returns (outputs, valid_ids):
#     # ensemble_data, valid_ids = simulate_ensemble(param_values, t_τL, x)

#     # Show shape or valid IDs
#     if isinstance(ensemble_data, tuple):
#         ensemble_data, valid_ids = ensemble_data
#         print("Valid IDs:", valid_ids)
#     print("ensemble_data.shape:", ensemble_data.shape)

#     # Separate out pLV, psa, Vlv data blocks
#     num_time_points = len(x)
#     pLV_block = ensemble_data[:num_time_points, :]
#     psa_block = ensemble_data[num_time_points:2 * num_time_points, :]
#     Vlv_block = ensemble_data[2 * num_time_points:, :]

#     # Evaluate final time (t = 35.0) to do a standard (full-model) Sobol
#     final_idx = num_time_points - 1
#     print(f"\n=== Sobol Indices at Final Time (t = {x[final_idx]:.3f}s) ===")

#     # Prepare the 3 final outputs
#     pLV_final = pLV_block[final_idx, :]
#     psa_final = psa_block[final_idx, :]
#     Vlv_final = Vlv_block[final_idx, :]

#     # Full-model Sobol analysis
#     si_pLV = sobol.analyze(problem, pLV_final, calc_second_order=True, print_to_console=False)
#     si_psa = sobol.analyze(problem, psa_final, calc_second_order=True, print_to_console=False)
#     si_Vlv = sobol.analyze(problem, Vlv_final, calc_second_order=True, print_to_console=False)

#     # Print some results
#     print("pLV: S1 =", si_pLV['S1'], "\n     ST =", si_pLV['ST'])
#     print("\npsa: S1 =", si_psa['S1'], "\n     ST =", si_psa['ST'])
#     print("\nVlv: S1 =", si_Vlv['S1'], "\n     ST =", si_Vlv['ST'])

#     # Make heatmaps for S1 and ST
#     param_names = problem['names']
#     output_names = ['pLV', 'psa', 'Vlv']

#     S1_mat = np.zeros((len(param_names), 3))
#     ST_mat = np.zeros((len(param_names), 3))

#     S1_mat[:, 0] = si_pLV['S1']
#     S1_mat[:, 1] = si_psa['S1']
#     S1_mat[:, 2] = si_Vlv['S1']

#     ST_mat[:, 0] = si_pLV['ST']
#     ST_mat[:, 1] = si_psa['ST']
#     ST_mat[:, 2] = si_Vlv['ST']

#     print("\n=== Heatmap Plots (Full-Model) ===")
#     plot_sobol_heatmap(S1_mat, param_names, output_names,
#                        "Sobol - First Order (Full Model)", colormap="plasma")
#     plot_sobol_heatmap(ST_mat, param_names, output_names,
#                        "Sobol - Total Order (Full Model)", colormap="plasma")


#     # ------------------------------------------------------------
#     # C) Build and use PCE Surrogate for faster sensitivity
#     # ------------------------------------------------------------
#     print("\n=== Building PCE Surrogates with OpenTURNS ===")

#     # Suppose we want to build a surrogate for final-time outputs
#     # X_train = param_values (the 9D param sets)
#     # Y_train_list = final outputs [pLV_final, psa_final, Vlv_final]

#     # (If you have invalid sims, use param_values[valid_ids] + final outputs from valid IDs.)
#     X_train = param_values
#     Y_train_list = [pLV_final, psa_final, Vlv_final]

#     # Build the PCE surrogates
#     from surrogate import build_pce_surrogates_openturns
#     meta_models, distribution = build_pce_surrogates_openturns(
#         X_train=X_train,
#         Y_train_list=Y_train_list,
#         bounds_list=problem['bounds'],  # from SALib problem
#         polynomial_degree=3
#     )
#     print("Built PCE surrogate for each output: pLV, psa, Vlv")

#     # Let's do a quick surrogate-based Sobol analysis for each output
#     # We'll re-use the same problem definition from SALib
#     from surrogate import sobol_analysis_with_surrogate

#     pLV_metamodel, psa_metamodel, Vlv_metamodel = meta_models  # three outputs
#     print("\n=== Surrogate-based Sobol for pLV ===")
#     si_pLV_sur = sobol_analysis_with_surrogate(pLV_metamodel, problem_dict=problem, N_sobol=1000)
#     print("pLV Surrogate: S1 =", si_pLV_sur['S1'], "\n               ST =", si_pLV_sur['ST'])

#     print("\n=== Surrogate-based Sobol for psa ===")
#     si_psa_sur = sobol_analysis_with_surrogate(psa_metamodel, problem_dict=problem, N_sobol=1000)
#     print("psa Surrogate: S1 =", si_psa_sur['S1'], "\n               ST =", si_psa_sur['ST'])

#     print("\n=== Surrogate-based Sobol for Vlv ===")
#     si_Vlv_sur = sobol_analysis_with_surrogate(Vlv_metamodel, problem_dict=problem, N_sobol=1000)
#     print("Vlv Surrogate: S1 =", si_Vlv_sur['S1'], "\n               ST =", si_Vlv_sur['ST'])

#     print("\nDone with surrogate-based analysis.")


# if __name__ == "__main__":
#     main()

import numpy as np
import matplotlib.pyplot as plt

# We still import SALib to generate parameter samples
from SALib.sample import saltelli

# Import the model and HRV function
from cardio_model import CardiovascularModel, HRV

# Import the parallel ensemble simulation
from simulation import simulate_ensemble

# Import the PCE-building functions
from surrogate import (
    build_pce_surrogates_openturns,
    evaluate_pce_model_openturns,
    sobol_analysis_with_surrogate
)

def baseline_run_and_plot():
    """
    Run the baseline (single) simulation of the cardiovascular model
    and plot the results (pressures + left ventricular volume).
    """
    p = [0.3, 0.45, 0.06, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]
    simulation_end_time = 35
    t_τL = HRV(simulation_end_time)

    u0 = np.array([8.0, 8.0, 8.0, 265.0, 0.0, 0.0, 0.0])
    udot0 = np.zeros(7)

    from assimulo.problem import Implicit_Problem
    from assimulo.solvers.sundials import IDA

    model = CardiovascularModel(u0, udot0, p, t_τL)
    problem = Implicit_Problem(model.res, u0, udot0, 0.0)
    problem.root = model.root
    problem.handle_event = model.handle_event
    problem.nroots = 1

    solver = IDA(problem)
    solver.report_continuously = True
    solver.atol = 1e-6
    solver.rtol = 1e-6
    solver.maxord = 5
    solver.maxh = 0.05
    solver.maxsteps = 20000

    tfinal = t_τL[-1] + 0.1
    plot_saveat = np.arange(0, tfinal + 0.002, 0.002)

    # Run the simulation
    t, y, yd = solver.simulate(tfinal, ncp_list=plot_saveat)

    # Extract & plot
    pLV = y[:, 0]
    psa = y[:, 1]
    psv = y[:, 2]
    Vlv = y[:, 3]

    # Plot pressures
    plt.figure(figsize=(12, 6))
    plt.plot(t, pLV, label='P_LV')
    plt.plot(t, psa, label='P_SA')
    plt.plot(t, psv, label='P_SV')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure')
    plt.title('Pressures Over Time (33 - 35 s)')
    plt.legend()
    plt.grid(True)
    plt.xlim(33, 35)
    plt.show()

    # Plot LV volume
    plt.figure(figsize=(12, 6))
    plt.plot(t, Vlv, label='V_LV')
    plt.xlabel('Time (s)')
    plt.ylabel('Volume')
    plt.title('Left Ventricular Volume (33 - 35 s)')
    plt.legend()
    plt.grid(True)
    plt.xlim(33, 35)
    plt.show()


def main():
    """
    1) Runs a single baseline simulation + plotting.
    2) Generates parameter samples & runs an ensemble simulation (only for surrogate training).
    3) Builds PCE Surrogates from the ensemble outputs.
    4) Optionally performs surrogate-based Sobol analysis.
    """

    # A) Baseline simulation & plots
    print("\n=== Running Baseline Simulation ===")
    baseline_run_and_plot()

    # B) Generate Saltelli param sets & run ensemble (NO full-model Sobol here)
    print("\n=== Generating Parameter Sets & Running Ensemble ===")

    problem = {
        'num_vars': 9,
        'names': ['τ_es', 'Δτ', 'Rmv', 'Zao', 'Rs', 'Csa', 'Csv', 'E_max', 'E_min'],
        'bounds': [
            [0.21, 0.34],
            [0.15, 0.205],
            [0.042, 0.078],
            [0.0231, 0.0429],
            [0.777, 1.443],
            [0.791, 1.469],
            [7.7, 14.3],
            [1.05, 1.95],
            [0.021, 0.039],
        ]
    }

    N = 32
    param_values = saltelli.sample(problem, N, calc_second_order=True)
    print("param_values.shape:", param_values.shape)

    # We only care about final-time data from t=33..35s
    x = np.arange(33, 35, 0.002)
    t_τL = HRV(35)

    # Run the ensemble to get time-series for pLV, psa, Vlv
    ensemble_data = simulate_ensemble(param_values, t_τL, x)
    if isinstance(ensemble_data, tuple):
        ensemble_data, valid_ids = ensemble_data
        print("[INFO] Valid IDs:", valid_ids)
    print("ensemble_data.shape:", ensemble_data.shape)

    num_time_points = len(x)
    pLV_block = ensemble_data[:num_time_points, :]
    psa_block = ensemble_data[num_time_points:2 * num_time_points, :]
    Vlv_block = ensemble_data[2 * num_time_points:, :]

    final_idx = num_time_points - 1
    pLV_final = pLV_block[final_idx, :]
    psa_final = psa_block[final_idx, :]
    Vlv_final = Vlv_block[final_idx, :]

    # C) Build the PCE Surrogates
    print("\n=== Building PCE Surrogates with OpenTURNS ===")
    X_train = param_values
    Y_train_list = [pLV_final, psa_final, Vlv_final]

    meta_models, distribution = build_pce_surrogates_openturns(
        X_train=X_train,
        Y_train_list=Y_train_list,
        bounds_list=problem['bounds'],
        polynomial_degree=3
    )
    print("Built PCE surrogate for each output: pLV, psa, Vlv")

    # D) Surrogate-based Sobol analysis for each output
    from surrogate import sobol_analysis_with_surrogate
    pLV_metamodel, psa_metamodel, Vlv_metamodel = meta_models

    print("\n=== Surrogate-based Sobol for pLV ===")
    si_pLV_sur = sobol_analysis_with_surrogate(pLV_metamodel, problem_dict=problem, N_sobol=1000)
    print("pLV Surrogate: S1 =", si_pLV_sur['S1'], "\n               ST =", si_pLV_sur['ST'])

    print("\n=== Surrogate-based Sobol for psa ===")
    si_psa_sur = sobol_analysis_with_surrogate(psa_metamodel, problem_dict=problem, N_sobol=1000)
    print("psa Surrogate: S1 =", si_psa_sur['S1'], "\n               ST =", si_psa_sur['ST'])

    print("\n=== Surrogate-based Sobol for Vlv ===")
    si_Vlv_sur = sobol_analysis_with_surrogate(Vlv_metamodel, problem_dict=problem, N_sobol=1000)
    print("Vlv Surrogate: S1 =", si_Vlv_sur['S1'], "\n               ST =", si_Vlv_sur['ST'])

    print("\nDone. (No full-model Sobol analysis was performed.)")


if __name__ == "__main__":
    main()

