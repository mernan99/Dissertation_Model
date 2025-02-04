import numpy as np
import matplotlib.pyplot as plt
import openturns as ot

from assimulo.problem import Implicit_Problem
from assimulo.solvers.sundials import IDA

from SALib.sample import saltelli
from SALib.analyze import sobol
from multiprocessing import Pool, cpu_count

###############################################################################
#                        MODEL CODE (BASELINE SIMULATION)                     #
###############################################################################

def Valve(R, deltaP):
    """
    Simple valve model:
    Returns flow = deltaP/R if the pressure drop is negative, otherwise zero.
    """
    R = max(R, 1e-6)
    return deltaP / R if -deltaP < 0 else 0.0

class CardiovascularModel:
    def __init__(self, u0, udot0, p, t_τL):
        """
        Initialize the model.
          - u0: initial state
          - udot0: initial derivative state
          - p: parameter array in the order
               [τ_es, τ_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min]
          - t_τL: array of cycle times
        """
        self.p = np.array(p)
        self.t_τL = t_τL
        self.τ = t_τL[0]
        self.tr = 0.0  # Reference time for each cycle
        self.n = 0     # Cycle counter
        self.u0 = u0
        self.udot0 = udot0
        self.Eshift = 0.0
        self.cycle_phase = "systole"

    def res(self, t, y, ydot):
        pLV, psa, psv, Vlv, Qav, Qmv, Qs = y
        τ_es, τ_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min = self.p

        E_t = self.ShiElastance(t, E_min, E_max, self.τ, τ_es, τ_ep, self.Eshift)
        DE_t = self.DShiElastance(t, E_min, E_max, self.τ, τ_es, τ_ep, self.Eshift)

        res = np.zeros_like(y)
        res[0] = ydot[0] - ((Qmv - Qav) * E_t + pLV / E_t * DE_t)
        res[1] = ydot[1] - (Qav - Qs) / Csa
        res[2] = ydot[2] - (Qs - Qmv) / Csv
        res[3] = ydot[3] - (Qmv - Qav)
        res[4] = Qav - Valve(Zao, pLV - psa)
        res[5] = Qmv - Valve(Rmv, psv - pLV)
        res[6] = ydot[6] - (ydot[1] - ydot[2]) / Rs
        return res

    def ShiElastance(self, t, E_min, E_max, τ, τ_es, τ_ep, Eshift):
        t_i = (t - self.tr) % τ
        if t_i <= τ_es:
            E_p = (1 - np.cos(t_i / τ_es * np.pi)) / 2
        elif t_i <= τ_ep:
            E_p = (1 + np.cos((t_i - τ_es) / (τ_ep - τ_es) * np.pi)) / 2
        else:
            E_p = 0.0
        return E_min + (E_max - E_min) * E_p

    def DShiElastance(self, t, E_min, E_max, τ, τ_es, τ_ep, Eshift):
        t_i = (t - self.tr) % τ
        if t_i <= τ_es:
            dE_p = (np.pi / τ_es) * np.sin(t_i / τ_es * np.pi) / 2
        elif t_i <= τ_ep:
            dE_p = - (np.pi / (τ_ep - τ_es)) * np.sin((t_i - τ_es) / (τ_ep - τ_es) * np.pi) / 2
        else:
            dE_p = 0.0
        return (E_max - E_min) * dE_p

    def handle_event(self, solver, event_info):
        self.n += 1
        self.tr = round(solver.t, 6)
        if self.n + 1 < len(self.t_τL):
            self.τ = max(1e-6, self.t_τL[self.n + 1] - self.t_τL[self.n])
        else:
            solver.terminate = True

    def root(self, t, y, ydot):
        t_i = (t - self.tr) % self.τ
        return np.array([t_i - self.p[0]])  # event when t_i = τ_es

def HRV(end_time):
    """
    Generate heart rate variability (HRV) times until end_time plus a buffer.
    """
    t_τL = []
    t_current = 0.0
    while t_current < end_time + 1.0:
        τ = np.random.uniform(0.8, 1.1)
        t_current += τ
        t_τL.append(t_current)
    return np.array(t_τL)

###############################################################################
#              1) BASELINE RUN + PLOTTING (MODEL GRAPH)                       #
###############################################################################

def baseline_run_and_plot():
    # Define baseline parameters
    p = [0.3, 0.45, 0.06, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]
    simulation_end_time = 35
    t_τL = HRV(simulation_end_time)

    u0 = np.array([8.0, 8.0, 8.0, 265.0, 0.0, 0.0, 0.0])
    udot0 = np.zeros(7)

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

    # Run solver
    t, y, yd = solver.simulate(tfinal, ncp_list=plot_saveat)

    pLV = y[:, 0]
    psa = y[:, 1]
    psv = y[:, 2]
    Vlv = y[:, 3]
    Qav = y[:, 4]
    Qmv = y[:, 5]
    Qs  = y[:, 6]

    # Plot pressures between 33 and 35s
    plt.figure(figsize=(12, 6))
    plt.plot(t, pLV, label='P_LV')
    plt.plot(t, psa, label='P_SA')
    plt.plot(t, psv, label='P_SV')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure')
    plt.title('Pressures Over Time')
    plt.legend()
    plt.grid(True)
    plt.xlim(33, 35)
    plt.show()

    # Plot ventr. volume between 33 and 35s
    plt.figure(figsize=(12, 6))
    plt.plot(t, Vlv, label='V_LV')
    plt.xlabel('Time (s)')
    plt.ylabel('Volume')
    plt.title('Left Ventricular Volume Over Time')
    plt.legend()
    plt.grid(True)
    plt.xlim(33, 35)
    plt.show()


###############################################################################
#              2) ENSEMBLE SIMULATION + SOBOL (TABLE + HEATMAP)               #
###############################################################################

def simulate_single_trajectory(args):
    """
    Run the model for one param set. Returns (pLV_timeSeries, psa_timeSeries, Vlv_timeSeries).
    """
    index, param_values, t_τL, x = args
    u0 = np.array([8.0, 8.0, 8.0, 265.0, 0.0, 0.0, 0.0])
    params = param_values[index, :]

    # Extract τ_es, Δτ, build param array
    τ_es = params[0]
    Δτ   = params[1]
    τ_ep = τ_es + Δτ
    model_params = np.insert(params, 1, τ_ep)
    model_params = np.delete(model_params, 2)  # remove Δτ

    if τ_ep <= τ_es:
        print(f"Invalid param set {index}: τ_ep ({τ_ep}) <= τ_es ({τ_es})")
        return None

    model = CardiovascularModel(u0, np.zeros(7), model_params, t_τL)
    problem = Implicit_Problem(model.res, u0, np.zeros(7), 0.0)
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
        t_res, y_res, _ = solver.simulate(tfinal=x[-1], ncp_list=x)
    except Exception as e:
        print(f"Simulation failed for index {index} with error: {e}")
        return None

    t_res   = t_res[1:]
    y_res   = y_res[1:, :]
    if len(t_res) != len(x):
        print(f"Time mismatch for index {index}.")
        return None

    # Return the time-series for pLV, psa, Vlv
    pLV = y_res[:, 0]
    psa = y_res[:, 1]
    Vlv = y_res[:, 3]
    return pLV, psa, Vlv

def simulate_ensemble(param_values, t_τL, x):
    """
    Run the model for all param_values in parallel.
    Returns a (3*len(x), num_valid) matrix with time-series data: 
       row 0..(len(x)-1): pLV(t)
       row len(x)..(2*len(x)-1): psa(t)
       row 2*len(x)..(3*len(x)-1): Vlv(t)
    """
    num_time_points = len(x)
    num_trajectories = param_values.shape[0]
    args_list = [(i, param_values, t_τL, x) for i in range(num_trajectories)]

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(simulate_single_trajectory, args_list)

    valid_res = [(i,res) for i,res in enumerate(results) if res is not None]
    if not valid_res:
        raise RuntimeError("No valid results for ensemble simulation.")

    outputs = np.zeros((3*num_time_points, len(valid_res)))
    for idx, (i, triple) in enumerate(valid_res):
        pLV, psa, Vlv = triple
        outputs[:num_time_points, idx]                  = pLV
        outputs[num_time_points:2*num_time_points, idx] = psa
        outputs[2*num_time_points:, idx]               = Vlv

    return outputs, [i for i,_ in valid_res]

def plot_sobol_heatmap(S_matrix, parameter_names, output_names, title, colormap="plasma"):
    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.imshow(S_matrix, cmap=colormap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(output_names)))
    ax.set_xticklabels(output_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(parameter_names)))
    ax.set_yticklabels(parameter_names, fontsize=10)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Outputs", fontsize=10)
    ax.set_ylabel("Parameters", fontsize=10)

    cbar = fig.colorbar(heatmap, ax=ax, orientation="vertical", shrink=0.8, pad=0.02)
    cbar.set_label("Sobol Index", fontsize=10)

    # Annotate
    for i in range(len(parameter_names)):
        for j in range(len(output_names)):
            val = S_matrix[i,j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val>0.5 else "black", fontsize=8)

    plt.tight_layout()
    plt.show()

def main():

    ############################################################################
    # A) BASELINE SIMULATION + PLOT
    ############################################################################
    print("\n=== Running Baseline Simulation ===")
    baseline_run_and_plot()

    ############################################################################
    # B) ENSEMBLE SIMULATION & SOBOL ANALYSIS
    ############################################################################
    print("\n=== Running Ensemble Simulations for Sensitivity ===")
    problem = {
        'num_vars': 9,
        'names': ['τ_es','Δτ','Rmv','Zao','Rs','Csa','Csv','E_max','E_min'],
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

    # Generate Saltelli param sets
    N = 32
    param_values = saltelli.sample(problem, N, calc_second_order=True)
    print("param_values.shape:", param_values.shape)

    # We simulate from t=33..35 seconds
    x = np.arange(33, 35, 0.002)
    # Make HRV times for up to 35 s
    t_τL = HRV(35)

    # Perform ensemble runs
    ensemble_data, valid_ids = simulate_ensemble(param_values, t_τL, x)
    print("ensemble_data.shape:", ensemble_data.shape)

    # each column is a valid simulation
    # rows 0..(len(x)-1) => pLV(t),
    #      len(x)..(2*len(x)-1) => psa(t),
    #      2*len(x)..(3*len(x)-1) => Vlv(t)

    # => We do final-time analysis for pLV, psa, Vlv
    num_time_points = len(x)
    pLV_block = ensemble_data[:num_time_points, :]
    psa_block = ensemble_data[num_time_points:2*num_time_points, :]
    Vlv_block = ensemble_data[2*num_time_points:, :]

    # 1) Print Sobol table at final time point
    final_idx = num_time_points - 1
    print("\n=== Sobol Indices at Final Time (t=%.3f s) ===" % x[final_idx])

    # pLV final
    pLV_final = pLV_block[final_idx, :]
    si_pLV = sobol.analyze(problem, pLV_final, calc_second_order=True, print_to_console=False)
    print("pLV: S1 =", si_pLV['S1'], "\n     ST =", si_pLV['ST'])

    # psa final
    psa_final = psa_block[final_idx, :]
    si_psa = sobol.analyze(problem, psa_final, calc_second_order=True, print_to_console=False)
    print("\npsa: S1 =", si_psa['S1'], "\n     ST =", si_psa['ST'])

    # Vlv final
    Vlv_final = Vlv_block[final_idx, :]
    si_Vlv = sobol.analyze(problem, Vlv_final, calc_second_order=True, print_to_console=False)
    print("\nVlv: S1 =", si_Vlv['S1'], "\n     ST =", si_Vlv['ST'])

    # 2) Heatmap of S1, ST across pLV, psa, Vlv
    param_names = problem['names']
    output_names = ['pLV','psa','Vlv']
    S1_mat = np.zeros((len(param_names), 3))
    ST_mat = np.zeros((len(param_names), 3))

    S1_mat[:, 0] = si_pLV['S1']
    ST_mat[:, 0] = si_pLV['ST']
    S1_mat[:, 1] = si_psa['S1']
    ST_mat[:, 1] = si_psa['ST']
    S1_mat[:, 2] = si_Vlv['S1']
    ST_mat[:, 2] = si_Vlv['ST']

    print("\n=== Heatmap Plots ===")
    plot_sobol_heatmap(S1_mat, param_names, output_names, "Sobol - First Order", colormap="plasma")
    plot_sobol_heatmap(ST_mat, param_names, output_names, "Sobol - Total Order", colormap="plasma")

    print("Done.")

if __name__ == "__main__":
    main()
