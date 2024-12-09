import numpy as np
import matplotlib.pyplot as plt
from assimulo.problem import Implicit_Problem
from assimulo.solvers.sundials import IDA
from SALib.sample import saltelli
from SALib.analyze import sobol
from tqdm import tqdm
import h5py
import seaborn as sns
from multiprocessing import Pool, cpu_count


# Valve function remains the same
def Valve(R, deltaP):
    q = 0.0
    R = max(R, 1e-6)    
    if -deltaP < 0.0:
        q = deltaP / R
    else:
        q = 0.0
    return q

# Define the model class
class CardiovascularModel:
    def __init__(self, u0, udot0, p, t_τL):
        self.p = p
        self.t_τL = t_τL
        self.τ = t_τL[0]
        self.tr = 0.0
        self.n = 0  # Counter for events
        self.u0 = u0
        self.udot0 = udot0
        self.Eshift = 0.0
        self.cycle_phase = "systole"  # Start with systole

    # Residual function for IDA
    def res(self, t, y, ydot):
        pLV, psa, psv, Vlv, Qav, Qmv, Qs = y
        τ_es, τ_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min = self.p
        E_t = self.ShiElastance(t, E_min, E_max, self.τ, τ_es, τ_ep, self.Eshift)
        DE_t = self.DShiElastance(t, E_min, E_max, self.τ, τ_es, τ_ep, self.Eshift)

        print(f"self.p shape: {self.p.shape}, self.p content: {self.p}")
        res = np.zeros_like(y)
        res[0] = ydot[0] - ((Qmv - Qav) * E_t + pLV / E_t * DE_t)
        res[1] = ydot[1] - (Qav - Qs) / Csa
        res[2] = ydot[2] - (Qs - Qmv) / Csv
        res[3] = ydot[3] - (Qmv - Qav)
        res[4] = Qav - Valve(Zao, pLV - psa)
        res[5] = Qmv - Valve(Rmv, psv - pLV)
        res[6] = ydot[6] - (ydot[1] - ydot[2]) / Rs
        return res

    # ShiElastance function as a method
    def ShiElastance(self, t, E_min, E_max, τ, τ_es, τ_ep, Eshift):
        t_i = (t - self.tr) % τ  # Time within the current cardiac cycle
        if t_i <= τ_es:
            E_p = (1 - np.cos(t_i / τ_es * np.pi)) / 2
        elif t_i <= τ_ep:
            E_p = (1 + np.cos((t_i - τ_es) / (τ_ep - τ_es) * np.pi)) / 2
        else:
            E_p = 0.0
        E = E_min + (E_max - E_min) * E_p
        return E

    # Derivative of ShiElastance as a method
    def DShiElastance(self, t, E_min, E_max, τ, τ_es, τ_ep, Eshift):
        t_i = (t - self.tr) % τ
        if t_i <= τ_es:
            dE_p = (np.pi / τ_es) * np.sin(t_i / τ_es * np.pi) / 2
        elif t_i <= τ_ep:
            dE_p = - (np.pi / (τ_ep - τ_es)) * np.sin((t_i - τ_es) / (τ_ep - τ_es) * np.pi) / 2
        else:
            dE_p = 0.0
        dE = (E_max - E_min) * dE_p
        return dE

    # Event function to update τ and tr
    def handle_event(self, solver, event_info):
        self.n += 1

        if self.cycle_phase == "systole":
            self.cycle_phase = "diastole"
            self.tr = solver.t
        elif self.cycle_phase == "diastole":
            self.cycle_phase = "systole"
            self.tr = solver.t
            if self.n + 1 < len(self.t_τL):
                self.τ = self.t_τL[self.n + 1] - self.t_τL[self.n]
            else:
                solver.terminate = True

    def root(self, t, y, ydot):
        t_i = (t - self.tr) % self.τ
        if self.cycle_phase == "systole":
            return np.array([t_i - self.p[0]])  # τ_es
        elif self.cycle_phase == "diastole":
            return np.array([t_i - self.p[1]])  # τ_ep
        else:
            return np.array([0.0])

# Define the parameters
p = np.array([0.3, 0.45, 0.06, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03])

# Generate the heart rate variability times
def HRV(end_time):
    t_τL = []
    t_current = 0.0
    while t_current < end_time + 1.0:  # Add buffer time
        τ = np.random.uniform(0.8, 1.1)
        t_current += τ
        t_τL.append(t_current)
    return np.array(t_τL)

# Define the time points for initial simulation
simulation_end_time = 35  # We need to simulate up to at least 35 seconds

# Generate t_τL to cover the simulation time beyond 35 seconds
t_τL = HRV(simulation_end_time)

# Print total simulation time for verification
print(f"Total simulation time: {t_τL[-1] + 0.1} seconds")

# Initial conditions
u0 = np.array([8.0, 8.0, 8.0, 265.0, 0.0, 0.0, 0.0])
udot0 = np.zeros(7)  # Initial derivatives

# Initialize the model
model = CardiovascularModel(u0, udot0, p, t_τL)

# Create the problem instance
problem = Implicit_Problem(model.res, u0, udot0, 0.0)
problem.name = 'Cardiovascular Model with Events'

# Set the root function and event handler
problem.root = model.root
problem.handle_event = model.handle_event

# Set the number of roots (events)
problem.nroots = 1

# Set up the solver
solver = IDA(problem)
solver.report_continuously = True  # Ensure continuous output
solver.atol = 1e-6
solver.rtol = 1e-6
solver.maxord = 5
solver.maxh = 0.05  # Increase maximum step size
solver.maxsteps = 10000  # Allow more solver steps

# Simulate
tfinal = t_τL[-1] + 0.1  # Ensure simulation covers all events

# Define the times at which to save the outputs for plotting
plot_saveat = np.arange(0, tfinal + 0.002, 0.002)

# Simulate the model with specified output times
t, y, yd = solver.simulate(tfinal, ncp_list=plot_saveat)

# Extract variables
pLV = y[:, 0]
psa = y[:, 1]
psv = y[:, 2]
Vlv = y[:, 3]
Qav = y[:, 4]
Qmv = y[:, 5]
Qs = y[:, 6]

# Plot pressures between 33 and 35 seconds
plt.figure(figsize=(12, 6))
plt.plot(t, pLV, label='P_LV')
plt.plot(t, psa, label='P_SA')
plt.plot(t, psv, label='P_SV')
plt.xlabel('Time (s)')
plt.ylabel('Pressure')
plt.title('Pressures Over Time')
plt.legend()
plt.grid(True)
plt.xlim(33, 35)  # Limit x-axis to between 33 and 35 seconds
plt.show()

# Plot left ventricular volume between 33 and 35 seconds
plt.figure(figsize=(12, 6))
plt.plot(t, Vlv, label='V_LV')
plt.xlabel('Time (s)')
plt.ylabel('Volume')
plt.title('Left Ventricular Volume Over Time')
plt.legend()
plt.grid(True)
plt.xlim(33, 35)  # Limit x-axis to between 33 and 35 seconds
plt.show()


"-------------------------------------------------------------------------------------------senstivity-analysis-----------------------------------------------------------------------------------------------------"

# Simulation functions with multiprocessing
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
    assert τ_ep > τ_es, f"Invalid parameters: τ_ep ({τ_ep}) should be greater than τ_es ({τ_es})"

    model = CardiovascularModel(u0, np.zeros(7), params, t_τL)
    problem = Implicit_Problem(model.res, u0, np.zeros(7), x[0])  # Set initial time to x[0]
    problem.root = model.root
    problem.handle_event = model.handle_event
    problem.nroots = 1

    solver = IDA(problem)
    solver.report_continuously = False
    solver.atol = 1e-6
    solver.rtol = 1e-6
    solver.maxord = 5
    solver.maxh = 0.05
    solver.maxsteps = 10000

    t, y, _ = solver.simulate(tfinal=x[-1], ncp_list=x)
    print(f"Expected time points: {len(x)}, Solver time points: {y.shape[0]}")

    if y.shape[0] != len(x):
        raise ValueError(f"Solver generated different number of time points: {y.shape[0]} vs {len(x)}")

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

    valid_results = [(i, res) for i, res in enumerate(results) if res is not None]

    num_valid = len(valid_results)
    outputs = np.zeros((3 * num_time_points, num_valid))

    # Combine results into outputs array
    for idx, (i, (pLV, psa, Vlv)) in enumerate(valid_results):
        outputs[:num_time_points, idx] = pLV
        outputs[num_time_points:2 * num_time_points, idx] = psa
        outputs[2 * num_time_points:, idx] = Vlv

    return outputs

def plot_sobol_heatmap(S_matrix, parameter_names, output_names, title, colormap="plasma"):
    """
    Function to plot a Sobol sensitivity heatmap.
    
    Parameters:
        S_matrix (2D array): Matrix of Sobol indices.
        parameter_names (list): Names of the parameters (row labels).
        output_names (list): Names of the outputs (column labels).
        title (str): Title for the heatmap.
        colormap (str): Matplotlib colormap for the heatmap.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the heatmap
    heatmap = ax.imshow(S_matrix, cmap=colormap, aspect="auto", vmin=0, vmax=1)

    # Set the ticks
    ax.set_xticks(np.arange(len(output_names)))
    ax.set_xticklabels(output_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(parameter_names)))
    ax.set_yticklabels(parameter_names, fontsize=10)

    # Add labels and title
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Outputs", fontsize=10)
    ax.set_ylabel("Parameters", fontsize=10)

    # Add colorbar
    cbar = fig.colorbar(heatmap, ax=ax, orientation="vertical", shrink=0.8, pad=0.02)
    cbar.set_label("Sobol Index", fontsize=10)

    # Annotate the heatmap with the values
    for i in range(len(parameter_names)):
        for j in range(len(output_names)):
            value = S_matrix[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.5 else "black", fontsize=8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parameters
    p = np.array([0.3, 0.45, 0.06, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03])
    problem = {
        'num_vars': 9,
        'names': ['τ_es', 'Δτ', 'Rmv', 'Zao', 'Rs', 'Csa', 'Csv', 'E_max', 'E_min'],
        'bounds': [
            [0.21, 0.34],        # τ_es
            [0.15, 0.245],       # τ_ep 
            [0.042, 0.078],      # Rmv
            [0.0231, 0.0429],    # Zao
            [0.777, 1.443],      # Rs
            [0.791, 1.469],      # Csa
            [7.7, 14.3],         # Csv
            [1.05, 1.95],        # E_max
            [0.021, 0.039],      # E_min
        ]
    }
    N = 4  # Increased Sobol sample size
    param_values = saltelli.sample(problem, N, calc_second_order=True)

    num_parameters = len(problem["names"])
    num_outputs = 3  # Number of outputs (e.g., pLV, psa, Vlv)
    # Generate t_τL
    simulation_end_time = 35
    t_τL = HRV(simulation_end_time)

    # Simulation time points
    x = np.arange(10, 13, 0.002)

    # Simulate ensemble
    ensemble_outputs = simulate_ensemble(param_values, t_τL, x)

    print("param_values.shape:", param_values.shape)
    print("ensemble_outputs.shape:", ensemble_outputs.shape)

    # Sobol analysis
    # Choose specific indices corresponding to variables and time points
    num_time_points = len(x)
    indices_to_analyze = [num_time_points - 1]  # Analyze the last time point
    variables_to_analyze = ['pLV', 'psa', 'Vlv']
    variable_indices = [0, 1, 3]  # Indices in y corresponding to the variables

    for idx in indices_to_analyze:
        Y = ensemble_outputs[idx, :]
        sobol_indices = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=False)
        S1 = sobol_indices["S1"]
        ST = sobol_indices["ST"]
        print(f"Time point: {x[idx]:.3f}")
        print("S1 indices:", S1)
        print("ST indices:", ST)

       # Initialize matrices for S1 and ST indices
    S1_matrix = np.zeros((num_parameters, num_outputs))
    ST_matrix = np.zeros((num_parameters, num_outputs))

    # Analyze Sobol indices for each output
    for output_idx, variable in enumerate(["pLV", "psa", "Vlv"]):
        Y = ensemble_outputs[output_idx * len(x):(output_idx + 1) * len(x), :]
        sobol_indices = sobol.analyze(problem, Y[-1, :], calc_second_order=True, print_to_console=False)
        S1_matrix[:, output_idx] = sobol_indices["S1"]
        ST_matrix[:, output_idx] = sobol_indices["ST"]

    parameter_names = problem["names"]
    output_names = ["pLV", "psa", "Vlv"]


    # Assuming you want to analyze pLV at all time points
    S1_list, ST_list = [], []
    for i in range(num_time_points):
        Y = ensemble_outputs[i, :]
        sobol_indices = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=False)
        S1_list.append(sobol_indices["S1"])
        ST_list.append(sobol_indices["ST"])
    # Convert lists to arrays for plotting
    S1_array = np.array(S1_list)
    ST_array = np.array(ST_list)

    # Sobol analysis at selected time points
    selected_time_indices = [0, num_time_points // 2, num_time_points - 1]  # Start, middle, end
    S1_results = {}
    ST_results = {}

    for idx in selected_time_indices:
        Y = ensemble_outputs[idx, :]  # pLV at selected time point
        sobol_indices = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=False)
        S1_results[x[idx]] = sobol_indices["S1"]
        ST_results[x[idx]] = sobol_indices["ST"]

    # Plotting Sobol indices over selected time points
    for param_idx, param_name in enumerate(problem["names"]):
        S1_values = [S1_results[time_point][param_idx] for time_point in x[selected_time_indices]]
        ST_values = [ST_results[time_point][param_idx] for time_point in x[selected_time_indices]]
        plt.plot([x[idx] for idx in selected_time_indices], S1_values, label=f"S1 - {param_name}")
        plt.plot([x[idx] for idx in selected_time_indices], ST_values, label=f"ST - {param_name}")

    plt.xlabel('Time (s)')
    plt.ylabel('Sobol Indices')
    plt.legend()
    plt.show()

    plot_sobol_heatmap(S1_matrix, parameter_names, output_names, "Sobol - First Order", colormap="plasma")

    # Total-order indices heatmap
    plot_sobol_heatmap(ST_matrix, parameter_names, output_names, "Sobol - Total Order", colormap="plasma")




# lb = [0.21, 0.36, 0.042, 0.0231, 0.777, 0.791, 7.7, 1.05, 0.021]
# ub = [0.34, 0.585, 0.078, 0.0429, 1.443, 1.469, 14.3, 1.95, 0.039]