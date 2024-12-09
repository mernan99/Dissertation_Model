import numpy as np
import matplotlib.pyplot as plt
from assimulo.problem import Implicit_Problem
from assimulo.solvers.sundials import IDA
from SALib.sample import saltelli
from SALib.analyze import sobol
from scipy.stats.qmc import Sobol
from tqdm import tqdm
import multiprocessing
from SALib.sample import sobol_sequence

# Valve function remains the same
def Valve(R, deltaP):
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
        t_i = (t - self.tr) % τ  # Modulus to keep t within the current cycle period
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
        t_i = (t - self.tr) % τ  # Modulus to keep t within the current cycle period
        if t_i <= τ_es:
            dE_p = (np.pi / τ_es) * np.sin(t_i / τ_es * np.pi) / 2
        elif t_i <= τ_ep:
            dE_p = - (np.pi / (τ_ep - τ_es)) * np.sin((t_i - τ_es) / (τ_ep - τ_es) * np.pi) / 2
        else:
            dE_p = 0.0
        dE = (E_max - E_min) * dE_p
        return dE

    # # Event function to update τ and tr
    # def handle_event(self, solver, event_info):
    #     """
    #     Handles events detected by the root function.
    #     Updates the cardiac cycle phase, elastance parameters, and time reference.
    #     """
    #     self.n += 1

    #     if self.cycle_phase == "systole":
    #         # Switch to diastole
    #         self.cycle_phase = "diastole"
    #         # Update reference time for diastole
    #         self.tr = solver.t
    #         # Update parameters for relaxation (change elastance accordingly)
    #     elif self.cycle_phase == "diastole":
    #         # Switch to systole
    #         self.cycle_phase = "systole"
    #         if self.n + 1 < len(self.t_τL):
    #             # Update tr and τ for next systolic event
    #             self.τ = self.t_τL[self.n + 1] - self.t_τL[self.n]
    #             self.tr = self.t_τL[self.n]
    #         else:
    #             # No more events, end the simulation
    #             solver.terminate = True

    # def root(self, t, y, ydot):
    #     """
    #     Detects events that correspond to changes in the cardiac cycle.
    #     Specifically detects when to switch between systole and diastole.
    #     """
    #     if self.cycle_phase == "systole":
    #         # Event to detect the end of systole (e.g., when elastance reaches max)
    #         return t - (self.tr + self.τ)
    #     elif self.cycle_phase == "diastole":
    #         # Event to detect the end of diastole (e.g., when filling ends)
    #         return t - (self.tr + self.τ)
    #     return 0.0  # This should return values that cross zero during the transitions

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
def HRV(c):
    t_τL = np.zeros(c)
    t_τL[0] = np.random.uniform(0.8, 1.1)
    for i in range(c - 1):
        t_τL[i + 1] = t_τL[i] + np.random.uniform(0.8, 1.1)
    return t_τL

c = 16
t_τL = HRV(c)

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
solver.report_continuously = True
solver.atol = 1e-8
solver.rtol = 1e-8
solver.maxord = 5

# Simulate
tfinal = t_τL[-1] + 0.1  # Ensure simulation covers all events
t, y, yd = solver.simulate(tfinal)

# Extract variables
pLV = y[:, 0]
psa = y[:, 1]
psv = y[:, 2]
Vlv = y[:, 3]
Qav = y[:, 4]
Qmv = y[:, 5]
Qs = y[:, 6]

# Plot pressures
plt.figure(figsize=(12, 6))
plt.plot(t, pLV, label='P_LV')
plt.plot(t, psa, label='P_SA')
plt.plot(t, psv, label='P_SV')
plt.xlabel('Time')
plt.ylabel('Pressure')
plt.title('Pressures Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot left ventricular volume
plt.figure(figsize=(12, 6))
plt.plot(t, Vlv, label='V_LV')
plt.xlabel('Time')
plt.ylabel('Volume')
plt.title('Left Ventricular Volume Over Time')
plt.legend()
plt.grid(True)
plt.show()

"-----------------------------------------------Sensitivity analysis---------------------------------------------------"

def run_simulation_wrapper(args):
    params, u0, udot0, t_τL, saveat = args
    return run_simulation(params, u0, udot0, t_τL, saveat)


def run_simulation(params, u0, udot0, t_τL, saveat):
    τ_es, τ_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min = params

    # Enforce constraints on parameters to prevent invalid values
    if τ_ep <= τ_es:
        τ_ep = τ_es + 1e-6

    if E_min >= E_max:
        E_min = E_max - 1e-6

    # Set minimum positive values to avoid division by zero
    Rmv = max(Rmv, 1e-6)
    Zao = max(Zao, 1e-6)
    Rs = max(Rs, 1e-6)
    Csa = max(Csa, 1e-6)
    Csv = max(Csv, 1e-6)
    E_max = max(E_max, 1e-6)
    E_min = max(E_min, 0.0)

    p = np.array([τ_es, τ_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min])

    # Initialize the model
    model = CardiovascularModel(u0, udot0, p, t_τL)
    problem_instance = Implicit_Problem(model.res, u0, udot0, 0.0)
    problem_instance.root = model.root
    problem_instance.handle_event = model.handle_event
    problem_instance.nroots = 1

    solver = IDA(problem_instance)
    solver.atol = 1e-6
    solver.rtol = 1e-6
    solver.maxord = 5
    solver.maxh = 0.01

    tfinal = t_τL[-1] + 0.1

    try:
        # Simulate the model
        solver.simulate(tfinal)
        # Interpolate results at desired time points
        y = np.array([solver.interpolate(ti)[0] for ti in saveat])

        # Extract variables
        pLV = y[:, 0]
        psa = y[:, 1]
        Vlv = y[:, 3]

        # Return time-series outputs
        return np.column_stack((pLV, psa, Vlv))
    except Exception as e:
        print(f"Simulation failed with exception: {e}")
        # Return NaNs for failed simulations
        num_time_points = len(saveat)
        return np.full((num_time_points, 3), np.nan)



# def run_all_simulations(param_values, u0, udot0, t_τL):
#     args = [(params, u0, udot0, t_τL) for params in param_values]

#     with multiprocessing.Pool() as pool:
#         results = list(
#             tqdm(pool.imap_unordered(run_simulation_wrapper, args), total=len(param_values))
#         )
#     return np.array(results)

if __name__ == "__main__":
    # Define the problem for SALib

    problem = {
        "num_vars": 9,
        "names": ["τ_es", "τ_ep", "Rmv", "Zao", "Rs", "Csa", "Csv", "E_max", "E_min"],
        "bounds": [
            (0.21, 0.34),    # τ_es
            (0.36, 0.585),   # τ_ep
            (0.042, 0.078),  # Rmv
            (0.0231, 0.0429),# Zao
            (0.777, 1.443),  # Rs
            (0.791, 1.469),  # Csa
            (7.7, 14.3),     # Csv
            (1.05, 1.95),    # E_max
            (0.021, 0.039)   # E_min
        ]
    }

    # Set the sample size (N = number of base samples)
    N = 100  # Adjust as needed

    # Generate Sobol samples using SALib's saltelli.sample
    param_values = saltelli.sample(problem, N, calc_second_order=True)

    # Define the time points at which to save the outputs
    start_time = 10  # As in your Julia code
    end_time = 13
    time_step = 0.002
    saveat = np.arange(start_time, end_time + time_step, time_step)

    # Initial conditions
    u0 = np.array([8.0, 8.0, 8.0, 265.0, 0.0, 0.0, 0.0])
    udot0 = np.zeros(7)
    t_τL = HRV(16)

    # Run simulations
    print("Running simulations...")
    args_list = [(params, u0, udot0, t_τL, saveat) for params in param_values]

    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap_unordered(run_simulation_wrapper, args_list), total=len(args_list)))

    # Convert results to a NumPy array
    Y = np.array(results)

    # Identify valid simulations (exclude those with NaNs)
    valid_simulations = ~np.isnan(Y).any(axis=(1, 2))
    Y_valid = Y[valid_simulations]
    param_values_valid = param_values[valid_simulations]

    if len(Y_valid) < 10:
        print("Not enough valid simulations to perform sensitivity analysis.")
        exit()
    else:
        # Proceed with Sobol sensitivity analysis
        num_time_points = len(saveat)
        num_params = problem['num_vars']
        num_outputs = 3  # pLV, psa, Vlv

        # Initialize dictionaries to hold Sobol indices for each output
        S1 = {'pLV': np.zeros((num_time_points, num_params)),
            'psa': np.zeros((num_time_points, num_params)),
            'Vlv': np.zeros((num_time_points, num_params))}

        ST = {'pLV': np.zeros((num_time_points, num_params)),
            'psa': np.zeros((num_time_points, num_params)),
            'Vlv': np.zeros((num_time_points, num_params))}

        output_names = ['pLV', 'psa', 'Vlv']
        parameter_names = problem['names']

        for output_idx, output_name in enumerate(output_names):
            print(f"\nPerforming Sobol sensitivity analysis for {output_name}...")
            for t_idx in tqdm(range(num_time_points)):
                Y_t = Y[:, t_idx, output_idx]

                # Check if variance is zero
                if np.var(Y_t) == 0:
                    S1[output_name][t_idx, :] = np.zeros(num_params)
                    ST[output_name][t_idx, :] = np.zeros(num_params)
                    continue

                Si = sobol.analyze(problem, Y_t, calc_second_order=True, print_to_console=False)
                S1[output_name][t_idx, :] = Si['S1']
                ST[output_name][t_idx, :] = Si['ST']

        # Time-averaged Sobol indices for pLV
        var_pLV = np.var(Y[:, :, 0], axis=0)  # Variance at each time point
        total_var_pLV = np.sum(var_pLV)

        S1_mean_pLV = np.nansum(S1['pLV'] * var_pLV[:, None], axis=0) / total_var_pLV
        ST_mean_pLV = np.nansum(ST['pLV'] * var_pLV[:, None], axis=0) / total_var_pLV
        # Plotting Sobol indices over time for pLV
        plt.figure(figsize=(12, 6))
        for i in range(num_params):
            plt.plot(saveat, S1['pLV'][:, i], label=parameter_names[i])
        plt.title('First-order Sobol indices for pLV over time')
        plt.xlabel('Time (s)')
        plt.ylabel('Sobol index')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

        # Similarly for psa and Vlv
        # You can also plot total-order indices (ST) instead of first-order indices (S1)

        # Time-averaged Sobol indices for pLV
        var_pLV = np.var(Y[:, :, 0], axis=0)  # Variance at each time point
        total_var_pLV = np.sum(var_pLV)

        S1_mean_pLV = np.nansum(S1['pLV'] * var_pLV[:, None], axis=0) / total_var_pLV
        ST_mean_pLV = np.nansum(ST['pLV'] * var_pLV[:, None], axis=0) / total_var_pLV

        # Bar plot for time-averaged Sobol indices
        x = np.arange(num_params)
        plt.figure(figsize=(12, 6))
        plt.bar(x - 0.2, S1_mean_pLV, width=0.4, label='First-order')
        plt.bar(x + 0.2, ST_mean_pLV, width=0.4, label='Total-order')
        plt.xticks(x, parameter_names, rotation=45)
        plt.ylabel('Time-averaged Sobol index')
        plt.title('Time-averaged Sobol indices for pLV')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Repeat the time-averaged calculation and plotting for psa and Vlv

        # For psa
        var_psa = np.var(Y[:, :, 1], axis=0)
        total_var_psa = np.sum(var_psa)

        S1_mean_psa = np.nansum(S1['psa'] * var_psa[:, None], axis=0) / total_var_psa
        ST_mean_psa = np.nansum(ST['psa'] * var_psa[:, None], axis=0) / total_var_psa

        # Plot for psa
        x = np.arange(num_params)
        plt.figure(figsize=(12, 6))
        plt.bar(x - 0.2, S1_mean_psa, width=0.4, label='First-order')
        plt.bar(x + 0.2, ST_mean_psa, width=0.4, label='Total-order')
        plt.xticks(x, parameter_names, rotation=45)
        plt.ylabel('Time-averaged Sobol index')
        plt.title('Time-averaged Sobol indices for psa')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # For Vlv
        var_Vlv = np.var(Y[:, :, 2], axis=0)
        total_var_Vlv = np.sum(var_Vlv)

        S1_mean_Vlv = np.nansum(S1['Vlv'] * var_Vlv[:, None], axis=0) / total_var_Vlv
        ST_mean_Vlv = np.nansum(ST['Vlv'] * var_Vlv[:, None], axis=0) / total_var_Vlv

        # Plot for Vlv
        x = np.arange(num_params)
        plt.figure(figsize=(12, 6))
        plt.bar(x - 0.2, S1_mean_Vlv, width=0.4, label='First-order')
        plt.bar(x + 0.2, ST_mean_Vlv, width=0.4, label='Total-order')
        plt.xticks(x, parameter_names, rotation=45)
        plt.ylabel('Time-averaged Sobol index')
        plt.title('Time-averaged Sobol indices for Vlv')
        plt.legend()
        plt.tight_layout()
        plt.show()