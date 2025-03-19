# cardio_model.py
import numpy as np
import matplotlib.pyplot as plt
from assimulo.problem import Implicit_Problem
from assimulo.solvers.sundials import IDA


# Valve function remains the same
def Valve(R, deltaP):
    q = 0.0
    R = max(R, 1e-6)
    if -deltaP < 0.0:
        q = deltaP / R
    else:
        q = 0.0
    return q

# Generate the heart rate variability times
def HRV(end_time):
    t_τL = []
    t_current = 0.0
    while t_current < end_time + 1.0:  
        τ = np.random.uniform(0.8, 1.1)
        t_current += τ
        t_τL.append(t_current)
    return np.array(t_τL)

# Define the model class
class CardiovascularModel:
    def __init__(self, u0, udot0, p, t_τL):
        
        """_summary_initialization_function
        set the initial values of the model
        
        """
        
        self.p = np.array(p)
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
        if self.cycle_phase not in ["systole", "diastole"]:
            raise ValueError(f"Invalid cycle phase: {self.cycle_phase}")

        self.tr = round(solver.t, 6)  # Reset to solver time with consistent precision

        if self.cycle_phase == "systole":
            self.cycle_phase = "diastole"
        elif self.cycle_phase == "diastole":
            self.cycle_phase = "systole"

        if self.n + 1 < len(self.t_τL):
            self.τ = max(1e-6, self.t_τL[self.n + 1] - self.t_τL[self.n])  # Update τ
        else:
            if self.n + 1 >= len(self.t_τL):
                print(f"Terminating simulation at t={solver.t}")
                solver.terminate = True

    def root(self, t, y, ydot):
        """
        Root function to detect events.
        Here we trigger an event when the time within the cycle equals τ_es.
        """
        t_i = (t - self.tr) % self.τ
        return np.array([t_i - self.p[0]])

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
    t_values, y, yd = solver.simulate(tfinal, ncp_list=plot_saveat)

    # Convert t_values to NumPy array to avoid AttributeError
    t_values = np.array(t_values)

    # Extract relevant training data
    X_train = np.tile(p, (len(t_values), 1))  # Ensure correct shape
    Y_train_list = [y[:, 0], y[:, 1], y[:, 3]]  # pLV, psa, Vlv

    print("Returning X_train shape:", X_train.shape)
    print("Returning Y_train_list lengths:", [len(y_data) for y_data in Y_train_list])
    print("Returning time values shape:", t_values.shape)  # Should now work

    return X_train, Y_train_list, t_values  # Ensure three values are returned

