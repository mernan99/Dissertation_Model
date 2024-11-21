import numpy as np
import matplotlib.pyplot as plt
from assimulo.problem import Implicit_Problem
from assimulo.solvers.sundials import IDA
import math
from SALib.analyze import sobol

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

    # Event function to update τ and tr
    def handle_event(self, solver, event_info):
        """
        Handles events detected by the root function.
        Updates the cardiac cycle phase, elastance parameters, and time reference.
        """
        self.n += 1

        if self.cycle_phase == "systole":
            # Switch to diastole
            self.cycle_phase = "diastole"
            # Update reference time for diastole
            self.tr = solver.t
            # Update parameters for relaxation (change elastance accordingly)
        elif self.cycle_phase == "diastole":
            # Switch to systole
            self.cycle_phase = "systole"
            if self.n + 1 < len(self.t_τL):
                # Update tr and τ for next systolic event
                self.τ = self.t_τL[self.n + 1] - self.t_τL[self.n]
                self.tr = self.t_τL[self.n]
            else:
                # No more events, end the simulation
                solver.terminate = True

    def root(self, t, y, ydot):
        """
        Detects events that correspond to changes in the cardiac cycle.
        Specifically detects when to switch between systole and diastole.
        """
        if self.cycle_phase == "systole":
            # Event to detect the end of systole (e.g., when elastance reaches max)
            return t - (self.tr + self.τ)
        elif self.cycle_phase == "diastole":
            # Event to detect the end of diastole (e.g., when filling ends)
            return t - (self.tr + self.τ)
        return 0.0  # This should return values that cross zero during the transitions


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
