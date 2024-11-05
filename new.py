import numpy as np
import matplotlib.pyplot as plt
from assimulo.problem import Implicit_Problem
from assimulo.solvers import IDA

# Define the Valve function
def Valve(R, deltaP):
    if -deltaP < 0.0:
        q = deltaP / R
    else:
        q = 0.0
    return q

# Define the ShiElastance function
def ShiElastance(t, E_min, E_max, τ, τ_es, τ_ep, Eshift, tr):
    t_i = t - tr
    if t_i <= τ_es:
        E_p = (1 - np.cos(t_i / τ_es * np.pi)) / 2
    elif τ_es < t_i <= τ_ep:
        E_p = (1 + np.cos((t_i - τ_es) / (τ_ep - τ_es) * np.pi)) / 2
    else:
        E_p = 0.0
    E = E_min + (E_max - E_min) * E_p
    return E

# Define the derivative of ShiElastance function
def DShiElastance(t, E_min, E_max, τ, τ_es, τ_ep, Eshift, tr):
    t_i = t - tr
    if t_i <= τ_es:
        DE_p = (np.pi / τ_es) * np.sin(t_i / τ_es * np.pi) / 2
    elif τ_es < t_i <= τ_ep:
        DE_p = (np.pi / (τ_ep - τ_es)) * np.sin((τ_es - t_i) / (τ_ep - τ_es) * np.pi) / 2
    else:
        DE_p = 0.0
    DE = (E_max - E_min) * DE_p
    return DE

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

    # Residual function for IDA
    def res(self, t, y, ydot):
        pLV, psa, psv, Vlv, Qav, Qmv, Qs = y
        τ_es, τ_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min = self.p
        Eshift = 0.0
        E_t = ShiElastance(t, E_min, E_max, self.τ, τ_es, τ_ep, Eshift, self.tr)
        DE_t = DShiElastance(t, E_min, E_max, self.τ, τ_es, τ_ep, Eshift, self.tr)

        res = np.zeros_like(y)
        res[0] = ydot[0] - ((Qmv - Qav) * E_t + pLV / E_t * DE_t)
        res[1] = ydot[1] - (Qav - Qs) / Csa
        res[2] = ydot[2] - (Qs - Qmv) / Csv
        res[3] = ydot[3] - (Qmv - Qav)
        res[4] = Qav - Valve(Zao, pLV - psa)
        res[5] = Qmv - Valve(Rmv, psv - pLV)
        res[6] = ydot[6] - (ydot[1] - ydot[2]) / Rs
        return res

    # Event function to update τ and tr
    def handle_event(self, solver, event_info):
        # Update parameters after event
        self.n += 1
        if self.n + 1 < len(self.t_τL):
            self.τ = self.t_τL[self.n + 1] - self.t_τL[self.n]
            self.tr = self.t_τL[self.n]
        else:
            pass  # No more τ values to update

    # Define events
    def root(self, t, y, ydot, args):
        return t - self.tr - self.τ

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
udot0 = np.zeros(7)  # Initial derivatives (can be estimated if needed)

# Initialize the model
model = CardiovascularModel(u0, udot0, p, t_τL)

# Create the problem instance
problem = Implicit_Problem(model.res, u0, udot0, 0.0)
problem.name = 'Cardiovascular Model with Events'

# Define the root function for event detection
problem.root = model.root
problem.handle_event = model.handle_event

# Set the number of roots
problem.nroots = 1

# Set up the solver
solver = IDA(problem)
solver.report_continuously = True
solver.atol = 1e-8
solver.rtol = 1e-8
solver.maxord = 5

# Simulate
tfinal = t_τL[-1] + 0.1  # Add a small buffer to ensure simulation covers all events
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
