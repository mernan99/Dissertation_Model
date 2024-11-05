import numpy as np
from assimulo.problem import Implicit_Problem
from assimulo.solvers import IDA
from assimulo.solvers.sundials import IDA
from assimulo import solvers
import matplotlib.pyplot as plt
import random

# Define global variables for event handling
n = 0  # Cycle counter
tr = 0.0  # Reference time for the current cycle

# Valve function
def Valve(R, deltaP):
    if -deltaP < 0.0:
        q = deltaP / R
    else:
        q = 0.0
    return q

# ShiElastance function
def ShiElastance(t, E_min, E_max, tau, tau_es, tau_ep, Eshift, tr):
    t_i = t - tr
    if t_i <= tau_es:
        E_p = (1 - np.cos(t_i / tau_es * np.pi)) / 2
    elif t_i <= tau_ep:
        E_p = (1 + np.cos((t_i - tau_es) / (tau_ep - tau_es) * np.pi)) / 2
    else:
        E_p = 0.0
    E = E_min + (E_max - E_min) * E_p
    return E

# DShiElastance function (Derivative of ShiElastance)
def DShiElastance(t, E_min, E_max, tau, tau_es, tau_ep, Eshift, tr):
    t_i = t - tr
    if t_i <= tau_es:
        DE_p = (np.pi / tau_es) * np.sin(t_i / tau_es * np.pi) / 2
    elif t_i <= tau_ep:
        DE_p = (np.pi / (tau_ep - tau_es)) * np.sin((tau_es - t_i) / (tau_ep - tau_es) * np.pi) / 2
    else:
        DE_p = 0.0
    DE = (E_max - E_min) * DE_p
    return DE

# Heart Rate Variability function
def HRV(c):
    t_tauL = np.zeros(c)
    t_tauL[0] = random.uniform(0.8, 1.1)
    for i in range(1, c):
        t_tauL[i] = t_tauL[i-1] + random.uniform(0.8, 1.1)
    return t_tauL

# NIK function (DAE system)
def NIK(t, y, yd, p):
    pLV, psa, psv, Vlv, Qav, Qmv, Qs = y
    pLV_dot, psa_dot, psv_dot, Vlv_dot, Qav_dot, Qmv_dot, Qs_dot = yd
    
    tau_es, tau_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min, Eshift, tr, tau = p
    
    E = ShiElastance(t, E_min, E_max, tau, tau_es, tau_ep, Eshift, tr)
    DE = DShiElastance(t, E_min, E_max, tau, tau_es, tau_ep, Eshift, tr)
    
    res = np.zeros(7)
    # Differential equations
    res[0] = pLV_dot - ((Qmv - Qav) * E + (pLV / E) * DE)
    res[1] = psa_dot - ((Qav - Qs) / Csa)
    res[2] = psv_dot - ((Qs - Qmv) / Csv)
    res[3] = Vlv_dot - (Qmv - Qav)
    # Algebraic equations (Valve flows)
    res[4] = Qav - Valve(Zao, pLV - psa)
    res[5] = Qmv - Valve(Rmv, psv - pLV)
    # Additional differential equation
    res[6] = Qs_dot - ((psa_dot - psv_dot) / Rs)
    return res

# Initial conditions
u0 = [8.0, 8.0, 8.0, 265.0, 0.0, 0.0, 0.0]
udot0 = [0.0]*7  # Initial derivatives

# Parameters
tau_es = 0.3
tau_ep = 0.45
Rmv = 0.06
Zao = 0.033
Rs = 1.11
Csa = 1.13
Csv = 11.0
E_max = 1.5
E_min = 0.03
Eshift = 0.0

# Generate heart rate variability
c = 16
t_tauL = HRV(c)
tau = t_tauL[0]

# Event handling variables
n = 0  # Cycle counter
tr = 0.0  # Reference time

# Create the problem instance
p = [tau_es, tau_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min, Eshift, tr, tau]
model = Implicit_Problem(NIK, u0, udot0, 0.0, p)
model.name = 'Cardiovascular Model'

# Specify the mass matrix
M = np.diag([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0])
model.mass_matrix = M

# Set up the solver
sim = IDA(model)
sim.atol = [1e-8]*7
sim.rtol = 1e-8
sim.suppress_alg = True  # Suppress algebraic variables in error control
sim.algvar = [1, 1, 1, 1, 0, 0, 1]  # Specify algebraic variables

# Event function
def event_func(t, y, yd, switch):
    return t - (tr + tau)

# Handle event function
def handle_event(solver, event_info):
    global n, tau, tr, p
    n += 1
    if n+1 < len(t_tauL):
        tau_new = t_tauL[n] - t_tauL[n-1]
        tau = tau_new
        tr = t_tauL[n]
        # Update parameters in the problem
        solver.problem.p[-2] = tr  # Update tr in parameters
        solver.problem.p[-1] = tau  # Update tau in parameters
    else:
        # No more cycles, stop integration
        solver.terminate = True
options = sim.get_options()
options['terminate'] = False
options['direction'] = 1  # Positive direction
options['eventtol'] = 1e-8
options['event_max_iterations'] = 100
options['max_events'] = 10000

sim.handle_event = handle_event
sim.event = event_func

# Simulation settings
tfinal = 15.0
ncp = 1500  # Number of communication points
t, y, yd = sim.simulate(tfinal, ncp)

# Plotting the results
t = np.array(t)
y = np.array(y)

plt.figure(figsize=(12, 6))
plt.plot(t, y[:, 0], label='P_LV')
plt.plot(t, y[:, 1], label='P_SA')
plt.plot(t, y[:, 2], label='P_SV')
plt.xlabel('Time [s]')
plt.ylabel('Pressure [mmHg]')
plt.legend()
plt.title('Pressure vs. Time')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(t, y[:, 3], label='V_LV')
plt.xlabel('Time [s]')
plt.ylabel('Volume [ml]')
plt.legend()
plt.title('Left Ventricular Volume vs. Time')
plt.grid(True)
plt.show()
print(dir(IDA))
print(dir(solvers))

