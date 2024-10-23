import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt
import h5py

# Valve function
def Valve(R, deltaP):
    if deltaP > 0:
        return deltaP / R
    else:
        return 0

# ShiElastance function
def ShiElastance(t, E_min, E_max, tau, tau_es, tau_ep, Eshift, tr):
    t_i = t - tr
    E_p = (t_i <= tau_es) * (1 - np.cos(t_i / tau_es * np.pi)) / 2 + \
          (t_i > tau_es) * (t_i <= tau_ep) * (1 + np.cos((t_i - tau_es) / (tau_ep - tau_es) * np.pi)) / 2
    return E_min + (E_max - E_min) * E_p

# Derivative of ShiElastance
def DShiElastance(t, E_min, E_max, tau, tau_es, tau_ep, Eshift, tr):
    t_i = t - tr
    DE_p = (t_i <= tau_es) * (np.pi / tau_es * np.sin(t_i / tau_es * np.pi)) / 2 + \
           (t_i > tau_es) * (t_i <= tau_ep) * (np.pi / (tau_ep - tau_es) * np.sin((tau_es - t_i) / (tau_ep - tau_es) * np.pi)) / 2
    return (E_max - E_min) * DE_p

# Define the mass matrix M
M = np.array([
    [1.0, 0, 0, 0, 0, 0, 0],
    [0, 1.0, 0, 0, 0, 0, 0],
    [0, 0, 1.0, 0, 0, 0, 0],
    [0, 0, 0, 1.0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],  # Zero row indicating algebraic equation
    [0, 0, 0, 0, 0, 0, 0],  # Zero row indicating algebraic equation
    [0, 0, 0, 0, 0, 0, 1.0]
])

# Inverse of the mass matrix M, but handle singular rows
# Use pseudo-inverse or block-inversion approach here (depends on the nature of the mass matrix)
M_inv = np.linalg.pinv(M)

# Main ODE system, modified by applying the mass matrix
def NIK(t, u, p, tr):
    pLV, psa, psv, Vlv, Qav, Qmv, Qs = u 
    tau_es, tau_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min = p
    Eshift = 0.0
    
    du = np.zeros_like(u)
    
    # Left ventricle dynamics
    du[0] = (Qmv - Qav) * ShiElastance(t, E_min, E_max, None, tau_es, tau_ep, Eshift, tr) + \
            pLV / ShiElastance(t, E_min, E_max, None, tau_es, tau_ep, Eshift, tr) * \
            DShiElastance(t, E_min, E_max, None, tau_es, tau_ep, Eshift, tr)

    # Systemic arteries
    du[1] = (Qav - Qs) / Csa
    
    # Venous system
    du[2] = (Qs - Qmv) / Csv
    
    # Volume dynamics
    du[3] = Qmv - Qav
    
    # Aortic valve (AV)
    du[4] = Valve(Zao, (pLV - psa)) - Qav
    
    # Mitral valve (MV)
    du[5] = Valve(Rmv, (psv - pLV)) - Qmv
    
    # Systemic flow
    du[6] = (du[1] - du[2]) / Rs
    
    # Apply the mass matrix
    return M_inv @ du

# Initial conditions and parameters
u0 = [8.0, 8.0, 8.0, 265.0, 0.0, 0.0, 0.0]
p = [0.3, 0.45, 0.06, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]
c = 16  # Number of cycles
n = 0  # Counter
tr = 0.0  # Timing variable

# HRV function
def HRV(c):
    t_tauL = np.zeros(c)
    t_tauL[0] = uniform.rvs(0.8, 1.1)
    for i in range(1, c):
        t_tauL[i] = t_tauL[i-1] + uniform.rvs(0.8, 1.1)
    return t_tauL

# Generate heart rate variability (HRV)
t_tauL = HRV(c)
tau = t_tauL[0]

# Event condition for ODE solver (no additional args)
def condition(t, y):
    global tr, tau
    return t - tr - tau

# Ensure we declare the condition as an event in solve_ivp
condition.terminal = True
condition.direction = 1

# Callback to adjust timing after each cycle
def affect_callback():
    global n, tr, tau
    n += 1
    if n < len(t_tauL) - 1:
        tau_new = t_tauL[n+1] - t_tauL[n]
        tau = tau_new
        tr = t_tauL[n]

# ODE solver function with an event to mimic callback behavior
def solve_with_event():
    global tr, tau, u0, p

    # Solve the ODE with event handling
    t_span = (0, 15)
    sol = solve_ivp(lambda t, u: NIK(t, u, p, tr), t_span, u0, method='Radau', events=condition, max_step=0.002)
    
    while sol.status == 1 and n < len(t_tauL) - 1:
        # Adjust the event-based parameters
        affect_callback()
        
        # Continue solving from the last time point
        u0_new = sol.y[:, -1]
        t_span_new = (sol.t[-1], 15)
        sol_new = solve_ivp(lambda t, u: NIK(t, u, p, tr), t_span_new, u0_new, method='Radau', events=condition, max_step=0.002)
        
        # Append new solution data
        sol.t = np.hstack((sol.t, sol_new.t[1:]))
        sol.y = np.hstack((sol.y, sol_new.y[:, 1:]))
        sol.status = sol_new.status

    return sol

# Solve the system
solution = solve_with_event()

# Create observations with noise
N = len(solution.t)
Obs = np.vstack([solution.y[0, :], solution.y[1, :], solution.y[3, :]]).T

# Adding noise to observations
ϵ = norm(0.0, 0.025)
noise = np.random.normal(0, 0.025, size=Obs.shape)
Nobs = Obs * (1 + noise)

# Plot observations
plt.figure(figsize=(10, 6))
plt.plot(solution.t, solution.y[0, :], label='LV - P')
plt.plot(solution.t, Nobs[:, 0], label='LV - P (Noisy)', linestyle='dashed')
plt.legend()
plt.show()

# Saving data to HDF5
with h5py.File('Xa.h5', 'w') as f:
    f.create_dataset('Xa', data=Nobs)

# Loading data from HDF5
with h5py.File('Xa.h5', 'r') as f:
    Xa = f['Xa'][:]

# RMSE function
def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))

# Example RMSE calculations for different parameters
rmse_tau_es = rmse(0.3, Xa[:, 0])
rmse_tau_ep = rmse(0.45, Xa[:, 1])

print(f"RMSE (tau_es): {rmse_tau_es}")
print(f"RMSE (tau_ep): {rmse_tau_ep}")

# Plot state variables
plt.plot(solution.t, solution.y[0, :], label='P_LV')
plt.plot(solution.t, solution.y[1, :], label='P_SA')
plt.plot(solution.t, solution.y[2, :], label='P_SV')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('State Variables')
plt.show()
