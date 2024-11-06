import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import uniform
import matplotlib.pyplot as plt
import math


# Valve function
def valve(R, deltaP):
    q = 0.0
    if deltaP > 0.0:
        q = deltaP / R
    return q

# ShiElastance function for elastance
def shi_elastance(t, E_min, E_max, τ, τ_es, τ_ep, Eshift):
    global tr
    t_i = math.remainder(t + (1 - Eshift) * τ, τ)
    E_p = (t_i <= τ_es) * (1 - np.cos(t_i / τ_es * np.pi)) / 2 + \
          ((t_i > τ_es) & (t_i <= τ_ep)) * (1 + np.cos((t_i - τ_es) / (τ_ep - τ_es) * np.pi)) / 2 \
          + (t_i <= τ_ep)
    E = E_min + (E_max - E_min) * E_p
    return E

# Derivative of ShiElastance
def DShiElastance(t, E_min, E_max, τ, τ_es, τ_ep, Eshift):
    global tr
    t_i = math.remainder(t + (1 - Eshift) * τ, τ)
    dE_p = (t_i <= τ_es) * np.pi / τ_es * np.sin(t_i / τ_es * np.pi) / 2 + \
           ((t_i > τ_es) & (t_i <= τ_ep)) * -np.pi / (τ_ep - τ_es) * np.sin((t_i - τ_es) / (τ_ep - τ_es) * np.pi) / 2
    dE = (E_max - E_min) * dE_p
    return dE

# Constants
Eshift = 0.0
E_min = 0.1     # Increased minimum elastance
τ_es = 0.2      # Shortened systolic duration for faster pressure build-up
τ_ep = 0.4      # Adjusted duration for realistic timing
E_max = 2.0     # Increased maximum elastance for a stronger contraction
Rmv = 0.06
Zao = 0.033
Rs = 1.11
Csa = 1.13
Csv = 11.0

# Define the mass matrix (similar to Julia)
M = np.array([
    [1., 0, 0, 0, 0, 0, 0],
    [0, 1., 0, 0, 0, 0, 0],
    [0, 0, 1., 0, 0, 0, 0],
    [0, 0, 0, 1., 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1.]
])
M_inv = np.linalg.pinv(M)

# Model differential equations (NIK function in Julia)
def NIK(t, u, p):
    pLV, psa, psv, Vlv, Qav, Qmv, Qs = u
    τ_es, τ_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min = p
    du = np.zeros_like(u)
    E = shi_elastance(t, E_min, E_max, τ, τ_es, τ_ep, Eshift)
    dE = DShiElastance(t, E_min, E_max, τ, τ_es, τ_ep, Eshift)
    du[0] = (Qmv - Qav) * E + pLV / E * dE
    du[1] = (Qav - Qs) / Csa
    du[2] = (Qs - Qmv) / Csv
    du[3] = Qmv - Qav
    du[4] = valve(Zao, pLV - psa) - Qav
    du[5] = valve(Rmv, psv - pLV) - Qmv
    du[6] = (du[1] - du[2]) / Rs
    return M_inv @ du

# Initial conditions and parameters
u0 = [70.0, 8.0, 8.0, 265.0, 0.0, 0.0, 0.0]  # Increased initial pLV to 70 for higher starting pressure
p = [τ_es, τ_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min]

# Define heart rate variability (HRV) function with reduced variability
def HRV(cycles):
    heartbeat_intervals = np.zeros(cycles)
    heartbeat_intervals[0] = 1.0  # Start with a standard interval
    for i in range(1, cycles):
        heartbeat_intervals[i] = heartbeat_intervals[i - 1] + 1.0  # Use constant interval of 1 second
    return heartbeat_intervals

# Time-varying heart rate based on HRV
heartbeat_intervals = HRV(16)
τ = heartbeat_intervals[0]
tr = 0.0
n = 0  # cycle counter

# Event-based callback functions
def condition(t, y):
    return t - tr - τ

condition.terminal = True
condition.direction = 1

def affect():
    global n, τ, tr
    n += 1
    if n < len(heartbeat_intervals):
        τ = heartbeat_intervals[n] - heartbeat_intervals[n - 1]
        tr = heartbeat_intervals[n - 1]
        print(f"Event at t = {tr:.2f}, cycle {n}, new τ = {τ:.2f}")

def solve_with_callback():
    global τ, tr, n
    t_eval = np.linspace(0, 15, 15000)  # Increased resolution to capture pressure changes accurately
    
    # Define the event condition to include τ and tr, ignoring extra args
    event_condition = lambda t, y, *args: t - tr - τ
    
    # Set the attributes for SciPy's event handling
    event_condition.terminal = True
    event_condition.direction = 1

    # Run the initial solve_ivp with the event
    sol = solve_ivp(NIK, [0, 15], u0, args=(p,), t_eval=t_eval, method='LSODA', events=event_condition, rtol=1e-8, atol=1e-8)

    # Process each event and update intervals as needed
    for i in range(len(sol.t_events[0])):  # Go through each event
        affect()
        
        # Find the index in `sol.t` closest to the event time
        event_time = sol.t_events[0][i]
        idx = np.argmin(np.abs(sol.t - event_time))
        
        # Limit t_eval to be within [event_time, 15]
        t_eval_segment = t_eval[t_eval >= event_time]
        
        # Solve the next segment with updated τ and tr
        sol_part = solve_ivp(NIK, [event_time, 15], sol.y[:, idx], args=(p,), t_eval=t_eval_segment, rtol=1e-8, atol=1e-8, events=event_condition)
        
        # Concatenate results
        sol.t = np.concatenate((sol.t, sol_part.t[1:]))  # Skip first point to avoid duplication
        sol.y = np.concatenate((sol.y, sol_part.y[:, 1:]), axis=1)
        
        if len(sol_part.t_events[0]) == 0:
            break  # No more events

    return sol

# Run the solver
solution = solve_with_callback()

# Plot results
plt.figure(figsize=(10, 8))
plt.plot(solution.t, solution.y[0], label="P_LV")
plt.plot(solution.t, solution.y[1], label="P_SA")
plt.plot(solution.t, solution.y[2], label="P_SV")
plt.plot(solution.t, solution.y[3], label="V_LV")
plt.plot(solution.t, solution.y[4], label="Q_av")
plt.plot(solution.t, solution.y[5], label="Q_mv")
plt.plot(solution.t, solution.y[6], label="Q_s")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Values")
plt.title("Cardiovascular Simulation with Heart Rate Variability")
plt.show()
