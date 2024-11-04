import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm
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
            +(t_i > τ_ep) * 0

    
    E = E_min + (E_max - E_min) * E_p
    return E

# Derivative of ShiElastance
def d_shi_elastance(t, E_min, E_max, τ, τ_es, τ_ep, Eshift):
    global tr
    t_i = math.remainder(t + (1 - Eshift) * τ, τ)
    dE_p = (t_i <= τ_es) * np.pi / τ_es * np.sin(t_i / τ_es * np.pi) / 2 + \
           ((t_i > τ_es) & (t_i <= τ_ep)) * -np.pi / (τ_ep - τ_es) * np.sin((t_i - τ_es) / (τ_ep - τ_es) * np.pi) / 2
    
    dE = (E_max - E_min) * dE_p
    return dE


# Model differential equations (NIK function in Julia)
def NIK(t, u, p, τ, tr):
    pLV, psa, psv, Vlv, Qav, Qmv, Qs = u
    τ_es, τ_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min = p
    du = np.zeros_like(u)
    
    E = shi_elastance(t, E_min, E_max, τ, τ_es, τ_ep, tr)
    dE = d_shi_elastance(t, E_min, E_max, τ, τ_es, τ_ep, tr)
    
    # Apply physiological constraints to avoid negative or excessively high values
    baseline_pressure = 5.0
    physiological_max = 200.0  # Example maximum physiological pressure

    du[0] = max(min((Qmv - Qav) * E + pLV / E * dE, physiological_max), baseline_pressure)  # Left Ventricle Pressure
    du[1] = max((Qav - Qs) / Csa, 0)  # Systemic artery pressure
    du[2] = max((Qs - Qmv) / Csv, 0)  # Systemic venous pressure
    du[3] = max(Qmv - Qav, 0)  # LV volume
    du[4] = max(valve(Zao, pLV - psa) - Qav, 0)  # Aortic valve flow
    du[5] = max(valve(Rmv, psv - pLV) - Qmv, 0)  # Mitral valve flow
    du[6] = max((du[1] - du[2]) / Rs, 0)  # Systemic flow
    return du

# Adjusted Initial conditions and parameters
u0 = [80.0, 80.0, 10.0, 120.0, 0.0, 0.0, 0.0]  # Set realistic LV pressure, systemic arterial pressure, and LV volume
p = [0.3, 0.45, 0.05, 0.05, 1.0, 2.0, 30.0, 1.5, 0.06]  # Updated parameters with realistic values
t_span = (0, 15)

# Define heartbeat intervals (HRV) using fixed intervals
heartbeat_intervals = np.cumsum(np.ones(16))  # Fixed 1-second intervals for simplicity
τ = heartbeat_intervals[0]  # Initial cycle interval
tr = 0.0  # Time reference
n = 0  # Counter for events

# Define event and affect functions for heartbeats
def heartbeat_event(t, y):
    return t - tr - τ

heartbeat_event.terminal = True
heartbeat_event.direction = 1

def affect():
    global n, τ, tr
    n += 1
    if n < len(heartbeat_intervals):
        τ = heartbeat_intervals[n] - heartbeat_intervals[n - 1]
        tr = heartbeat_intervals[n - 1]
        print(f"Event at t = {tr:.2f}, cycle {n}, new τ = {τ:.2f}")

# Solver with callback handling
def solve_with_callback():
    global τ, tr, n
    t_eval = np.linspace(0, 15, 10000)

    # Run the initial solve_ivp with the event
    sol = solve_ivp(lambda t, u: NIK(t, u, p, τ, tr), t_span, u0, method='Radau',
                    t_eval=t_eval, events=heartbeat_event, rtol=1e-10, atol=1e-10)

    # Process each event and apply the callback
    for i in range(len(sol.t_events[0])):
        affect()
        
        # Extract the time and state where the event occurred
        event_time = sol.t_events[0][i]
        
        # Find the closest index to the event time
        idx = np.argmin(np.abs(sol.t - event_time))
        
        # Slice t_eval to avoid values outside [event_time, 15]
        t_eval_segment = t_eval[t_eval >= event_time]
        
        # Continue solving from the event time
        sol_part = solve_ivp(lambda t, u: NIK(t, u, p, τ, tr), [event_time, 15], sol.y[:, idx],
                             method='Radau', t_eval=t_eval_segment, events=heartbeat_event,
                             rtol=1e-10, atol=1e-10)

        # Append the new solution segment
        sol.t = np.concatenate((sol.t, sol_part.t[1:]))  # Skip duplicate point
        sol.y = np.concatenate((sol.y, sol_part.y[:, 1:]), axis=1)
        
        if len(sol_part.t_events[0]) == 0:
            break  # Stop if no further events

    return sol


# Run the solver with callbacks
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
