# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import numpy as np
# import matplotlib.pyplot as plt

# # Chaospy with recommended orth_ttr import:
# import chaospy as cp
# from chaospy.expansion import orth_ttr  # recommended replacement for cp.orth_ttr

# # Scikit-learn for regression (using LinearRegression for OLS)
# from sklearn.linear_model import LinearRegression

# # Assimulo for solving the ODE model
# from assimulo.problem import Implicit_Problem
# from assimulo.solvers.sundials import IDA

# ############################################
# # PARTIAL SCALING UTILITY
# ############################################
# def partial_scale_fit(X, tol=1e-12):
#     """
#     Scale columns of X only if their variance > tol.
#     Returns: X_scaled, const_mask (True if nearly constant), scaler.
#     """
#     var_ = X.var(axis=0)
#     const_mask = var_ < tol
#     X_scaled = np.zeros_like(X)
#     X_scaled[:, const_mask] = X[:, const_mask]
#     from sklearn.preprocessing import StandardScaler
#     scaler = StandardScaler()
#     nonconst_mask = ~const_mask
#     X_scaled[:, nonconst_mask] = scaler.fit_transform(X[:, nonconst_mask])
#     return X_scaled, const_mask, scaler

# def partial_scale_transform(X, const_mask, scaler):
#     """
#     Apply the same partial scaling: leave nearly constant columns as-is.
#     """
#     X_scaled = np.zeros_like(X)
#     X_scaled[:, const_mask] = X[:, const_mask]
#     nonconst_mask = ~const_mask
#     X_scaled[:, nonconst_mask] = scaler.transform(X[:, nonconst_mask])
#     return X_scaled

# ############################################
# # ODE MODEL (same as your original)
# ############################################
# def Valve(R, deltaP):
#     R = max(R, 1e-6)
#     return deltaP/R if deltaP > 0 else 0.0

# def HRV(end_time):
#     """
#     Create random cycle times (each between 0.4 and 1.1 seconds).
#     """
#     t_τL = []
#     t_current = 0.0
#     while t_current < end_time + 1.0:
#         τ = np.random.uniform(0.4, 1.1)
#         t_current += τ
#         t_τL.append(t_current)
#     return np.array(t_τL)

# class CardiovascularModel:
#     """
#     Basic cardiovascular model with time-varying elastance.
#     p: [τ_es, τ_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min]
#     """
#     def __init__(self, u0, udot0, p, t_τL):
#         self.p = np.array(p)
#         self.t_τL = t_τL
#         self.τ = t_τL[0]
#         self.tr = 0.0
#         self.n = 0
#         self.u0 = u0
#         self.udot0 = udot0
#         self.event_times = [0.0]
#         self.cycle_phase = "systole"

#     def res(self, t, y, ydot):
#         pLV, psa, psv, Vlv, Qav, Qmv, Qs = y
#         τ_es, τ_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min = self.p
#         E_t = self.ShiElastance(t)
#         DE_t = self.DShiElastance(t)
#         res = np.zeros_like(y)
#         res[0] = ydot[0] - ((Qmv - Qav)*E_t + pLV/E_t*DE_t)
#         res[1] = ydot[1] - (Qav - Qs)/Csa
#         res[2] = ydot[2] - (Qs - Qmv)/Csv
#         res[3] = ydot[3] - (Qmv - Qav)
#         deltaP_av = pLV - psa
#         res[4] = Qav - Valve(Zao, deltaP_av)
#         deltaP_mv = psv - pLV
#         res[5] = Qmv - Valve(Rmv, deltaP_mv)
#         res[6] = ydot[6] - (ydot[1] - ydot[2]) / Rs
#         return res

#     def ShiElastance(self, t):
#         τ_es, τ_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min = self.p
#         t_i = (t - self.tr) % self.τ
#         if t_i <= τ_es:
#             E_p = 0.5*(1 - np.cos(np.pi*(t_i/τ_es)))
#         elif t_i <= τ_ep:
#             E_p = 0.5*(1 + np.cos(np.pi*((t_i - τ_es)/(τ_ep - τ_es))))
#         else:
#             E_p = 0.0
#         return E_min + (E_max - E_min)*E_p

#     def DShiElastance(self, t):
#         τ_es, τ_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min = self.p
#         t_i = (t - self.tr) % self.τ
#         if t_i <= τ_es:
#             dE_p = 0.5*(np.pi/τ_es)*np.sin(np.pi*t_i/τ_es)
#         elif t_i <= τ_ep:
#             dE_p = -0.5*(np.pi/(τ_ep - τ_es))*np.sin(np.pi*((t_i - τ_es)/(τ_ep - τ_es)))
#         else:
#             dE_p = 0.0
#         return (E_max - E_min)*dE_p

#     def handle_event(self, solver, event_info):
#         self.n += 1
#         self.tr = round(solver.t, 6)
#         self.event_times.append(self.tr)
#         if self.cycle_phase == "systole":
#             self.cycle_phase = "diastole"
#         else:
#             self.cycle_phase = "systole"
#         if self.n+1 < len(self.t_τL):
#             self.τ = max(1e-6, self.t_τL[self.n+1] - self.t_τL[self.n])

#     def root(self, t, y, ydot):
#         τ_es = self.p[0]
#         t_i = (t - self.tr) % self.τ
#         return np.array([t_i - τ_es])

# def run_baseline_ODE(p, end_time=43):
#     """
#     Run the ODE using the CardiovascularModel.
#     Returns: t_vals, Y, event_times.
#     Y columns: [pLV, pSA, pSV, Vlv, Qav, Qmv, Qs].
#     """
#     t_τL = HRV(end_time)
#     u0 = np.array([8.0, 8.0, 8.0, 265.0, 0.0, 0.0, 0.0])
#     udot0 = np.zeros(7)
#     model = CardiovascularModel(u0, udot0, p, t_τL)
#     problem = Implicit_Problem(model.res, u0, udot0, 0.0)
#     problem.root = model.root
#     problem.handle_event = model.handle_event
#     problem.nroots = 1
#     solver = IDA(problem)
#     solver.atol = 1e-6
#     solver.rtol = 1e-6
#     solver.maxord = 5
#     solver.maxh = 0.02
#     solver.maxsteps = 100000
#     tfinal = t_τL[-1] + 0.1
#     plot_saveat = np.arange(0, tfinal+0.001, 0.001)
#     t_vals, Y, Ydot = solver.simulate(tfinal, ncp_list=plot_saveat)
#     return np.array(t_vals), np.array(Y), np.array(model.event_times)

# ############################################
# # SURROGATE: ENGINEERED FEATURES WITH ADDITIONAL HARMONICS
# ############################################
# def build_surrogate_phase_elastance(t_vals, Y, event_times, p, deg=5):
#     """
#     Build a PCE surrogate from engineered features computed from a single cycle.
#     Engineered features are:
#       - Normalized phase:  φ = (t - t_start) / T_cycle  ∈ [0, 1]
#       - Normalized elastance: computed using the same analytic formula as ODE, then normalized to [0, 1]
#       - sin(2πφ) and cos(2πφ)
#       - sin(4πφ) and cos(4πφ)
#     The surrogate maps these 6 features to outputs [pLV, pSA, Vlv].
#     The cycle is taken as the last complete cycle if available,
#     otherwise the fixed window [33,35] s is used.
#     """
#     # Determine the time window to use.
#     if len(event_times) >= 2:
#         t_start = event_times[-2]
#         t_end = event_times[-1]
#     else:
#         t_start = 33.0
#         t_end = 35.0
#     T_cycle = t_end - t_start

#     # Extract data from the chosen cycle/window.
#     mask = (t_vals >= t_start) & (t_vals <= t_end)
#     t_window = t_vals[mask]
#     Y_window = Y[mask, :]

#     # Compute normalized phase: φ = (t - t_start) / T_cycle
#     phase = (t_window - t_start) / T_cycle

#     # Compute elastance E(t) using the same piecewise function as in the ODE.
#     τ_es, τ_ep, _, _, _, _, _, E_max, E_min = p
#     t_rel = t_window - t_start
#     E_p_array = np.zeros_like(t_rel)
#     for i, t_val in enumerate(t_rel):
#         if t_val <= τ_es:
#             E_p_array[i] = 0.5*(1 - np.cos(np.pi*(t_val/τ_es)))
#         elif t_val <= τ_ep:
#             E_p_array[i] = 0.5*(1 + np.cos(np.pi*((t_val - τ_es)/(τ_ep - τ_es))))
#         else:
#             E_p_array[i] = 0.0
#     E_vals = E_min + (E_max - E_min)*E_p_array
#     # Normalize elastance to [0,1]:
#     E_feature = (E_vals - E_min) / (E_max - E_min)

#     # Compute additional trigonometric features
#     phi_sin = np.sin(2*np.pi*phase)
#     phi_cos = np.cos(2*np.pi*phase)
#     phi_sin2 = np.sin(4*np.pi*phase)
#     phi_cos2 = np.cos(4*np.pi*phase)

#     # Form the engineered feature matrix X_train (6 features)
#     X_train = np.column_stack((phase, E_feature, phi_sin, phi_cos, phi_sin2, phi_cos2))
    
#     # Define outputs: use pLV (col 0), pSA (col 1), and Vlv (col 3)
#     pLV = Y_window[:, 0]
#     pSA = Y_window[:, 1]
#     Vlv = Y_window[:, 3]
#     Y_train = np.column_stack((pLV, pSA, Vlv))
    
#     # (Optional) Downsample if needed
#     factor = 10
#     if X_train.shape[0] > 200000:
#         X_train = X_train[::factor, :]
#         Y_train = Y_train[::factor, :]
#         print(f"Downsampled training data to {X_train.shape[0]} points.")

#     # Partial scaling on the 6D input.
#     X_scaled, const_mask, scaler = partial_scale_fit(X_train)

#     # Define joint distribution for the 6 features.
#     # Phase and E_feature are in [0,1]; the trig features are in [-1,1].
#     dist = cp.J(
#         cp.Uniform(0,1),         # phase
#         cp.Uniform(0,1),         # normalized elastance
#         cp.Uniform(-1,1),        # sin(2πφ)
#         cp.Uniform(-1,1),        # cos(2πφ)
#         cp.Uniform(-1,1),        # sin(4πφ)
#         cp.Uniform(-1,1)         # cos(4πφ)
#     )
#     poly_exp = orth_ttr(deg, dist)

#     # Fit a separate PCE surrogate for each output using OLS.
#     expansions = []
#     X_scaled_t = X_scaled.T   # shape: (6, n_samples)
#     out_dim = Y_train.shape[1]
#     ols_model = LinearRegression(fit_intercept=False)
#     for odim in range(out_dim):
#         y_i = Y_train[:, odim]
#         poly_coef = cp.fit_regression(poly_exp, X_scaled_t, y_i, model=ols_model)
#         expansions.append(poly_coef)

#     def surrogate_fn(X_input):
#         """
#         Evaluate the surrogate given a new input array X_input of shape (#samples, 6):
#         Columns: [phase, E_feature, sin(2πφ), cos(2πφ), sin(4πφ), cos(4πφ)].
#         Returns a (#samples, 3) matrix.
#         """
#         X_input = np.array(X_input)
#         if X_input.ndim == 1:
#             X_input = X_input[None, :]
#         X_input_scaled = partial_scale_transform(X_input, const_mask, scaler)
#         X_input_scaled_t = X_input_scaled.T   # shape: (6, #samples)
#         n_eval = X_input.shape[0]
#         res_mat = np.zeros((n_eval, out_dim))
#         for odim, poly_coef in enumerate(expansions):
#             args = [X_input_scaled_t[d, :] for d in range(6)]
#             res_mat[:, odim] = poly_coef(*args)
#         return res_mat

#     return surrogate_fn, (t_start, t_end)

# ############################################
# # MAIN FUNCTION
# ############################################
# def main():
#     # (1) Baseline parameter (fixed)
#     p_baseline = [0.3, 0.45, 0.06, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]

#     # (2) Run the ODE model (returns time values, Y, event times)
#     t_vals, Y, ev_times = run_baseline_ODE(p_baseline, end_time=43)
#     print("ODE shape =", Y.shape)
#     print("Event times =", ev_times)

#     # (3) Build the surrogate using engineered features over a selected cycle.
#     surrogate_fn, cycle_range = build_surrogate_phase_elastance(t_vals, Y, ev_times, p_baseline, deg=5)
#     t_cycle_start, t_cycle_end = cycle_range
#     print("Using cycle/window: [{:.2f}, {:.2f}] s".format(t_cycle_start, t_cycle_end))
    
#     # (4) For testing, select a window (here we use [33,35] seconds).
#     test_start = 33.0
#     test_end = 35.0
#     mask_test = (t_vals >= test_start) & (t_vals <= test_end)
#     t_test = t_vals[mask_test]
#     Y_test = Y[mask_test, :]

#     # To evaluate the surrogate, we must build the corresponding feature set.
#     # We use the cycle from the surrogate (t_cycle_start, t_cycle_end) for computing features.
#     T_cycle = t_cycle_end - t_cycle_start
#     # For each test time, we clamp to [t_cycle_start, t_cycle_end] (if necessary)
#     t_clamped = np.clip(t_test, t_cycle_start, t_cycle_end)
#     phase_test = (t_clamped - t_cycle_start) / T_cycle
#     # Compute elastance using the same procedure:
#     τ_es, τ_ep, _, _, _, _, _, E_max, E_min = p_baseline
#     t_rel_test = t_clamped - t_cycle_start
#     E_p_test = np.zeros_like(t_rel_test)
#     for i, t_val in enumerate(t_rel_test):
#         if t_val <= τ_es:
#             E_p_test[i] = 0.5*(1 - np.cos(np.pi*(t_val/τ_es)))
#         elif t_val <= τ_ep:
#             E_p_test[i] = 0.5*(1 + np.cos(np.pi*((t_val - τ_es)/(τ_ep - τ_es))))
#         else:
#             E_p_test[i] = 0.0
#     E_val_test = E_min + (E_max - E_min)*E_p_test
#     E_feature_test = (E_val_test - E_min) / (E_max - E_min)

#     # Compute trig features (first and second harmonic):
#     phi_sin_test = np.sin(2*np.pi*phase_test)
#     phi_cos_test = np.cos(2*np.pi*phase_test)
#     phi_sin2_test = np.sin(4*np.pi*phase_test)
#     phi_cos2_test = np.cos(4*np.pi*phase_test)
    
#     # Form the test feature matrix: shape (#samples, 6)
#     X_test = np.column_stack((phase_test, E_feature_test, phi_sin_test, phi_cos_test, phi_sin2_test, phi_cos2_test))
#     Y_sur = surrogate_fn(X_test)

#     # (5) Plot comparisons.
#     plt.figure(figsize=(10,5))
#     plt.plot(t_test, Y_test[:,0], label="True pLV")
#     plt.plot(t_test, Y_test[:,1], label="True pSA")
#     plt.plot(t_test, Y_test[:,2], label="True pSV", alpha=0.5)
#     plt.plot(t_test, Y_sur[:,0], "--", label="Surrogate pLV")
#     plt.plot(t_test, Y_sur[:,1], "--", label="Surrogate pSA")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Pressure")
#     plt.title(f"Surrogate vs. True Model (Pressures) in [{test_start}, {test_end}] s, deg=5")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     plt.figure()
#     plt.plot(t_test, Y_test[:,3], label="True Vlv")
#     plt.plot(t_test, Y_sur[:,2], "--", label="Surrogate Vlv")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Volume")
#     plt.title(f"Surrogate vs. True Model (Volume) in [{test_start}, {test_end}] s, deg=5")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive surrogate and validation workflow using OpenTURNS:
 1) Run baseline ODE model.
 2) Engineer phase-elastance-harmonic features from one cardiac cycle.
 3) Build PCE surrogates for pLV, pSA, Vlv using OpenTURNS.
 4) Evaluate and compare surrogate vs. true model over a test window.
"""
import numpy as np
import matplotlib.pyplot as plt
import openturns as ot
from assimulo.problem import Implicit_Problem
from assimulo.solvers.sundials import IDA

# ================= Configuration =================
p_baseline = [0.3, 0.45, 0.06, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]
degree = 5  # PCE degree for surrogate
# ================================================

def HRV(end_time):
    arr, t = [], 0.0
    while t < end_time + 1:
        t += np.random.uniform(0.4, 1.1)
        arr.append(t)
    return np.array(arr)

def Valve(R, deltaP):
    R = max(R, 1e-6)
    return deltaP/R if deltaP > 0 else 0.0

class CardiovascularModel:
    def __init__(self, u0, udot0, p, tL):
        self.p = np.array(p)
        self.tL = tL
        self.cycle = tL[0]
        self.tref = 0.0
        self.count = 0
        self.u0 = u0
        self.udot0 = udot0
    def residual(self, t, y, ydot):
        pLV, psa, psv, Vlv, Qav, Qmv, Qs = y
        tau_es, tau_ep, Rmv, Zao, Rs, Csa, Csv, Emax, Emin = self.p
        phase = (t - self.tref) % self.cycle
        # elastance
        if phase <= tau_es:
            phi = (1 - np.cos(np.pi*phase/tau_es)) / 2
        elif phase <= tau_ep:
            phi = (1 + np.cos(np.pi*(phase - tau_es)/(tau_ep - tau_es))) / 2
        else:
            phi = 0.0
        E = Emin + (Emax - Emin) * phi
        # dE
        if phase <= tau_es:
            dphi = (np.pi / tau_es) * np.sin(np.pi * phase / tau_es) / 2
        elif phase <= tau_ep:
            dphi = - (np.pi / (tau_ep - tau_es)) * np.sin(np.pi * (phase - tau_es) / (tau_ep - tau_es)) / 2
        else:
            dphi = 0.0
        dE = (Emax - Emin) * dphi
        res = np.zeros_like(y)
        res[0] = ydot[0] - ((Qmv - Qav)*E + pLV/E*dE)
        res[1] = ydot[1] - (Qav - Qs)/Csa
        res[2] = ydot[2] - (Qs - Qmv)/Csv
        res[3] = ydot[3] - (Qmv - Qav)
        res[4] = Qav - Valve(Zao, pLV - psa)
        res[5] = Qmv - Valve(Rmv, psv - pLV)
        res[6] = ydot[6] - (ydot[1] - ydot[2]) / Rs
        return res
    def root(self, t, y, ydot):
        phase = (t - self.tref) % self.cycle
        return np.array([phase - self.p[0]])
    def handle_event(self, solver, info):
        self.count += 1
        self.tref = round(solver.t, 6)
        if self.count + 1 < len(self.tL):
            self.cycle = max(1e-6, self.tL[self.count+1] - self.tL[self.count])
        else:
            solver.terminate = True

# Run baseline ODE
def run_baseline(end_time=43):
    tL = HRV(end_time)
    u0 = np.array([8,8,8,265,0,0,0]); udot0 = np.zeros(7)
    model = CardiovascularModel(u0, udot0, p_baseline, tL)
    prob = Implicit_Problem(model.residual, u0, udot0, 0.0)
    prob.root = model.root
    prob.handle_event = model.handle_event
    prob.nroots = 1
    solver = IDA(prob)
    solver.atol = solver.rtol = 1e-6
    solver.maxord, solver.maxh, solver.maxsteps = 5, 0.02, 100000
    tf = tL[-1] + 0.1
    t, Y, _ = solver.simulate(tf, ncp_list=np.arange(0, tf+0.001, 0.001))
    return np.array(t), np.array(Y), np.array(model.tL)

# Engineered features extraction

def compute_engineered_features(times, t0, t1, tau_es, tau_ep, E_min, E_max):
    """
    Compute features for given times and cycle [t0, t1].
    Returns feature matrix shape (n_times, 6).
    """
    T = t1 - t0
    t_rel = np.clip(times, t0, t1) - t0
    phase = (t_rel / T)
    # elastance profile
    Ep = np.where(
        t_rel <= tau_es,
        0.5*(1 - np.cos(np.pi * t_rel / tau_es)),
        np.where(
            t_rel <= tau_ep,
            0.5*(1 + np.cos(np.pi * (t_rel - tau_es)/(tau_ep - tau_es))),
            0.0
        )
    )
    E_feat = (E_min + (E_max - E_min)*Ep - E_min)/(E_max - E_min)
    return np.column_stack([
        phase,
        E_feat,
        np.sin(2*np.pi * phase), np.cos(2*np.pi * phase),
        np.sin(4*np.pi * phase), np.cos(4*np.pi * phase)
    ])

# Determine appropriate cycle window for given test start time
def find_cycle_window(ev_times, test_start):
    """
    Find the cycle [t0, t1] that contains test_start.
    Fallback to first beat if out of range.
    """
    idx = np.searchsorted(ev_times, test_start)
    if idx == 0:
        return ev_times[0], ev_times[1]
    if idx >= len(ev_times):
        return ev_times[-2], ev_times[-1]
    return ev_times[idx-1], ev_times[idx]

# Extract features and training data for a specific cycle
def extract_cycle_data(t_vals, Y, t0, t1):
    """
    Extract training features and outputs over the cycle [t0, t1].
    """
    mask = (t_vals >= t0) & (t_vals <= t1)
    t_window = t_vals[mask]
    Y_window = Y[mask]
    X_train = compute_engineered_features(
        t_window, t0, t1,
        p_baseline[0], p_baseline[1], p_baseline[-1], p_baseline[-2]
    )
    Y_train = np.column_stack([Y_window[:,0], Y_window[:,1], Y_window[:,3]])
    return X_train, Y_train, (t0, t1)

# OpenTURNS surrogate builder and evaluator
def build_openturns_surrogate(X, Y, degree):
    dim = X.shape[1]
    marginals = [
        ot.Uniform(np.min(X[:,i]), np.max(X[:,i])) for i in range(dim)
    ]
    dist = ot.ComposedDistribution(marginals)
    input_sample = ot.Sample(X.tolist())
    output_sample = ot.Sample(Y.tolist())
    basis = ot.OrthogonalProductPolynomialFactory(
        [ot.LegendreFactory()]*dim,
        ot.LinearEnumerateFunction(dim)
    )
    strategy = ot.SequentialStrategy(
        basis,
        ot.LinearEnumerateFunction(dim).getBasisSizeFromTotalDegree(degree)
    )
    algo = ot.FunctionalChaosAlgorithm(
        input_sample, output_sample, dist, strategy
    )
    algo.run()
    return algo.getResult().getMetaModel()

# Main workflow
def main():
    # 1) simulate
    t_vals, Y, ev_times = run_baseline()
            # 2) training data: pick the cycle containing our test window
    test_start = 33.0
    t0, t1 = find_cycle_window(ev_times, test_start)
    X_train, Y_train, _ = extract_cycle_data(t_vals, Y, t0, t1)
    print(f"Training cycle [{t0:.2f},{t1:.2f}] with {X_train.shape[0]} points")
    # 3) build surrogates for each of 3 outputs
    metas = []
    for i in range(3):
        meta = build_openturns_surrogate(X_train, Y_train[:,i:i+1], degree)
        metas.append(meta)
    # 4) test window
    mask_test = (t_vals >= 33) & (t_vals <= 35)
    t_test = t_vals[mask_test]
    Y_true = Y[mask_test]
    # 5) test features
    X_test = compute_engineered_features(
        t_test, t0, t1,
        p_baseline[0], p_baseline[1], p_baseline[-1], p_baseline[-2]
    )
    # 6) evaluate surrogates
    Y_sur = np.hstack([
        np.array(meta(ot.Sample(X_test.tolist()))) for meta in metas
    ])
    # 7) plot comparisons
    plt.figure(figsize=(8,5))
    plt.plot(t_test, Y_true[:,0], label='True pLV')
    plt.plot(t_test, Y_sur[:,0], '--', label='Sur pLV')
    plt.plot(t_test, Y_true[:,1], label='True pSA')
    plt.plot(t_test, Y_sur[:,1], '--', label='Sur pSA')
    plt.xlabel('Time (s)'); plt.ylabel('Pressure')
    plt.legend(); plt.grid(True); plt.show()
    plt.figure(figsize=(6,4))
    plt.plot(t_test, Y_true[:,3], label='True Vlv')
    plt.plot(t_test, Y_sur[:,2], '--', label='Sur Vlv')
    plt.xlabel('Time (s)'); plt.ylabel('Volume')
    plt.legend(); plt.grid(True); plt.show()

if __name__ == '__main__':
    main()
