#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import openturns as ot
import time

from assimulo.problem import Implicit_Problem
from assimulo.solvers.sundials import IDA

############################################
# 1) FULL MODEL CODE (same as before)
############################################
def Valve(R, deltaP):
    R = max(R, 1e-6)
    return deltaP / R if deltaP > 0 else 0.0

def HRV(end_time):
    t_τL = []
    t_current = 0.0
    while t_current < end_time + 1.0:
        tau = np.random.uniform(0.4, 1.1)
        t_current += tau
        t_τL.append(t_current)
    return np.array(t_τL)

class CardiovascularModel:
    """
    A simple cardiovascular model using time-varying elastance.
    p = [tau_es, tau_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min]
    """
    def __init__(self, u0, udot0, p, t_τL):
        self.p = np.array(p)
        self.t_τL = t_τL
        self.tau = t_τL[0]
        self.tr = 0.0
        self.n = 0
        self.u0 = u0
        self.udot0 = udot0
        self.event_times = [0.0]
        self.cycle_phase = "systole"

    def res(self, t, y, ydot):
        pLV, psa, psv, Vlv, Qav, Qmv, Qs = y
        tau_es, tau_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min = self.p
        E_t = self.ShiElastance(t)
        DE_t = self.DShiElastance(t)
        res = np.zeros_like(y)
        res[0] = ydot[0] - ((Qmv - Qav)*E_t + pLV/E_t * DE_t)
        res[1] = ydot[1] - (Qav - Qs)/Csa
        res[2] = ydot[2] - (Qs - Qmv)/Csv
        res[3] = ydot[3] - (Qmv - Qav)
        res[4] = Qav - Valve(Zao, pLV - psa)
        res[5] = Qmv - Valve(Rmv, psv - pLV)
        res[6] = ydot[6] - (ydot[1]-ydot[2]) / Rs
        return res

    def ShiElastance(self, t):
        tau_es, tau_ep, _, _, _, _, _, E_max, E_min = self.p
        t_i = (t - self.tr) % self.tau
        if t_i <= tau_es:
            E_p = 0.5*(1 - np.cos(np.pi*(t_i/tau_es)))
        elif t_i <= tau_ep:
            E_p = 0.5*(1 + np.cos(np.pi*((t_i - self.p[0])/(tau_ep-self.p[0]))))
        else:
            E_p = 0.0
        return E_min + (E_max - E_min)*E_p

    def DShiElastance(self, t):
        tau_es, tau_ep, _, _, _, _, _, E_max, E_min = self.p
        t_i = (t - self.tr) % self.tau
        if t_i <= tau_es:
            dE_p = 0.5*(np.pi/tau_es)*np.sin(np.pi*t_i/tau_es)
        elif t_i <= tau_ep:
            dE_p = -0.5*(np.pi/(tau_ep - self.p[0]))*np.sin(np.pi*((t_i-self.p[0])/(tau_ep-self.p[0])) )
        else:
            dE_p = 0.0
        return (E_max - E_min)*dE_p

    def handle_event(self, solver, event_info):
        self.n += 1
        self.tr = round(solver.t, 6)
        self.event_times.append(self.tr)
        self.cycle_phase = "diastole" if self.cycle_phase=="systole" else "systole"
        if self.n+1 < len(self.t_τL):
            self.tau = max(1e-6, self.t_τL[self.n+1] - self.t_τL[self.n])
        else:
            solver.terminate = True

    def root(self, t, y, ydot):
        tau_es = self.p[0]
        t_i = (t - self.tr) % self.tau
        return np.array([t_i - tau_es])

def run_baseline_ODE(p, end_time=43):
    t_τL = HRV(end_time)
    u0 = np.array([8.0, 8.0, 8.0, 265.0, 0.0, 0.0, 0.0])
    udot0 = np.zeros(7)
    model = CardiovascularModel(u0, udot0, p, t_τL)
    prob = Implicit_Problem(model.res, u0, udot0, 0.0)
    prob.root = model.root
    prob.handle_event = model.handle_event
    prob.nroots = 1
    solver = IDA(prob)
    solver.atol = 1e-6
    solver.rtol = 1e-6
    solver.maxord = 5
    solver.maxh = 0.02
    solver.maxsteps = 200000
    tfinal = t_τL[-1] + 0.1
    plot_saveat = np.arange(0, tfinal+0.001, 0.001)
    t_vals, Y, _ = solver.simulate(tfinal, ncp_list=plot_saveat)
    return np.array(t_vals), np.array(Y)

############################################
# 2) TIME-SERIES COMPARISON PLOT
############################################
def plot_time_series_comparison(
    t_vals, Y_true,
    t_vals_surr, Y_surr,
    t_range=(33.0, 35.0),
    deg=5
):
    """
    Plot side-by-side comparison for pLV, pSA, pSV (pressures) and Vlv (volume)
    over the specified time range (default [33,35]).
    deg is just for annotation (the polynomial degree used).
    Y_true, Y_surr each shape: (num_time_points, 4) => columns are pLV, pSA, pSV, Vlv.
    """
    # Mask the time range
    mask_true = (t_vals >= t_range[0]) & (t_vals <= t_range[1])
    mask_surr = (t_vals_surr >= t_range[0]) & (t_vals_surr <= t_range[1])

    # Extract times & data
    t_true = t_vals[mask_true]
    pLV_true = Y_true[mask_true, 0]
    pSA_true = Y_true[mask_true, 1]
    pSV_true = Y_true[mask_true, 2]
    Vlv_true = Y_true[mask_true, 3]

    t_surr = t_vals_surr[mask_surr]
    pLV_surr = Y_surr[mask_surr, 0]
    pSA_surr = Y_surr[mask_surr, 1]
    pSV_surr = Y_surr[mask_surr, 2]
    Vlv_surr = Y_surr[mask_surr, 3]

    # 1) Pressures Plot
    plt.figure(figsize=(10,6))
    plt.plot(t_true, pLV_true, label="True pLV", color="blue")
    plt.plot(t_true, pSA_true, label="True pSA", color="red")
    plt.plot(t_true, pSV_true, label="True pSV", color="green")
    plt.plot(t_surr, pLV_surr, "--", label="Surrogate pLV", color="blue", alpha=0.7)
    plt.plot(t_surr, pSA_surr, "--", label="Surrogate pSA", color="red", alpha=0.7)
    plt.plot(t_surr, pSV_surr, "--", label="Surrogate pSV", color="green", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure")
    plt.title(f"Surrogate vs. True Model (Pressures) in [{t_range[0]}, {t_range[1]}], deg={deg}")
    plt.legend()
    plt.grid(True)
    plt.xlim(*t_range)
    plt.show()

    # 2) Volume Plot
    plt.figure(figsize=(8,6))
    plt.plot(t_true, Vlv_true, label="True Vlv", color="tab:blue")
    plt.plot(t_surr, Vlv_surr, "--", label="Surrogate Vlv", color="tab:orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Volume")
    plt.title(f"Surrogate vs. True Model (Volume) in [{t_range[0]}, {t_range[1]}], deg={deg}")
    plt.legend()
    plt.grid(True)
    plt.xlim(*t_range)
    plt.show()

############################################
# 3) EXAMPLE MAIN
############################################
def main():
    start_time = time.time()

    # Baseline param for demonstration
    p_baseline = [0.3, 0.45, 0.06, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]
    # Suppose we have a time-based surrogate approach that yields pLV(t), pSA(t), pSV(t), Vlv(t).
    # For demonstration, let's create "fake" surrogate data by injecting small offsets or noise.

    # 1) Get the true time-series at baseline param
    t_full, Y_full = run_baseline_ODE(p_baseline, end_time=35)
    # Suppose Y_full has shape (n_points, 7). We'll pick columns [0->pLV, 1->pSA, 2->pSV, 3->Vlv].
    # Let's store them in a separate array shape (n_points, 4).
    # (You might adapt your code if pSA is Y_full[:,1], pSV=Y_full[:,2], etc.)

    # We'll create a sub-array with just the columns we want to plot
    Y_true_extract = np.column_stack([Y_full[:,0], # pLV
                                      Y_full[:,1], # pSA
                                      Y_full[:,2], # pSV
                                      Y_full[:,3]])# Vlv

    # 2) Create "fake" Surrogate data, just for demonstration
    # If you have an actual time-based surrogate, evaluate it at each time in t_full.
    # For example:
    Y_surr_extract = Y_true_extract.copy()
    # let's tweak it slightly to mimic a surrogate mismatch
    Y_surr_extract[:, 0] *= 0.98  # pLV offset
    Y_surr_extract[:, 1] *= 1.02  # pSA offset
    Y_surr_extract[:, 2] *= 1.01  # pSV offset
    Y_surr_extract[:, 3] *= 0.99  # Vlv offset

    # 3) Plot side by side, e.g. for deg=5
    plot_time_series_comparison(
        t_vals=t_full,
        Y_true=Y_true_extract,
        t_vals_surr=t_full,  # same times, but in practice might differ
        Y_surr=Y_surr_extract,
        t_range=(33.0,35.0),
        deg=5
    )

    end_time_total = time.time()
    print(f"Done. Total runtime: {end_time_total - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
