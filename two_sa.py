#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two‑chamber Shi model with physiologically‑informed parameter bounds
└─ Polynomial‑Chaos surrogate + Sobol (window‑averaged outputs)

 • deterministic cardiac cycle (period = 1 s)
 • three scalar PCE surrogates (Plv, Psas, Vlv) over a 2 s window
 • parity plots & Sobol heat‑maps (full vs surrogate)
 • sample counts and wall‑clock times shown in titles; figs in ./results/
 • parameter bounds reflect normal‑function variability, for personalized UQ
"""
# ─────────────────── imports ───────────────────
import os, time, warnings, math
import numpy as np, matplotlib.pyplot as plt, openturns as ot
from assimulo.problem import Implicit_Problem
from assimulo.solvers.sundials import IDA
from multiprocessing import Pool, cpu_count
from SALib.sample  import saltelli
from SALib.analyze import sobol

# reproducibility & warnings
np.random.seed(2025)
warnings.filterwarnings("ignore", category=UserWarning)

# ═════════════ user‑configurable “knobs” ═════════════
DT          = 0.002              # integrator output step
WIN         = (33., 35.)         # analyse this 2‑s window
PCE_DEGREE  = 5                  # polynomial chaos degree
N_TRAIN     = 25                 # Saltelli base size for training
N_SOBOL     = 25                 # Saltelli base size for Sobol
N_CORES     = max(1, cpu_count() - 1)

# where figures will be written
RESULT_DIR  = 'results'
os.makedirs(RESULT_DIR, exist_ok=True)

# ═══════════ baseline parameter vector (Shi 2006) ════════════
param_names = [
    'v0_lv','Emin_lv','Emax_lv','tau_es_lv','tau_ed_lv',
    'v0_la','Emin_la','Emax_la','tau_es_la','tau_ed_la',
    'Zao','Rmv',
    'Csas','Rsas','Lsas',
    'Csat','Rsat','Lsat',
    'Rsar','Rscp',
    'Csvn','Rsvn'
]

BASE_P = np.array([
    10.0, 0.1,  2.5,  0.30, 0.45,     # LV
    10.0, 0.15, 0.25, 0.045, 0.09,    # LA
    0.033, 0.06,                       # valves
    0.08, 0.06, 6.2e-5,                # systemic arteries
    1.6,  0.05, 0.0017,                # systemic veins
    0.5,  0.52,                        # resistances (Rsar, Rscp)
    20.5, 0.075                        # Csvn, Rsvn
])

# ═══════════ physiologically‑informed parameter bounds ═══════════
param_bounds = [
    [  8.0,  12.0],    # v0_lv    unstressed LV volume (±20%)
    [  0.03,  0.06],   # Emin_lv  LV min elastance
    [  1.5,   3.5 ],   # Emax_lv  LV max elastance
    [  0.28,  0.32],   # tau_es_lv LV systolic duration (fraction)
    [  0.43,  0.47],   # tau_ed_lv LV relaxation duration

    [  8.0,  12.0],    # v0_la    unstressed LA volume
    [  0.10,  0.20],   # Emin_la  LA min elastance
    [  0.20,  0.30],   # Emax_la  LA max elastance
    [  0.04,  0.05],   # tau_es_la LA systolic duration
    [  0.08,  0.10],   # tau_ed_la LA relaxation duration

    [  0.01,  0.04],   # Zao      aortic valve inertance/resistance
    [  0.01,  0.06],   # Rmv      mitral valve resistance

    [  0.05,  0.10],   # Csas     aortic root compliance
    [  0.03,  0.09],   # Rsas     aortic root resistance
    [3e-5,    9e-5 ],  # Lsas     aortic root inertance

    [  1.0,   2.5 ],   # Csat     systemic arterial compliance
    [  0.03,  0.07],   # Rsat     systemic arterial resistance
    [0.001,  0.0025],  # Lsat     systemic arterial inertance

    [  0.4,   0.6 ],   # Rsar     arteriole resistance
    [  0.4,   0.6 ],   # Rscp     capillary resistance

    [ 15.0,  30.0],    # Csvn     systemic venous compliance
    [  0.05,  0.10]    # Rsvn     systemic venous resistance
]

# ═══════════ outputs fed to the surrogate ═══════════
OUT_IDX    = [0, 4, 2]         # Plv, Psas, Vlv
out_labels = ['pLV', 'pSA', 'Vlv']


# ─────────────────── Shi model code (unchanged) ───────────────────
def elastance_LV(t, Emin, Emax, τ_es, τ_ed):
    T = 1.0
    tc = t % T
    if tc <= τ_es:
        Ep = 0.5*(1-np.cos(math.pi*tc/τ_es))
        dEp = 0.5*(math.pi/τ_es)*math.sin(math.pi*tc/τ_es)
    elif tc <= τ_ed:
        x  = (tc-τ_es)/(τ_ed-τ_es)
        Ep = 0.5*(1+np.cos(math.pi*x))
        dEp =-0.5*(math.pi/(τ_ed-τ_es))*math.sin(math.pi*x)
    else:
        Ep = dEp = 0.0
    E  = Emin + (Emax - Emin)*Ep
    dE = (Emax - Emin)*dEp
    return E, dE

def elastance_LA(t, Emin, Emax, τ_es, τ_ed, shift):
    T = 1.0
    tc = (t + (1 - shift)*T) % T
    if tc <= τ_es:
        Ep  = 0.5*(1-np.cos(math.pi*tc/τ_es))
        dEp = 0.5*(math.pi/τ_es)*math.sin(math.pi*tc/τ_es)
    elif tc <= τ_ed:
        x   = (tc-τ_es)/(τ_ed-τ_es)
        Ep  = 0.5*(1+np.cos(math.pi*x))
        dEp =-0.5*(math.pi/(τ_ed-τ_es))*math.sin(math.pi*x)
    else:
        Ep = dEp = 0.0
    E  = Emin + (Emax - Emin)*Ep
    dE = (Emax - Emin)*dEp
    return E, dE

class ShiModel:
    def __init__(self, p):
        (self.v0_lv,self.Emin_lv,self.Emax_lv,self.τ_es_lv,self.τ_ed_lv,
         self.v0_la,self.Emin_la,self.Emax_la,self.τ_es_la,self.τ_ed_la,
         self.Zao ,self.Rmv,
         self.Csas,self.Rsas,self.Lsas,
         self.Csat,self.Rsat,self.Lsat,
         self.Rsar,self.Rscp,
         self.Csvn,self.Rsvn) = p
        self.shift_la = .92

    def res(self, t, y, yd):
        Plv,Pla,Vlv,Vla,Psas,Qsas,Psat,Qsat,Psvn,Qsvn,Qav,Qmv = y
        dPlv,dPla,dVlv,dVla,dPsas,dQsas,dPsat,dQsat,dPsvn,dQsvn = yd[:10]

        E_LV, dE_LV = elastance_LV(t, self.Emin_lv, self.Emax_lv,
                                   self.τ_es_lv, self.τ_ed_lv)
        E_LA, dE_LA = elastance_LA(t, self.Emin_la, self.Emax_la,
                                   self.τ_es_la, self.τ_ed_la, self.shift_la)

        Qav_calc = max((Plv-Psas)/self.Zao, 0.0)
        Qmv_calc = max((Pla-Plv)/self.Rmv, 0.0)

        r = np.zeros(12)
        r[0]  = dPlv - ((Qmv - Qav)*E_LV + dE_LV*(Vlv - self.v0_lv))
        r[1]  = dPla - ((Qsvn - Qmv)*E_LA + dE_LA*(Vla - self.v0_la))
        r[2]  = dVlv - (Qmv - Qav)
        r[3]  = dVla - (Qsvn - Qmv)
        r[4]  = dPsas - ((Qav - Qsas)/self.Csas)
        r[5]  = dQsas - ((Psas - Psat - self.Rsas*Qsas)/self.Lsas)
        r[6]  = dPsat - ((Qsas - Qsat)/self.Csat)
        r[7]  = dQsat - ((Psat - Psvn - (self.Rsat+self.Rsar+self.Rscp)*Qsat)/self.Lsat)
        r[8]  = dPsvn - ((Qsat - Qsvn)/self.Csvn)
        r[9]  = dQsvn - ((dPsvn - dVla)/self.Rsvn)
        r[10] = Qav - Qav_calc
        r[11] = Qmv - Qmv_calc
        return r

def run_model(params, tgrid):
    mdl = ShiModel(params)
    y0 = np.array([10,8,150,40,100,0,100,0,5,0,0,0], dtype=float)
    ydot0 = np.zeros_like(y0)
    prob = Implicit_Problem(mdl.res, y0, ydot0, 0.0)
    prob.algvar = np.r_[np.ones(10,int),0,0]
    solver = IDA(prob); solver.atol = solver.rtol = 1e-5
    solver.maxord = 2; solver.maxh = 0.002; solver.maxsteps = 5_000
    MAX_T = WIN[1] + 1.0
    t, y, _ = solver.simulate(MAX_T, ncp_list=tgrid)
    return np.asarray(t), np.asarray(y)

def window_mean(params, tgrid):
    t, y = run_model(params, tgrid)
    m = (t >= WIN[0]) & (t <= WIN[1])
    return np.nanmean(y[m][:, OUT_IDX], axis=0)

def wm_job(arg_tuple):
    params, tgrid = arg_tuple
    return window_mean(params, tgrid)

def build_pce(X, Y, deg):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X); Xs = sc.transform(X)
    inp = ot.Sample(Xs.tolist())
    out = ot.Sample(Y.reshape(-1,1).tolist())
    marg = [ot.Uniform(Xs[:,i].min(), Xs[:,i].max()) for i in range(Xs.shape[1])]
    algo = ot.FunctionalChaosAlgorithm(
        inp, out, ot.ComposedDistribution(marg),
        ot.SequentialStrategy(
            ot.OrthogonalProductPolynomialFactory(
                [ot.LegendreFactory()]*Xs.shape[1],
                ot.LinearEnumerateFunction(Xs.shape[1])),
            ot.LinearEnumerateFunction(Xs.shape[1]).getBasisSizeFromTotalDegree(deg)
        )
    )
    algo.run()
    meta = algo.getResult().getMetaModel()
    return lambda x: float(meta(sc.transform(np.atleast_2d(x)))[0,0])

def heatmaps(S1, ST, prefix, npts, secs, path):
    M, N = len(param_names), len(out_labels)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, lbl in zip((a1, a2), (S1, ST), ('S₁', 'Sₜ')):
        im = ax.imshow(mat, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        ax.set_xticks(range(N))
        ax.set_xticklabels(out_labels, rotation=45)
        ax.set_yticks(range(M))
        ax.set_yticklabels(param_names)
        ax.set_title(f'{prefix} {lbl}  (N={npts}, t={secs:.2f}s)')
        for i in range(M):
            for j in range(N):
                ax.text(j, i, f'{mat[i,j]:.2f}',
                        ha='center',
                        color='white' if mat[i,j] > .5 else 'black')
    cax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cax)
    plt.tight_layout(rect=(0, 0, 0.9, 1))
    plt.savefig(path, dpi=150)
    plt.close()
    
def plot_sensitivity_matrices(S1, ST, param_names, out_labels,
                              n_samples, elapsed_s, title_prefix, fname):
    """
    Draw side‑by‑side S₁ and Sₜ heatmaps with unified colorbar,
    annotated cells, and a main title.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    M, N = len(param_names), len(out_labels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=False)
    fig.suptitle(f"{title_prefix} Sensitivities (N={n_samples}, t={elapsed_s:.2f}s)", fontsize=14)

    # find common vmin/vmax
    vmax = max(np.nanmax(S1), np.nanmax(ST))
    vmin = 0.0

    for ax, mat, tag in zip((ax1, ax2), (S1, ST), ('S₁','Sₜ')):
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap='viridis', aspect='auto')
        ax.set_xticks(range(N)); ax.set_xticklabels(out_labels, rotation=45, ha='right')
        ax.set_yticks(range(M)); ax.set_yticklabels(param_names)
        ax.set_title(tag)
        # annotate
        for i in range(M):
            for j in range(N):
                val = mat[i,j]
                if np.isnan(val): txt = "–"
                else: txt = f"{val:.2f}"
                color = "white" if val > vmax*0.6 else "black"
                ax.text(j, i, txt, ha='center', va='center', color=color, fontsize=8)

    # shared colorbar on the right
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cax, label='Sensitivity')
    plt.savefig(fname, dpi=150)
    plt.close(fig)


def main():
    tgrid = np.arange(0, WIN[1] + 1.0, DT)

    # SALib problem with physiological bounds -------------------------------
    problem = {
        'num_vars': len(BASE_P),
        'names':    param_names,
        'bounds':   param_bounds
    }

    # surrogate training -----------------------------------------------------
    X_train = saltelli.sample(problem, N_TRAIN, calc_second_order=False)
    with Pool(N_CORES) as P:
        Y_train = np.vstack(P.map(wm_job, [(p, tgrid) for p in X_train]))
    surgs = [build_pce(X_train, Y_train[:,k], PCE_DEGREE) for k in range(len(OUT_IDX))]

    # validation / Sobol -----------------------------------------------------
    X_val = saltelli.sample(problem, N_SOBOL, calc_second_order=True)

    tic = time.perf_counter()
    with Pool(N_CORES) as P:
        Y_full = np.vstack(P.map(wm_job, [(p, tgrid) for p in X_val]))
    full_t = time.perf_counter() - tic

    tic = time.perf_counter()
    Y_sur = np.column_stack([[s(p) for p in X_val] for s in surgs])
    sur_t = time.perf_counter() - tic

    # Sobol indices ----------------------------------------------------------
    sob_full = {i: sobol.analyze(problem, Y_full[:,i], print_to_console=False)
                for i in range(len(OUT_IDX))}
    sob_sur  = {i: sobol.analyze(problem, Y_sur[:,i], print_to_console=False)
                for i in range(len(OUT_IDX))}
    S1_f = np.column_stack([sob_full[i]['S1'] for i in range(len(OUT_IDX))])
    ST_f = np.column_stack([sob_full[i]['ST'] for i in range(len(OUT_IDX))])
    S1_s = np.column_stack([sob_sur[i]['S1'] for i in range(len(OUT_IDX))])
    ST_s = np.column_stack([sob_sur[i]['ST'] for i in range(len(OUT_IDX))])

    # plots ------------------------------------------------------------------
    tag = len(X_val)
    plot_sensitivity_matrices(
        S1_f, ST_f,
        param_names, out_labels,
        len(X_val), full_t,
        'Full',
        os.path.join(RESULT_DIR,f'2_heat_full_{tag}.png')
    )
    plot_sensitivity_matrices(
        S1_s, ST_s,
        param_names, out_labels,
        len(X_val), sur_t,
        'Surrogate',
        os.path.join(RESULT_DIR,f'2_heat_sur_{tag}.png')
    )

    print(f'Saved to {RESULT_DIR}/ (full {full_t:.2f}s, sur {sur_t:.2f}s)')

if __name__ == '__main__':
    main()
