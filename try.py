#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time‑resolved PCE surrogate (POD + PCE) for the 33‑35 s window
--------------------------------------------------------------
• full model: deterministic‑beat CardioODE
• surrogate:  point‑wise reconstruction on a 0.02 s grid (99 % POD energy)
• parity plots & Sobol heat‑maps use 33‑35 s window‑mean outputs
• extra figure: full waveform vs. surrogate waveform (dashed)
"""
# ─────────────────── imports ───────────────────
import os, time, warnings, math, numpy as np, matplotlib.pyplot as plt, openturns as ot
from assimulo.problem import Implicit_Problem
from assimulo.solvers.sundials import IDA
from multiprocessing import Pool, cpu_count
from SALib.sample  import saltelli
from SALib.analyze import sobol
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
np.random.seed(2025)
warnings.filterwarnings('ignore', category=UserWarning)

# ═════════════ user “knobs” ═════════════
DT_MODEL        = 0.02            # integration output step
WIN             = (33., 35.)      # analysis window
DT_SLICE        = 0.02            # resolution of the time‑series surrogate
PCE_DEGREE      = 4
N_TRAIN, N_VAL  = 20, 20          # Saltelli base sizes
N_CORES         = max(1, cpu_count()-1)
ENERGY_KEEP     = 0.99            # POD energy to retain
# ═══════════════════════════════════════

out_labels   = ['pLV', 'pSA', 'pSV', 'Vlv']                # ← 4 signals
param_names  = ['tau_es','tau_ep','Rmv','Zao','Rs','Csa','Csv','E_max','E_min']
param_bounds = [[0.21,0.34], [0.36,0.50], [0.042,0.078], [0.0231,0.0429],
                [0.777,1.443], [0.791,1.469], [7.7,14.3], [1.05,1.95],
                [0.025,0.039]]
RESULT_DIR   = 'results'; os.makedirs(RESULT_DIR, exist_ok=True)

# =============== helper functions ===========================================
def HRV(end_t):
    seq, t = [], 0.0
    while t < end_t + 1:
        t += np.random.uniform(0.8, 1.1)
        seq.append(t)
    return np.asarray(seq)

def Valve(R,dP): return dP/max(R,1e-6) if dP>0 else 0.

# ---------------- ODE -------------------------------------------------------
class CardioODE:
    def __init__(self,u0,ud0,p,beats):
        self.p,self.beats=np.asarray(p),beats
        self.cycle,self.tref,self.i = beats[0],0.,0
    def residual(self,t,y,yd):
        pLV,psa,psv,Vlv,Qav,Qmv,Qs = y
        τ_es,τ_ep,Rmv,Zao,Rs,Csa,Csv,Emax,Emin = self.p
        ph=(t-self.tref)%self.cycle
        if ph<=τ_es:
            φ=.5*(1-np.cos(math.pi*ph/τ_es)); dφ=.5*(math.pi/τ_es)*math.sin(math.pi*ph/τ_es)
        elif ph<=τ_ep:
            φ=.5*(1+np.cos(math.pi*(ph-τ_es)/(τ_ep-τ_es)))
            dφ=-.5*(math.pi/(τ_ep-τ_es))*math.sin(math.pi*(ph-τ_es)/(τ_ep-τ_es))
        else: φ=dφ=0.
        E  = Emin+(Emax-Emin)*φ
        dE = (Emax-Emin)*dφ
        r  = np.zeros_like(y)
        r[0]=yd[0]-((Qmv-Qav)*E + pLV/E*dE)
        r[1]=yd[1]-(Qav-Qs)/Csa
        r[2]=yd[2]-(Qs-Qmv)/Csv
        r[3]=yd[3]-(Qmv-Qav)
        r[4]=Qav-Valve(Zao,pLV-psa)
        r[5]=Qmv-Valve(Rmv,psv-pLV)
        r[6]=yd[6]-(yd[1]-yd[2])/Rs
        return r
    def root(self,t,*_): return np.array([((t-self.tref)%self.cycle)-self.p[0]])
    def handle_event(self,solver,_):
        self.i+=1; self.tref=round(solver.t,6)
        if self.i+1<len(self.beats):
            self.cycle=max(1e-6,self.beats[self.i+1]-self.beats[self.i])
        else: solver.terminate=True

def simulate(p,beats,tgrid):
    u0=np.array([8,8,8,265,0,0,0]); ud0=np.zeros(7)
    prob=Implicit_Problem(CardioODE(u0,ud0,p,beats).residual,u0,ud0,0.0)
    ida=IDA(prob); ida.atol=ida.rtol=1e-5; ida.maxord=5; ida.maxh=0.05
    t,y,_=ida.simulate(beats[-1]+0.1,ncp_list=tgrid)
    return np.asarray(t),np.asarray(y)

# ---------------------------------------------------------------------------
def slice_waveform(p):
    beats   = HRV(43)
    t_model = np.arange(0, beats[-1]+0.1, DT_MODEL)
    t_out,y = simulate(p, beats, t_model)

    t_slice = np.arange(WIN[0], WIN[1], DT_SLICE)

    Plv = np.interp(t_slice, t_out, y[:,0])
    pSA = np.interp(t_slice, t_out, y[:,1])
    pSV = np.interp(t_slice, t_out, y[:,2])
    Vlv = np.interp(t_slice, t_out, y[:,3])

    return np.vstack([Plv, pSA, pSV, Vlv]).T.ravel()       # (nt*4,)

def window_mean_from_wave(w):
    nt=len(w)//4
    return w.reshape(nt,4).mean(axis=0)

# ---------------- surrogate builders ----------------------------------------
def build_scalar_pce(X,y):
    sc=StandardScaler().fit(X); Xs=sc.transform(X)
    inp,out=ot.Sample(Xs.tolist()),ot.Sample(y.reshape(-1,1).tolist())
    marg=[ot.Uniform(Xs[:,i].min(),Xs[:,i].max()) for i in range(Xs.shape[1])]
    algo=ot.FunctionalChaosAlgorithm(
        inp,out,ot.ComposedDistribution(marg),
        ot.SequentialStrategy(
            ot.OrthogonalProductPolynomialFactory(
              [ot.LegendreFactory()]*Xs.shape[1],
              ot.LinearEnumerateFunction(Xs.shape[1])),
            ot.LinearEnumerateFunction(Xs.shape[1]).getBasisSizeFromTotalDegree(PCE_DEGREE)))
    algo.run(); meta=algo.getResult().getMetaModel()
    return sc, meta

# ---------------- plotting helpers -----------------------------------------
def parity_plot(y_true,y_pred,label,n,file):
    plt.figure(figsize=(4,4))
    plt.scatter(y_true,y_pred,alpha=.4)
    lo,hi=min(y_true.min(),y_pred.min()),max(y_true.max(),y_pred.max())
    plt.plot([lo,hi],[lo,hi],'k--')
    plt.title(f'Parity – {label}  (N={n})')
    plt.xlabel('Full'); plt.ylabel('Surr')
    plt.grid(True); plt.tight_layout(); plt.savefig(file,dpi=150); plt.close()

def heatmap(S1,ST,prefix,n,secs,file):
    M,N=len(param_names),len(out_labels)
    fig,(a1,a2)=plt.subplots(1,2,figsize=(12,5))
    for ax,mat,tit in zip((a1,a2),(S1,ST),('S₁','Sₜ')):
        im=ax.imshow(mat,vmin=0,vmax=1,cmap='viridis',aspect='auto')
        ax.set_xticks(range(N)); ax.set_xticklabels(out_labels,rotation=45)
        ax.set_yticks(range(M)); ax.set_yticklabels(param_names)
        ax.set_title(f'{prefix} {tit} (N={n}, t={secs:.2f}s)')
        for i in range(M):
            for j in range(N):
                ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',
                        color='white' if mat[i,j]>.5 else 'black')
    cax=fig.add_axes([0.93,0.15,0.02,0.7]); fig.colorbar(im,cax=cax)
    plt.tight_layout(rect=(0,0,0.9,1)); plt.savefig(file,dpi=150); plt.close()

# ========================================================================== #
def main():
    problem={'num_vars':len(param_names),'names':param_names,'bounds':param_bounds}

    # ---------- TRAINING ----------
    X_tr = saltelli.sample(problem,N_TRAIN,False)
    with Pool(N_CORES) as P:
        Y_tr = np.vstack(P.map(slice_waveform,X_tr))

    nt  = len(np.arange(WIN[0],WIN[1],DT_SLICE))
    svd = TruncatedSVD(
            n_components=np.searchsorted(
                np.cumsum(TruncatedSVD(min(30,nt*4-1)).fit(Y_tr)
                           .explained_variance_ratio_), ENERGY_KEEP)+1).fit(Y_tr)
    print(f'POD modes retained: {svd.n_components}')

    coeff_sur=[build_scalar_pce(X_tr, svd.transform(Y_tr)[:,i])
            for i in range(svd.n_components)]


    def surrogate_wave(p):
        coeff=np.array([meta(sc.transform(p.reshape(1,-1)))[0,0]
                        for sc,meta in coeff_sur]).reshape(1,-1)
        return svd.inverse_transform(coeff)[0]

    # window‑mean surrogates (for parity/Sobol)
    win_tr = np.apply_along_axis(window_mean_from_wave,1,Y_tr)
    mean_sur=[build_scalar_pce(X_tr,win_tr[:,i]) for i in range(4)]
    def surrogate_mean(p):
        return np.array([meta(sc.transform(p.reshape(1,-1)))[0,0]
                         for sc,meta in mean_sur])

    # ---------- VALIDATION ----------
    X_val = saltelli.sample(problem,N_VAL,True)

    tic=time.perf_counter()
    with Pool(N_CORES) as P:
        Y_full=np.vstack(P.map(slice_waveform,X_val))
    full_t=time.perf_counter()-tic
    means_full = np.apply_along_axis(window_mean_from_wave,1,Y_full)

    tic=time.perf_counter()
    means_sur  = np.vstack([surrogate_mean(p) for p in X_val])
    sur_t=time.perf_counter()-tic

    # ---------- FIGURES ----------
    for i,lab in enumerate(out_labels):
        parity_plot(means_full[:,i],means_sur[:,i],lab,len(X_val),
                    os.path.join(RESULT_DIR,f'parity_{lab}.png'))

    sob_full={i:sobol.analyze(problem,means_full[:,i],print_to_console=False)
              for i in range(4)}
    sob_sur ={i:sobol.analyze(problem,means_sur [:,i],print_to_console=False)
              for i in range(4)}
    S1_f=np.column_stack([sob_full[i]['S1'] for i in range(4)])
    ST_f=np.column_stack([sob_full[i]['ST'] for i in range(4)])
    S1_s=np.column_stack([sob_sur [i]['S1'] for i in range(4)])
    ST_s=np.column_stack([sob_sur [i]['ST'] for i in range(4)])

    heatmap(S1_f,ST_f,'Full',len(X_val),full_t,
            os.path.join(RESULT_DIR,'heatmap_full.png'))
    heatmap(S1_s,ST_s,'Surr',len(X_val),sur_t,
            os.path.join(RESULT_DIR,'heatmap_sur.png'))

    # ---------- DETAILED WAVEFORM ----------
    idx        = 0
    wave_full  = Y_full[idx].reshape(nt,4)
    wave_sur   = surrogate_wave(X_val[idx]).reshape(nt,4)
    t_slice    = np.arange(WIN[0],WIN[1],DT_SLICE)

    # PRESSURE panel (three traces)
    plt.figure(figsize=(10,6))
    for k,lbl,c in zip(range(3),['True pLV','True pSA','True pSV'],
                       ['tab:blue','tab:orange','tab:green']):
        plt.plot(t_slice,wave_full[:,k],label=lbl,color=c,lw=1.7)
        plt.plot(t_slice,wave_sur[:,k],'--',label='Surr '+lbl[5:],color=c,lw=1.2)
    plt.xlim(*WIN); plt.xlabel('Time (s)'); plt.ylabel('Pressure (mmHg)')
    plt.title(f'Pressures Over Time (deg={PCE_DEGREE})')
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR,'Figure_left_ventricle.png'),dpi=150)
    plt.close()

    # VOLUME panel (true vs surrogate)
    plt.figure(figsize=(10,6))
    plt.plot(t_slice,wave_full[:,3],label='True Vlv',lw=1.7)
    plt.plot(t_slice,wave_sur[:,3],'--',label='Surr Vlv',lw=1.2)
    plt.xlim(*WIN); plt.xlabel('Time (s)'); plt.ylabel('Volume (mL)')
    plt.title('Left‑ventricular Volume Over Time')
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR,'Figure_left_ventricle_2.png'),dpi=150)
    plt.close()

    print(f'Figures saved in “{RESULT_DIR}/”  |  full {full_t:.2f}s  sur {sur_t:.2f}s')

# ========================================================================== #
if __name__=='__main__':
    main()
