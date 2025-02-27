import numpy as np
import matplotlib.pyplot as plt
import openturns as ot
import time  # <-- Import the time module
from assimulo.problem import Implicit_Problem
from assimulo.solvers.sundials import IDA
from SALib.sample import saltelli
from SALib.analyze import sobol
from multiprocessing import Pool, cpu_count


class TwoChamberModel:

    def __init__(self, p, u0, udot0, t_τL):

        self.p = np.array(p)
        self.t_τL = t_τL
        self.τ = t_τL[0] if len(t_τL) > 0 else 1.0
        self.tr = 0.0
        self.n = 0.0
        self.u0 = u0
        self.udot0 = udot0
        self.Eshift_lv = 0.0
        self.Eshift_la = 0.92
        self.cycle_phase = "systole"
        
    def res(self, t, y, ydot):
        
        #application of chamber pressure, volume, and flow equations
        Plv, Pla, Vlv, Vla, Psas, Qsas, Psat, Qsat, Psvn, Qsvn, Qav, Qmv  = y
        (v0_lv, Emin_lv, Emax_lv, τes_lv, τed_lv, v0_la, Emin_la, Emax_la, τes_la, τed_la, CQ_AV, CQ_MV, Csas, Rsas, Lsas, Csat, Rsat, Lsat, Rsar, Rscp, Csvn, Rsvn) = self.p
        
        E_t_LV = self.shi_elastance_lv(t,self.τ,Emin_lv,Emax_lv,τes_lv,τed_lv,self.Eshift_lv)
        DE_t_LV = self.d_shi_elastance_lv(t,self.τ,Emin_lv,Emax_lv,τes_lv,τed_lv,self.Eshift_lv)
        E_t_LA = self.shi_elastance_la(t,self.τ,Emin_la,Emax_la,τes_la,τed_la,self.Eshift_la)
        DE_t_LA = self.d_shi_elastance_la(t,self.τ,Emin_la,Emax_la,τes_la,τed_la,self.Eshift_la)
        
        res = np.zeros(12)
        u = np.zeros(2)
        #chamber pressure
        #left ventricle
        res[0] = ydot[0] - ((Qmv - Qav) * E_t_LV + Plv / E_t_LV * DE_t_LV)
        #left atrium
        res[1] = ydot[1] - ((Qsvn - Qmv) * E_t_LA + DE_t_LA * (Vla - v0_la))
        #chamber volume
        #left ventricle
        res[2] = ydot[2] - (Qav - Qmv)
        #left atrium
        res[3] = ydot[3] - (Qsvn - Qmv)
        #ciruclation flow
        #sinus
        res[4] = ydot[4] - (Qsas - Qsat) / Csas
        res[5] = ydot[5] - (Psas - Psat - Rsas*Qsas) / Lsas
        #systemic artery
        res[6] = ydot[6] - (Qsas - Qsat) / Csat
        res[7] = ydot [7] - (Psas - Psvn - (Rsat + Rsar + Rscp) * Qsat) / Lsat
        #systemic vein
        res[8] = ydot[8] - (Qsat - Qsvn) / Csvn
        res[9] = ydot[9] - (ydot[8] - ydot[3]) / Rsvn
        #valve flow
        #aortic valve

        u[0] = CQ_AV * (np.sign(Plv - Psas) * np.abs(Plv - Psas)**0.5) if Plv >= Psas else 0.0
        u[1] = CQ_MV * (np.sign(Pla - Plv) * np.abs(Pla - Plv)**0.5) if Pla >= Plv else 0.0
            
        res[10] = Qav - u[0]
        res[11] = Qmv - u[1]
        
        return res


    def shi_elastance_lv(self, t, τ, Emin_lv, Emax_lv, τes_lv, τed_lv, Eshift_lv):
        
        ti_lv = ((t + (1 - self.Eshift_lv) * τ) % τ)
        
        if (ti_lv <= τes_lv):
            Ep_lv = (1 - np.cos(ti_lv / τes_lv * np.pi)) / 2.0 
        elif (ti_lv <= τed_lv):
            Ep_lv = (1 + np.cos((ti_lv - τes_lv) / (τed_lv - τes_lv) * np.pi)) / 2.0 
        else:
            Ep_lv = 0.0
        return Emin_lv + (Emax_lv - Emin_lv) * Ep_lv

    def d_shi_elastance_lv(self,t,τ,Emin_lv,Emax_lv,τes_lv,τed_lv,Eshift_lv):
        
        ti_lv = ((t + (1 - self.Eshift_lv) * τ) % τ)
        if (ti_lv <= τes_lv):
            dE_plv = (np.pi / τes_lv) * np.sin(ti_lv / τes_lv * np.pi) / 2.0
        elif (ti_lv <= τed_lv):
            dE_plv = - (np.pi / (τed_lv - τes_lv)) * np.sin((ti_lv - τes_lv) / (τed_lv - τes_lv) * np.pi) / 2.0
        else:
            dE_plv = 0.0
        return (Emax_lv - Emin_lv) * dE_plv

    def shi_elastance_la(self,t,τ,Emin_la,Emax_la,τes_la,τed_la,Eshift_la):
        
        ti_la = ((t + (1 - self.Eshift_la) * τ) % τ)
        
        if (ti_la <= τes_la):
            Ep_la = (1 - np.cos(ti_la / τes_la * np.pi)) / 2.0
        elif (ti_la <= τed_la):
            Ep_la = (1 + np.cos((ti_la - τes_la) / (τed_la - τes_la) * np.pi)) / 2.0
        else:
            Ep_la = 0.0
        return Emin_la + (Emax_la - Emin_la) * Ep_la

    def d_shi_elastance_la(self,t,τ,Emin_la,Emax_la,τes_la,τed_la,Eshift_la):
        
        ti_la = ((t + (1 - self.Eshift_la) * τ) % τ)
        if (ti_la <= τes_la):
            dE_pla = (np.pi / τes_la) * np.sin(ti_la / τes_la * np.pi) / 2.0
        elif (ti_la <= τed_la):
            dE_pla = - (np.pi / (τed_la - τes_la)) * np.sin((ti_la - τes_la) / (τed_la - τes_la) * np.pi) / 2.0
        else:
            dE_pla = 0.0
        return (Emax_la - Emin_la) * dE_pla
    
    def handle_event(self, solver, event_info):
        self.n += 1
        self.tr = round(solver.t, 6)
        if self.n + 1 < len(self.t_τL):
            self.τ = max(1e-6, self.t_τL[self.n + 1] - self.t_τL[self.n])
    
    # def root(self, t, y, ydot):
    #     t_i = (t - self.tr) % self.τ
    #     return np.array([t_i - self.τ])  # event when t_i = τ_es
    
    def root(self, t, y, ydot):
        return np.array([(t - self.tr) % self.τ - self.τ])  # Ensures fewer evaluations

    
    
def HRV(end_time):
        """
        Generate heart rate variability (HRV) times until end_time plus a buffer.
        """
        t_τL = []
        t_current = 0.0
        while t_current < end_time + 1.0:
            # τ = np.random.uniform(0.9, 1.0)
            t_current += 1.0
            t_τL.append(t_current)
        return np.array(t_τL)
    
###############################################################################
# #             1) BASELINE RUN + PLOTTING (MODEL GRAPH)                       ##
###############################################################################

def simulate_single_trajectory(args):

    i, param_values, t_τL, x = args
    p = param_values[i, :]  

    u0 = np.array([1.0, 1.0, 5.0, 4.0, 100.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    udot0 = np.zeros_like(u0)
    model = TwoChamberModel(p, u0, udot0, t_τL)
    
    # Define the problem
    problem = Implicit_Problem(model.res, u0, udot0, 0.0)
    problem.root = model.root
    problem.handle_event = model.handle_event
    problem.nroot = 1

    algvar = np.ones(12, dtype=int)
    algvar[10] = 0  # Qav is algebraic
    algvar[11] = 0  # Qmv is algebraic
    problem.algvar = algvar
    
    solver = IDA(problem)        
    solver.report_continuously = True
    solver.display_progress = True
    solver.atol = 1e-2
    solver.rtol = 1e-2
    solver.maxord = 5
    solver.maxh = 0.05
    solver.maxsteps = 20000
    
    # tfinal = t_τL[-1] + 0.1
    tfinal = x[-1]
    # plot_saveat = np.arange(0, tfinal + 0.002, 0.002)
    try:

        t_out, y_out, yd_out = solver.simulate(tfinal, ncp_list=x)

        # Parse out the states of interest
        # y_out is shape (len(t_out), 12). Indices:
        #   pLV = y_out[:, 0], pSA = y_out[:, 4], Vlv = y_out[:, 2]
        pLV = y_out[:, 0]
        pSA = y_out[:, 4]
        Vlv = y_out[:, 2]

        # Return them as (pLV, pSA, Vlv)
        return (pLV, pSA, Vlv)
    except Exception as e:
        print(f"[simulate_single_trajectory] Solver failed for trajectory {i} with error: {e}")
        return None

def simulate_ensemble(param_values, t_τL, x):

    num_time_points = len(x)
    num_trajectories = param_values.shape[0]
    args_list = [(i, param_values, t_τL, x) for i in range(num_trajectories)]

    # Run in parallel
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(simulate_single_trajectory, args_list)

    # Filter out None results
    valid_res = [(i, res) for i, res in enumerate(results) if res is not None]
    if not valid_res:
        raise RuntimeError("No valid results for ensemble simulation.")

    # Initialize output
    # outputs = np.zeros((3 * num_time_points, len(valid_res)))

    # Fill outputs
    # for idx, (i, triple) in enumerate(valid_res):
    #     pLV, pSA, Vlv = triple
    #     outputs[0:num_time_points, idx]                    = pLV
    #     outputs[num_time_points:2*num_time_points, idx]    = pSA
    #     outputs[2*num_time_points:3*num_time_points, idx]  = Vlv

    # return outputs, [i for i, _ in valid_res]
    output_list = []
    valid_indices = []
    for idx, (i, (t_out, y_out, yd_out)) in enumerate(valid_res):
        output_list.append((t_out, y_out, yd_out))
        valid_indices.append(i)

    return output_list, valid_indices


def baseline_run_and_plot():
    
    """
    Run the model with default parameters and plot the results.
    """
    # Define the model parameters
    # v0_lv, Emin_lv, Emax_lv, τes_lv, τed_lv, v0_la, Emin_la, Emax_la, τes_la, τed_la, CQ_AV, CQ_MV, Csas, Rsas, Lsas, Csat, Rsat, Lsat, Rsar, Rscp, Csvn, Rsvn = self.p
    p = [
        5.0,    # v0_lv
        0.1,    # Emin_lv
        2.5,    # Emax_lv
        0.3,    # τes_lv
        0.45,   # τed_lv
        4.0,    # v0_la
        0.15,   # Emin_la
        0.25,   # Emax_la
        0.045,  # τes_la
        0.09,   # τed_la
        350.0,  # CQ_AV
        400.0,  # CQ_MV
        0.08,   # Csas
        0.003,  # Rsas
        6.2e-5, # Lsas
        1.6,    # Csat
        0.05,   # Rsat
        0.0017, # Lsat
        0.5,    # Rsar
        0.52,   # Rscp
        20.5,   # Csvn
        0.075   # Rsvn
    ]

    simulation_end_time = 5.0
    t_τL = HRV(simulation_end_time)
    ##u0 = [p0_lv, p0_la, v0_lv, v0_la, pt0sas, qt0sas, pt0sat, qt0sat, pt0svn, qt0svn, 0.0, 0.0]

    x = np.linspace(0, simulation_end_time, 100)  # 100 points
    param_values = np.array([p])
    outputs, valid_idx = simulate_ensemble(param_values, t_τL, x)

    pLV = outputs[0:len(x), 0]
    pSA = outputs[len(x):2*len(x), 0]
    Vlv = outputs[2*len(x):, 0]

    t_out, y_out, yd_out = all_solutions[0]



    # u0 = np.array([1.0, 1.0, 5.0, 4.0, 100.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # udot0 = np.zeros_like(u0)
    # model = TwoChamberModel(p, u0, udot0, t_τL)
    
    # Define the problem
    # problem = Implicit_Problem(model.res, u0, udot0, 0.0)
    # problem.root = model.root
    # problem.handle_event = model.handle_event
    # problem.nroot = 0
    
    # solver = IDA(problem)        
    # solver.report_continuously = True
    # solver.display_progress = True
    # solver.atol = 1e-2
    # solver.rtol = 1e-2
    # solver.maxord = 5
    # solver.maxh = 2.0
    # solver.maxsteps = 1000
    
    # tfinal = t_τL[-1] + 0.1
    # # plot_saveat = np.arange(0, tfinal + 0.002, 0.002)
    # plot_saveat = np.arange(0, tfinal + 0.01, 0.5)  # Every 10ms instead of 2ms

    
    # t, y, yd = solver.simulate(tfinal, ncp_list=plot_saveat)
    
    #Plv, Pla, Vlv, Vla, Psas, Qsas, Psat, Qsat, Psvn, Qsvn, Qav, Qmv  = y
    
    Plv = y[:, 0]
    Pla = y[:, 1]
    Vlv = y[:, 2]
    Vla = y[:, 3]
    Psas = y[:, 4]
    Qsas = y[:, 5]
    Psat = y[:, 6]
    Qsat = y[:, 7]
    Psvn = y[:, 8]
    Qsvn = y[:, 9]
    Qav = y[:, 10]
    Qmv = y[:, 11]
    
    
    plt.figure(figsize=(10, 6))
    #plot pressures
    plt.plot(t, Plv, label='Plv')
    plt.plot(t, Pla, label='Pla')
    plt.plot(t, Psas, label='Psas')
    plt.plot(t, Psat, label='Psat')
    plt.plot(t, Psvn, label='Psvn')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (mmHg)')
    plt.legend()
    plt.grid(True)
    plt.xlim(3, 5)
    plt.show()
    
    #plot volumes
    plt.figure(figsize=(10, 6))
    plt.plot(t, Vlv, label='Vlv')
    plt.plot(t, Vla, label='Vla')
    plt.xlabel('Time (s)')
    plt.ylabel('Volume (ml)')
    plt.legend()
    plt.grid(True)
    plt.xlim(3, 5)
    plt.show()
    
    #plot flows
    plt.figure(figsize=(10, 6))
    plt.plot(t, Qsas, label='Qsas')
    plt.plot(t, Qsat, label='Qsat')
    plt.plot(t, Qsvn, label='Qsvn')
    plt.plot(t, Qav, label='Qav')
    plt.plot(t, Qmv, label='Qmv')
    plt.xlabel('Time (s)')
    plt.ylabel('Flow (ml/s)')
    plt.legend()
    plt.grid(True)
    plt.xlim(3, 5)
    plt.show()
    
def main():
    print("Running baseline simulation...")
    baseline_run_and_plot()
    stats = solver.get_statistics()
    print (stats)
if __name__ == "__main__":
    main()
        


    

    
    
    
    

    
    



    
    
