import numpy as np
import matplotlib.pyplot as plt
from assimulo.problem import Implicit_Problem
from assimulo.solvers.sundials import IDA

def elastance_LV(t, Emin_lv, Emax_lv, τ_es_lv, τ_ed_lv):

    T_cycle = 1.0  
    t_in_cycle = (t % T_cycle)

    if t_in_cycle <= τ_es_lv:
        # systolic rise
        E_p = 0.5 * (1.0 - np.cos(np.pi * (t_in_cycle / τ_es_lv)))
        dE_p = 0.5 * (np.pi / τ_es_lv) * np.sin(np.pi * (t_in_cycle / τ_es_lv))
    elif t_in_cycle <= τ_ed_lv:
        # early diastolic fall
        x = (t_in_cycle - τ_es_lv) / (τ_ed_lv - τ_es_lv)
        E_p = 0.5 * (1.0 + np.cos(np.pi * x))
        dE_p = 0.5 * (np.pi / (τ_ed_lv - τ_es_lv)) * -np.sin(np.pi * x)
    else:
        E_p = 0.0
        dE_p = 0.0

    E = Emin_lv + (Emax_lv - Emin_lv)*E_p
    dE = (Emax_lv - Emin_lv)*dE_p
    return E, dE


def elastance_LA(t, Emin_la, Emax_la, τ_es_la, τ_ed_la, Eshift_la):

    T_cycle = 1.0
    # shift the time as in: tᵢ_la = rem(t + (1 - Eshift_la)*τ, τ)
    t_shifted = (t + (1.0 - Eshift_la)*T_cycle) % T_cycle

    if t_shifted <= τ_es_la:
        E_p  = 0.5*(1.0 - np.cos(np.pi * (t_shifted / τ_es_la)))
        dE_p = 0.5*(np.pi / τ_es_la)*np.sin(np.pi * (t_shifted / τ_es_la))
    elif t_shifted <= τ_ed_la:
        x    = (t_shifted - τ_es_la)/(τ_ed_la - τ_es_la)
        E_p  = 0.5*(1.0 + np.cos(np.pi * x))
        dE_p = 0.5*(np.pi/(τ_ed_la - τ_es_la)) * -np.sin(np.pi * x)
    else:
        E_p  = 0.0
        dE_p = 0.0

    E  = Emin_la + (Emax_la - Emin_la)*E_p
    dE = (Emax_la - Emin_la)*dE_p
    return E, dE

class ShiModel:


    def __init__(self, params):

        self.v0_lv, self.Emin_lv, self.Emax_lv, self.tau_es_lv, self.tau_ed_lv, \
        self.v0_la, self.Emin_la, self.Emax_la, self.tau_es_la, self.tau_ed_la, \
        self.CQ_AV,  self.CQ_MV,  self.Csas,    self.Rsas,   self.Lsas, \
        self.Csat,   self.Rsat,   self.Lsat,   self.Rsar,   self.Rscp, \
        self.Csvn,   self.Rsvn   = params

        # Shift for LA was 0.92 in your snippet
        self.Eshift_la = 0.92

    def res(self, t, y, ydot):


        # Unpack states
        Plv   = y[0]
        Pla   = y[1]
        Vlv   = y[2]
        Vla   = y[3]
        Psas  = y[4]
        Qsas  = y[5]
        Psat  = y[6]
        Qsat  = y[7]
        Psvn  = y[8]
        Qsvn  = y[9]
        Qav   = y[10]  # algebraic
        Qmv   = y[11]  # algebraic

        # Unpack derivatives
        dPlv  = ydot[0]
        dPla  = ydot[1]
        dVlv  = ydot[2]
        dVla  = ydot[3]
        dPsas = ydot[4]
        dQsas = ydot[5]
        dPsat = ydot[6]
        dQsat = ydot[7]
        dPsvn = ydot[8]
        dQsvn = ydot[9]

        # Time-varying elastances
        E_LV, dE_LV = elastance_LV(
            t, self.Emin_lv, self.Emax_lv,
            self.tau_es_lv, self.tau_ed_lv
        )
        E_LA, dE_LA = elastance_LA(
            t, self.Emin_la, self.Emax_la,
            self.tau_es_la, self.tau_ed_la, self.Eshift_la
        )

        # if Plv > Psas:
        #     Qav_calc = self.CQ_AV * np.sqrt(Plv - Psas)
        # else:
        #     Qav_calc = 0.0

        # if Pla > Plv:
        #     Qmv_calc = self.CQ_MV * np.sqrt(Pla - Plv)
        # else:
        #     Qmv_calc = 0.0
        # Valve flows (algebraic, linear resistive model)
        Qav_calc = max((Plv - Psas) / self.CQ_AV, 0.0)  # CQ_AV becomes Zao (aortic valve resistance)
        Qmv_calc = max((Pla - Plv) / self.CQ_MV, 0.0)   # CQ_MV becomes Rmv (mitral valve resistance)


        # Build the residual vector
        res = np.zeros(12)

        res[0] = dPlv - ( (Qmv - Qav)*E_LV + dE_LV*(Vlv - self.v0_lv) )

        res[1] = dPla - ( (Qsvn - Qmv)*E_LA + dE_LA*(Vla - self.v0_la) )

        res[2] = dVlv - ( Qmv - Qav )

        res[3] = dVla - ( Qsvn - Qmv )

        res[4] = dPsas - ( (Qav - Qsas)/self.Csas )

        res[5] = dQsas - ( (Psas - Psat - self.Rsas*Qsas)/self.Lsas )

        res[6] = dPsat - ( (Qsas - Qsat)/self.Csat )

        res[7] = dQsat - ( (Psat - Psvn - (self.Rsat + self.Rsar + self.Rscp)*Qsat)/self.Lsat )

        res[8] = dPsvn - ( (Qsat - Qsvn)/self.Csvn )

        res[9] = dQsvn - ( (dPsvn - dVla)/self.Rsvn )

        res[10] = Qav - Qav_calc

        res[11] = Qmv - Qmv_calc

        return res

def baseline_run_and_plot():

    params = [
        10.0,     # v0_lv 
        0.1,      # Emin_lv
        2.5,      # Emax_lv
        0.3,      # τes_lv
        0.45,     # τed_lv
        10.0,     # v0_la 
        0.15,     # Emin_la
        0.25,     # Emax_la
        0.045,    # τes_la
        0.09,     # τed_la
        0.033,    # Zao (aortic valve resistance)
        0.06,     # Rmv (mitral valve resistance)
        0.08,     # Csas
        0.06,     # Rsas
        6.2e-5,   # Lsas
        1.6,      # Csat
        0.05,     # Rsat
        0.0017,   # Lsat
        0.5,      # Rsar
        0.52,     # Rscp
        20.5,     # Csvn
        0.075     # Rsvn
    ]

    y0 = np.array([
        10.0,    # Plv
        8.0,     # Pla
        150.0,   # Vlv
        40.0,    # Vla
        100.0,    # Psas
        0.0,     # Qsas
        100.0,    # Psat
        0.0,     # Qsat
        5.0,     # Psvn
        0.0,     # Qsvn
        0.0,     # Qav
        0.0      # Qmv
    ])


    ydot0 = np.zeros_like(y0)

    model = ShiModel(params)

    problem = Implicit_Problem(model.res, y0, ydot0, t0=0.0)
    problem.nroots = 0  

    algvar = np.ones(12, dtype=int)
    algvar[10] = 0
    algvar[11] = 0
    problem.algvar = algvar


    solver = IDA(problem)
    solver.make_consistent('IDA_YA_YDP_INIT')
    solver.report_continuously = True
    solver.atol = 1e-5
    solver.rtol = 1e-5
    solver.maxsteps = 5000
    solver.maxh = 0.001
    solver.maxord = 2
    solver.display_progress = True

    tfinal = 35.0
    save_times = np.linspace(0, tfinal, 5000)
    t_out, y, yd_vals = solver.simulate(tfinal, ncp_list=save_times)

    t_out = np.array(t_out)
    y = np.array(y)

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

    # # Plot: 
    # #  - left ventricle pressure (Plv = y[:,0])
    # #  - left atrium pressure (Pla = y[:,1])
    # #  - LV volume (Vlv = y[:,2])
    # fig1 = plt.figure()
    # plt.title("Left Ventricle Pressure")
    # plt.plot(t_out, y[:, 0])
    # plt.xlabel("time (s)")
    # plt.ylabel("Plv (mmHg)")

    # fig2 = plt.figure()
    # plt.title("Left Atrium Pressure")
    # plt.plot(t_out, y[:, 1])
    # plt.xlabel("time (s)")
    # plt.ylabel("Pla (mmHg)")

    # fig3 = plt.figure()
    # plt.title("Left Ventricle Volume")
    # plt.plot(t_out, y[:, 2])
    # plt.xlabel("time (s)")
    # plt.ylabel("Vlv (mL)")

    # plt.show()

    plt.plot(t_out, Plv, label='Plv')
    plt.plot(t_out, Pla, label='Pla')
    plt.plot(t_out, Psas, label='Psas')
    plt.plot(t_out, Psat, label='Psat')
    plt.plot(t_out, Psvn, label='Psvn')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (mmHg)')
    plt.legend()
    plt.grid(True)
    plt.xlim(33, 35)
    plt.show()
    
    #plot volumes
    plt.figure(figsize=(10, 6))
    plt.plot(t_out, Vlv, label='Vlv')
    plt.plot(t_out, Vla, label='Vla')
    plt.xlabel('Time (s)')
    plt.ylabel('Volume (ml)')
    plt.legend()
    plt.grid(True)
    plt.xlim(33, 35)
    plt.show()
    
    #plot flows
    plt.figure(figsize=(10, 6))
    plt.plot(t_out, Qsas, label='Qsas')
    plt.plot(t_out, Qsat, label='Qsat')
    plt.plot(t_out, Qsvn, label='Qsvn')
    plt.plot(t_out, Qav, label='Qav')
    plt.plot(t_out, Qmv, label='Qmv')
    plt.xlabel('Time (s)')
    plt.ylabel('Flow (ml/s)')
    plt.legend()
    plt.grid(True)
    plt.xlim(33, 35)
    plt.show()

    return t_out, y, yd_vals



if __name__ == "__main__":
    t, y, yd = baseline_run_and_plot()
