import numpy as np
import matplotlib.pyplot as plt
from assimulo.problem import Implicit_Problem
from assimulo.solvers.sundials import IDA

class TwoChamberModel:

    def __init__(self, p, u0, udot0, τ=1.0):
        self.p     = np.array(p)
        self.u0    = u0
        self.udot0 = udot0
        self.τ     = τ

        # Shifts for elastance timing
        self.Eshift_lv = 0.0
        self.Eshift_la = 0.92



    def valve_flow(self, CQ, p_up, p_down, smoothing=0.1):
        """
        Smooth valve flow function with exponent clipping to avoid overflow.
        """
        deltaP = p_up - p_down

        # Clip the exponent to some safe range, e.g. [-30, 30]
        exponent = np.clip(-deltaP / smoothing, -30.0, 30.0)

        opening_fraction = 1.0 / (1.0 + np.exp(exponent))

        flow = CQ * np.sqrt(np.maximum(deltaP, 0.0)) * opening_fraction
        return flow


    def res(self, t, y, ydot):
        Plv, Pla, Vlv, Vla, Psas, Qsas, Psat, Qsat, Psvn, Qsvn, Qav, Qmv = y
        (v0_lv, Emin_lv, Emax_lv, τes_lv, τed_lv,
         v0_la, Emin_la, Emax_la, τes_la, τed_la,
         CQ_AV, CQ_MV, Csas, Rsas, Lsas, Csat, Rsat,
         Lsat, Rsar, Rscp, Csvn, Rsvn) = self.p

        E_t_LV   = self.shi_elastance_lv(t)
        DE_t_LV  = self.d_shi_elastance_lv(t)
        E_t_LA   = self.shi_elastance_la(t)
        DE_t_LA  = self.d_shi_elastance_la(t)

        res = np.zeros(12)

        # 1) dPlv/dt
        res[0] = ydot[0] - ((Qmv - Qav)*E_t_LV + (Plv/E_t_LV)*DE_t_LV)

        # 2) dPla/dt
        res[1] = ydot[1] - ((Qsvn - Qmv)*E_t_LA + (Pla/E_t_LA)*DE_t_LA)

        # 3) dVlv/dt
        res[2] = ydot[2] - (Qmv - Qav)

        # 4) dVla/dt
        res[3] = ydot[3] - (Qsvn - Qmv)

        # 5) dPsas/dt
        res[4] = ydot[4] - ((Qav - Qsas)/Csas)

        # 6) dQsas/dt
        res[5] = ydot[5] - ((Psas - Psat - Rsas*Qsas)/Lsas)

        # 7) dPsat/dt
        res[6] = ydot[6] - ((Qsas - Qsat)/Csat)

        # 8) dQsat/dt
        res[7] = ydot[7] - ((Psat - Psvn - (Rsat+Rsar+Rscp)*Qsat)/Lsat)

        # 9) dPsvn/dt
        res[8] = ydot[8] - ((Qsat - Qsvn)/Csvn)

        # 10) dQsvn/dt
        res[9] = ydot[9] - ((ydot[8] - ydot[3])/Rsvn)

        # 11) Qav
        # Qav_expr = 0.0
        # if Plv >= Psas:
        #     Qav_expr = CQ_AV * np.sqrt(Plv - Psas)
        # res[10] = Qav - Qav_expr

        # # 12) Qmv
        # Qmv_expr = 0.0
        # if Pla >= Plv:
        #     Qmv_expr = CQ_MV * np.sqrt(Pla - Plv)
        # res[11] = Qmv - Qmv_expr
        # 11) Qav
        Qav_expr = self.valve_flow(CQ_AV, Plv, Psas, smoothing=0.1)
        res[10] = Qav - Qav_expr

        # 12) Qmv
        Qmv_expr = self.valve_flow(CQ_MV, Pla, Plv, smoothing=0.1)
        res[11] = Qmv - Qmv_expr

        return res

    def shi_elastance_lv(self, t):
        """
        Hard-code the piecewise function using self.τ, self.p, self.Eshift_lv
        """
        τ = self.τ
        v0_lv, Emin_lv, Emax_lv, τes_lv, τed_lv = self.p[:5]
        ti_lv = (t + (1.0 - self.Eshift_lv)*τ) % τ

        if ti_lv <= τes_lv:
            Ep_lv = 0.5*(1.0 - np.cos(np.pi*ti_lv/τes_lv))
        elif ti_lv <= τed_lv:
            r = (ti_lv - τes_lv)/(τed_lv - τes_lv)
            Ep_lv = 0.5*(1.0 + np.cos(np.pi*r))
        else:
            Ep_lv = 0.0
        return Emin_lv + (Emax_lv - Emin_lv)*Ep_lv

    def d_shi_elastance_lv(self, t):
        τ = self.τ
        v0_lv, Emin_lv, Emax_lv, τes_lv, τed_lv = self.p[:5]
        ti_lv = (t + (1.0 - self.Eshift_lv)*τ) % τ

        if ti_lv <= τes_lv:
            dE_p_lv = (np.pi/τes_lv)*np.sin(np.pi*ti_lv/τes_lv)/2.0
        elif ti_lv <= τed_lv:
            r = (ti_lv - τes_lv)/(τed_lv - τes_lv)
            dE_p_lv = - (np.pi/(τed_lv - τes_lv))*np.sin(np.pi*r)/2.0
        else:
            dE_p_lv = 0.0
        return (Emax_lv - Emin_lv)*dE_p_lv

    def shi_elastance_la(self, t):
        τ = self.τ
        # v0_la= p[5], Emin_la= p[6], Emax_la= p[7], τes_la= p[8], τed_la= p[9], ...
        Emin_la, Emax_la, τes_la, τed_la = self.p[6:10]
        ti_la = (t + (1.0 - self.Eshift_la)*τ) % τ

        if ti_la <= τes_la:
            Ep_la = 0.5*(1.0 - np.cos(np.pi*ti_la/τes_la))
        elif ti_la <= τed_la:
            r = (ti_la - τes_la)/(τed_la - τes_la)
            Ep_la = 0.5*(1.0 + np.cos(np.pi*r))
        else:
            Ep_la = 0.0
        return Emin_la + (Emax_la - Emin_la)*Ep_la

    def d_shi_elastance_la(self, t):
        τ = self.τ
        Emin_la, Emax_la, τes_la, τed_la = self.p[6:10]
        ti_la = (t + (1.0 - self.Eshift_la)*τ) % τ

        if ti_la <= τes_la:
            dE_pla = (np.pi/τes_la)*np.sin(np.pi*ti_la/τes_la)/2.0
        elif ti_la <= τed_la:
            r = (ti_la - τes_la)/(τed_la - τes_la)
            dE_pla = - (np.pi/(τed_la - τes_la))*np.sin(np.pi*r)/2.0
        else:
            dE_pla = 0.0
        return (Emax_la - Emin_la)*dE_pla

def run_minimal():
    p = [
        5.0,  # v0_lv
        0.1,  # Emin_lv
        2.5,  # Emax_lv
        0.3,  # τes_lv
        0.45, # τed_lv
        4.0,  # v0_la
        0.15, # Emin_la
        0.25, # Emax_la
        0.045,# τes_la
        0.09, # τed_la
        350.0,# CQ_AV
        400.0,# CQ_MV
        0.08, # Csas
        0.003,# Rsas
        6.2e-5,# Lsas
        1.6,  # Csat
        0.05, # Rsat
        0.0017,# Lsat
        0.5,  # Rsar
        0.52, # Rscp
        20.5, # Csvn
        0.075 # Rsvn
    ]

    u0 = np.array([1.0,1.0,5.0,4.0,  100.0,0.0,100.0,0.0,  0.0,0.0, 0.0,0.0])
    udot0 = np.zeros_like(u0)

    model = TwoChamberModel(p, u0, udot0, τ=1.0)

    problem = Implicit_Problem(model.res, u0, udot0, t0=0.0)
    # Mark Qav, Qmv as algebraic
    algvar = np.ones(12, dtype=int)
    algvar[10] = 0
    algvar[11] = 0
    problem.algvar = algvar

    solver = IDA(problem)
    solver.make_consistent('IDA_YA_YDP_INIT')

    # 1) Loose Tolerances
    solver.atol = 1e-5
    solver.rtol = 1e-5

    # 2) Big maxsteps, smaller step size
    solver.maxsteps = 1000000
    solver.maxh = 0.001
    solver.maxord = 2
    solver.display_progress = True

    t_final = 35.0
    save_times = np.linspace(0, t_final, 5000)

    t_out, y, yd_out = solver.simulate(t_final, ncp_list=save_times)

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

    # import matplotlib.pyplot as plt
    # plt.plot(t_out, Plv, label='Plv')
    # plt.plot(t_out, Pla, label='Pla')
    # plt.plot(t_out, Psas,label='Psas')
    # plt.legend()
    # plt.show()

        #plot pressures
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

if __name__=="__main__":
    run_minimal()
