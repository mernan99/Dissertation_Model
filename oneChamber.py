# cardio_model.py
import numpy as np
import matplotlib.pyplot as plt
from assimulo.problem import Implicit_Problem
from assimulo.solvers.sundials import IDA


def Valve(R, deltaP):
    R = max(R, 1e-6)
    if deltaP > 0.0:
        return deltaP / R
    else:
        return 0.0

def HRV(end_time):
    """
    Create random cycles each >= 0.4 s, ensuring tau_es=0.3 can be reached.
    """
    t_τL = []
    t_current = 0.0
    while t_current < end_time + 1.0:
        τ = np.random.uniform(0.4, 1.1)
        t_current += τ
        t_τL.append(t_current)
    return np.array(t_τL)

class CardiovascularModel:
    """
    Basic model with Shi elastance, triggers an event when time_in_cycle == tau_es.
    """
    def __init__(self, u0, udot0, p, t_τL):
        self.p = np.array(p)
        self.t_τL = t_τL
        self.τ = t_τL[0]
        self.tr = 0.0
        self.n = 0

        self.u0 = u0
        self.udot0 = udot0

        self.event_times = [0.0]
        self.cycle_phase = "systole"

    def res(self, t, y, ydot):
        pLV, psa, psv, Vlv, Qav, Qmv, Qs = y
        (τ_es, τ_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min) = self.p

        E_t = self.ShiElastance(t)
        DE_t = self.DShiElastance(t)

        res = np.zeros_like(y)
        res[0] = ydot[0] - ((Qmv - Qav)*E_t + pLV/E_t*DE_t)
        res[1] = ydot[1] - (Qav - Qs)/Csa
        res[2] = ydot[2] - (Qs - Qmv)/Csv
        res[3] = ydot[3] - (Qmv - Qav)
        deltaP_av = pLV - psa
        res[4] = Qav - Valve(Zao, deltaP_av)
        deltaP_mv = psv - pLV
        res[5] = Qmv - Valve(Rmv, deltaP_mv)
        res[6] = ydot[6] - (ydot[1] - ydot[2]) / Rs
        return res

    def ShiElastance(self, t):
        (τ_es, τ_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min) = self.p
        t_i = (t - self.tr) % self.τ
        if t_i <= τ_es:
            E_p = 0.5*(1 - np.cos(np.pi*(t_i/τ_es)))
        elif t_i <= τ_ep:
            E_p = 0.5*(1 + np.cos(np.pi*((t_i - τ_es)/(τ_ep - τ_es))))
        else:
            E_p = 0.0
        return E_min + (E_max - E_min)*E_p

    def DShiElastance(self, t):
        (τ_es, τ_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min) = self.p
        t_i = (t - self.tr) % self.τ
        if t_i <= τ_es:
            dE_p = 0.5*(np.pi/τ_es)*np.sin(np.pi*t_i/τ_es)
        elif t_i <= τ_ep:
            dE_p = -0.5*(np.pi/(τ_ep - τ_es))*np.sin(np.pi*(t_i - τ_es)/(τ_ep - τ_es))
        else:
            dE_p = 0.0
        return (E_max - E_min)*dE_p

    def handle_event(self, solver, event_info):
        self.n += 1
        self.tr = round(solver.t, 6)
        self.event_times.append(self.tr)

        if self.cycle_phase == "systole":
            self.cycle_phase = "diastole"
        else:
            self.cycle_phase = "systole"

        if self.n+1 < len(self.t_τL):
            self.τ = max(1e-6, self.t_τL[self.n+1] - self.t_τL[self.n])

    def root(self, t, y, ydot):
        τ_es = self.p[0]
        t_i = (t - self.tr) % self.τ
        return np.array([t_i - τ_es])



def baseline_run_and_plot():
    # Define baseline parameters
    p = [0.3, 0.45, 0.06, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]
    simulation_end_time = 43
    t_τL = HRV(simulation_end_time)

    u0 = np.array([8.0, 8.0, 8.0, 265.0, 0.0, 0.0, 0.0])
    udot0 = np.zeros(7)

    model = CardiovascularModel(u0, udot0, p, t_τL)
    problem = Implicit_Problem(model.res, u0, udot0, 0.0)
    problem.root = model.root
    problem.handle_event = model.handle_event
    problem.nroots = 1

    solver = IDA(problem)
    solver.report_continuously = True
    solver.atol = 1e-6
    solver.rtol = 1e-6
    solver.maxord = 5
    solver.maxh = 0.05
    solver.maxsteps = 20000

    tfinal = t_τL[-1] + 0.1
    plot_saveat = np.arange(0, tfinal + 0.002, 0.002)

    # Run solver
    t_values, y, yd = solver.simulate(tfinal, ncp_list=plot_saveat)

    # Convert t_values to NumPy array to avoid AttributeError
    t_values = np.array(t_values)

    # Extract relevant training data
    X_train = np.tile(p, (len(t_values), 1))  # Ensure correct shape
    Y_train_list = [y[:, 0], y[:, 1], y[:, 3]]  # pLV, psa, Vlv

    print("Returning X_train shape:", X_train.shape)
    print("Returning Y_train_list lengths:", [len(y_data) for y_data in Y_train_list])
    print("Returning time values shape:", t_values.shape)  # Should now work

    pLV = y[:, 0]
    psa = y[:, 1]
    psv = y[:, 2]
    Vlv = y[:, 3]
    Qav = y[:, 4]
    Qmv = y[:, 5]
    Qs  = y[:, 6]

    # Plot pressures between 33 and 35s
    plt.figure(figsize=(12, 6))
    plt.plot(t_values, pLV, label='P_LV')
    plt.plot(t_values, psa, label='P_SA')
    plt.plot(t_values, psv, label='P_SV')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure')
    plt.title('Pressures Over Time')
    plt.legend()
    plt.grid(True)
    plt.xlim(33, 35)
    plt.show()

    # Plot ventr. volume between 33 and 35s
    plt.figure(figsize=(12, 6))
    plt.plot(t_values, Vlv, label='V_LV')
    plt.xlabel('Time (s)')
    plt.ylabel('Volume')
    plt.title('Left Ventricular Volume Over Time')
    plt.legend()
    plt.grid(True)
    plt.xlim(33, 35)
    plt.show()

    return X_train, Y_train_list, t_values  # Ensure three values are returned


if __name__ == "__main__":
    baseline_run_and_plot()

