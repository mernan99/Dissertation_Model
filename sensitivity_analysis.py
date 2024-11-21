import numpy as np
from left import CardiovascularModel
from left import Valve
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import h5py

def define_problem(τ_es=0.3, τ_ep=0.45, Rmv=0.06, Zao=0.033, Rs=1.11, Csa=1.13, Csv=11.0, E_max=1.5, E_min=0.03):

    problem = { 
    'num_vars': 9,
    'names': ['τ_es', 'τ_ep', 'Rmv', 'Zao', 'Rs', 'Csa', 'Csv', 'E_max', 'E_min'],
    'bounds': [[0.9*τ_es, 1.1*τ_es],
               [0.9*τ_ep, 1.1*τ_ep],
               [0.9*Rmv, 1.1*Rmv],
               [0.9*Zao, 1.1*Zao],
               [0.9*Rs, 1.1*Rs],
               [0.9*Csa, 1.1*Csa],
               [0.9*Csv, 1.1*Csv],
               [0.9*E_max, 1.1*E_max],
               [0.9*E_min, 1.1*E_min]]
    }

    return problem

param_values = saltelli.sample(problem,1000)

def run_simulation(params):

    model = CardiovascularModel(u0, udot0, patams, t_τL)

    solver = IDA(model)

    solver.set_initial_condition(u0, udot0)

    t,y = solver.simulate(10.0)

    return y[:, 0]

Y = np.zeros([param_values.shape[0]])
for i, params in enumerate(param_values):
    Y[i] = run_simulation(params)

Si = sobol.analyze(problem, Y)

print(Si['S1'])  # First-order sensitivity indices
print(Si['ST'])  # Total-order sensitivity indices
