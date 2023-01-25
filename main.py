from turbo import Turbo1
import numpy as np
import pandas as pd
import re
import torch
import math
import matplotlib
import matplotlib.pyplot as plt

def pyflatten(l):
    return [item for sublist in l for item in sublist]

def procPVL(l): #apply to 'PV' or 'Load' features and transform them from strings to float arrays
  l=l.split('\n')
  r=[]
 # print(l)
  for i in l:
    i = re.sub(r'\[', '', i)
    i = re.sub(r'  +', ' ', i)
    i = re.sub(r']', '', i)
    
    u = i.split(' ')

    u = list(filter(lambda x: x != '', u))

    r.append(u)

  r = [*map(float, pyflatten(r))]
  r = np.array(r)
  return r

df = pd.read_csv("ANTICIPATE_dataset.csv")
df1 = pd.read_csv("CONTINGENCY_dataset.csv")

# process instances data
df['PV(kW)'] = df['PV(kW)'].apply(procPVL)
df['Load(kW)'] = df['Load(kW)'].apply(procPVL)

class dataANT:
    def __init__(self):
        self.dim = 2
        self.lb = np.concatenate([1 * np.ones(1), 0 * np.ones(1)])
        self.ub = np.concatenate([100 * np.ones(1), 99 * np.ones(1)])

    def get_df_feat(self, x, s):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        par = int(round(x[0])) # parametro discreto
        inst = int(round(x[1]))

        if s == ['sol(keuro)']:
            return df.loc[df['nScenarios']==par].loc[:,s].iloc[inst].values[0]
        else:
            return df.loc[df['nScenarios']==par].loc[:,s].iloc[inst].values
        
    def __call__(self, x):
        return self.get_df_feat(x, ['sol(keuro)'])

    def get_res(self, x): 
        return self.get_df_feat(x, ['time(sec)','memAvg(MB)'])


f = dataANT()

class Levy:
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        
    def __call__(self, x):

        w = 1 + (x - 1.0) / 4.0
        val = np.sin(np.pi * w[0]) ** 2 + \
            np.sum((w[1:self.dim - 1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[1:self.dim - 1] + 1) ** 2)) + \
            (w[self.dim - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[self.dim - 1])**2)
        return val

f2 = Levy(10)

turbo1 = Turbo1(
    f=f,  # Handle to objective function
    cs=[0,1],
    lb=f.lb,  # Numpy array specifying lower bounds
    ub=f.ub,  # Numpy array specifying upper bounds
    n_init=20,  # Number of initial bounds from an Latin hypercube design
    max_evals = 500,  # Maximum number of evaluations
    batch_size=100,  # How large batch size TuRBO uses
    verbose=True,  # Print information from each batch
    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
    min_cuda=1024,  # Run on the CPU for small datasets
    device="cpu",  # "cpu" or "cuda"
    dtype="float64",  # float64 or float32
)


turbo2 = Turbo1(
    f=f2,  # Handle to objective function
    cs=[0,1],
    lb=f2.lb,  # Numpy array specifying lower bounds
    ub=f2.ub,  # Numpy array specifying upper bounds
    n_init=20,  # Number of initial bounds from an Latin hypercube design
    max_evals = 500,  # Maximum number of evaluations
    batch_size=100,  # How large batch size TuRBO uses
    verbose=True,  # Print information from each batch
    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
    min_cuda=1024,  # Run on the CPU for small datasets
    device="cpu",  # "cpu" or "cuda"
    dtype="float64",  # float64 or float32
)

turbo1.optimize()

X = turbo1.X  # Evaluated points
fX = turbo1.fX  # Observed values
ind_best = np.argmin(fX)
f_best, x_best = fX[ind_best], X[ind_best, :]

print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3)))