from turbo import Turbo1, TurboM
import numpy as np
import pandas as pd
import re
import torch
import math
import matplotlib
import matplotlib.pyplot as plt
import argparse

import warnings
warnings.filterwarnings("ignore")

def pyflatten(l):
    return [item for sublist in l for item in sublist]

def procPVL(l): #apply to 'PV' or 'Load' features and transform them from strings to float arrays
  l=l.split('\n')
  r=[]
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

        tor = df.loc[df['nScenarios']==par].loc[:,s].iloc[inst].values

        return tor[0] if s == ['sol(keuro)'] else tor
        
    def __call__(self, x):
        return self.get_df_feat(x, ['sol(keuro)'])

    def get_res(self, x): 
        return self.get_df_feat(x, ['time(sec)','memAvg(MB)'])

def check_one(value):
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("%s must be at least 1" % value)
    return ivalue

def check_zero(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s must be at least 0" % value)
    return ivalue

if __name__== "__main__":
    f = dataANT()

    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--max_time', default=0, type=check_zero, help='Time constraint for function calls, pass 0 to not use constraint')
    parser.add_argument('--max_mem', default=0, type=check_zero, help='Memory constraint for function calls, pass 0 to not use constraint')
    parser.add_argument('--max_evals', default=200, type=check_zero)
    parser.add_argument('--n_init', default=20, type=check_one)
    parser.add_argument('--trust_regions', default=5, type=check_one)
    # Parse the argument
    args = parser.parse_args()

    if args.trust_regions == 1:
        turbo = Turbo1(
            f=f,  # Handle to objective function
            tcs=[0, 0], # (!!!) DEBUG passing
            cs=[args.max_time, args.max_mem],
            lb=f.lb,  # Numpy array specifying lower bounds
            ub=f.ub,  # Numpy array specifying upper bounds
            n_init=20,  # Number of initial bounds from an Latin hypercube design
            max_evals = args.max_evals,  # Maximum number of evaluations
            batch_size=1,  # How large batch size TuRBO uses
            verbose=True,  # Print information from each batch
            use_ard=True,  # Set to true if you want to use ARD for the GP kernel
            max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
            n_training_steps=50,  # Number of steps of ADAM to learn the hypers
            min_cuda=1024,  # Run on the CPU for small datasets
            device="cpu",  # "cpu" or "cuda"
            dtype="float64",  # float64 or float32
        )
    else:
      turbo = TurboM(
        f=f,  # Handle to objective function
        tcs=[args.max_time, args.max_mem],
        lb=f.lb,  # Numpy array specifying lower bounds
        ub=f.ub,  # Numpy array specifying upper bounds
        n_init=20,  # Number of initial bounds from an Latin hypercube design
        max_evals = args.max_evals,  # Maximum number of evaluations
        n_trust_regions=args.trust_regions,  # Number of trust regions
        batch_size=1,  # How large batch size TuRBO uses
        verbose=True,  # Print information from each batch
        use_ard=True,  # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
        n_training_steps=50,  # Number of steps of ADAM to learn the hypers
        min_cuda=1024,  # Run on the CPU for small datasets
        device="cpu",  # "cpu" or "cuda"
        dtype="float64",  # float64 or float32
        )

    turbo.optimize()

    X = turbo.X.astype(int)  # Evaluated points
    fX = turbo.fX.astype(int)  # Observed values
    ind_best = np.argmin(fX)
    f_best, x_best = fX[ind_best], X[ind_best, :]

    print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s\nWith time: %d seconds ,and memory: %d Mb" % (f_best, np.around(x_best, 3), turbo.currTime, turbo.currMem))