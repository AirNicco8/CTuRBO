
from turbo.gp import train_gp

import gpytorch
import numpy as np
import torch
import pandas as pd
import re
import random
import argparse
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

def check_split(value):
    ivalue = int(value)
    if not ivalue in [20,30,40]:
        raise argparse.ArgumentTypeError("%s must be one of [20,30,40]" % value)
    return ivalue

# Create the parser
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='ant', type=str)
parser.add_argument('--test_split', default=30, type=check_split)
parser.add_argument('--steps', default=50, type=int)
# Parse the argument
args = parser.parse_args()
ts = args.test_split

base_path = "dataset_splits/"

if args.data == 'ant':
    dataset_name = 'ant/'
    par_name = 'nScenarios'
else:
    dataset_name = 'cont/'
    par_name = 'nTraces'


dir_name = str(100-ts)+'_'+str(ts)+'/'
test_name = 'train_split'

df = pd.read_csv(base_path+dataset_name+dir_name+test_name)

instances = len(df)

max_cholesky_size = 2000
device, dtype = torch.device("cpu"), torch.float64
use_ard=True
n_training_steps = args.steps
n_constraints = 2

X = df[[par_name, 'instance']].values
fX = df[['sol(keuro)']].values
cX = df[['time(sec)','memAvg(MB)']].values

with gpytorch.settings.max_cholesky_size(max_cholesky_size):
    X_torch = torch.tensor(X).to(device=device, dtype=dtype)
    y_torch = torch.tensor(fX).to(device=device, dtype=dtype).ravel()
    gp = train_gp(
        train_x=X_torch, train_y=y_torch, use_ard=use_ard, num_steps=n_training_steps, state_dict=None
    )

    # Save state dict
    torch.save(gp.state_dict(), base_path+dataset_name+dir_name+'objective_state.pth')

# GPs for constraints
cgps = [0]*n_constraints

for i in range(n_constraints):
    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        cgps[i] = train_gp(
            train_x=X_torch, train_y=torch.tensor(cX[:,i]).to(device=device, dtype=dtype).ravel(),
             use_ard=use_ard, num_steps=n_training_steps, state_dict=None
        )

        # Save state dict
        torch.save(cgps[i].state_dict(), base_path+dataset_name+dir_name+'cons_{}_state.pth'.format(i))