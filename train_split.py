
from turbo.gp import train_gp
from turbo.utils import gaussian_copula

import gpytorch
import numpy as np
import torch
import pandas as pd

import argparse
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm


def check_split(value):
    ivalue = int(value)
    if not ivalue in [20,30,40]:
        raise argparse.ArgumentTypeError("%s must be one of [20,30,40]" % value)
    return ivalue

def check_zero(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s must be at least 0" % value)
    return ivalue

# Create the parser
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='ant', type=str)
parser.add_argument('--test_split', default=30, type=check_split)
parser.add_argument('--steps', default=50, type=int)    
parser.add_argument('--transform', action='store_true', help='Use the transformations on the observations')
parser.add_argument('--max_time', default=0, type=check_zero, help='Time constraint for single function call, pass 0 to not use constraint')
parser.add_argument('--max_mem', default=0, type=check_zero, help='Memory constraint for single function call, pass 0 to not use constraint')

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
cs = [args.max_time, args.max_mem]

X = df[[par_name, 'instance']].values
fX = df[['sol(keuro)']].values.ravel()
cX = df[['time(sec)','memAvg(MB)']].values
obj_name = 'objective_state_raw.pth'
transf = 'raw'

if args.transform:
    for i in range(cX.shape[1]):
        k = cs[i]
        cX[:,i] = np.sign(cX[:,i] - k) * np.log(1 + np.absolute(cX[:,i] - k)) + k
    fX = gaussian_copula(fX)

    obj_name = 'objective_state_transformed_t{}_m{}.pth'.format(cs[0], cs[1])
    transf = 'transformed'

with gpytorch.settings.max_cholesky_size(max_cholesky_size):
    X_torch = torch.tensor(X).to(device=device, dtype=dtype)
    y_torch = torch.tensor(fX).to(device=device, dtype=dtype).ravel()
    gp = train_gp(
        train_x=X_torch, train_y=y_torch, use_ard=use_ard, num_steps=n_training_steps, dict_key='', freeze=False, state_dict=None
    )

    # Save state dict
    torch.save(gp.state_dict(), base_path+dataset_name+dir_name+obj_name)

# GPs for constraints
cgps = [0]*n_constraints

for i in range(n_constraints):
    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        cgps[i] = train_gp(
            train_x=X_torch, train_y=torch.tensor(cX[:,i]).to(device=device, dtype=dtype).ravel(),
             use_ard=use_ard, num_steps=n_training_steps, dict_key='', freeze=False, state_dict=None
        )

        # Save state dict
        torch.save(cgps[i].state_dict(), base_path+dataset_name+dir_name+'cons_{}_state_{}_t{}_m{}.pth'.format(i, transf, cs[0], cs[1]))