
from turbo.gp import train_gp
from turbo.utils import from_unit_cube, to_unit_cube

import gpytorch
import numpy as np
import torch
import pandas as pd
from scipy.stats import norm
import argparse
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm

def gaussian_copula(observation):
    # Convert observations to quantiles using empirical CDF
    quantiles = np.argsort(observation).argsort() / len(observation)
    
    # Map quantiles through inverse Gaussian CDF
    transformed = norm.ppf(quantiles)
    
    return transformed

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
parser.add_argument('--transform', action='store_true', help='Use the transformations on the observations')

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
fX = df[['sol(keuro)']].values.ravel()
cX = df[['time(sec)','memAvg(MB)']].values
obj_name = 'objective_state_raw.pth'
transf = 'raw'

if args.transform:
    cX = np.sign(np.log(1+np.absolute(cX)))

    obs_sorted = np.sort(fX)
    # Map observations to quantiles
    quantiles = np.linspace(0, 1, len(obs_sorted))
    empirical_cdf = dict(zip(obs_sorted, quantiles))
    rank = np.array([empirical_cdf[x] for x in fX])
    quantiles = rank / len(obs_sorted)
    # Apply inverse Gaussian CDF
    fX = norm.ppf(quantiles)

    obj_name = 'objective_state_transformed.pth'
    transf = 'transformed'

with gpytorch.settings.max_cholesky_size(max_cholesky_size):
    X_torch = torch.tensor(X).to(device=device, dtype=dtype)
    y_torch = torch.tensor(fX).to(device=device, dtype=dtype).ravel()
    gp = train_gp(
        train_x=X_torch, train_y=y_torch, use_ard=use_ard, num_steps=n_training_steps, freeze=False, state_dict=None
    )

    # Save state dict
    torch.save(gp.state_dict(), base_path+dataset_name+dir_name+obj_name)

# GPs for constraints
cgps = [0]*n_constraints

for i in range(n_constraints):
    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        cgps[i] = train_gp(
            train_x=X_torch, train_y=torch.tensor(cX[:,i]).to(device=device, dtype=dtype).ravel(),
             use_ard=use_ard, num_steps=n_training_steps, freeze=False, state_dict=None
        )

        # Save state dict
        torch.save(cgps[i].state_dict(), base_path+dataset_name+dir_name+'cons_{}_state_{}.pth'.format(i, transf))