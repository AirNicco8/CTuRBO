from turbo import Turbo1, TurboM
from turbo.utils import bilog
import numpy as np
import pandas as pd
import torch
import argparse

import warnings
warnings.filterwarnings("ignore")

class dataF:
    def __init__(self, df, par_name, ub_split):
        self.dim = 2
        self.df = df
        self.inst = -1
        self.instances = set(df.instance.values)
        self.par_name = par_name
        self.lb = np.concatenate([1 * np.ones(1), 0 * np.ones(1)])
        self.ub = np.concatenate([100 * np.ones(1), 99 * np.ones(1)])

        self.olb = np.concatenate([1 * np.ones(1), 0 * np.ones(1)])
        self.oub = np.concatenate([100 * np.ones(1), 99 * np.ones(1)])

    def get_df_feat(self, x, s):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        par = int(round(x[0])) # parametro discreto
        insta = int(round(x[1])) # indice istanza da dataset base
        tor = self.df.loc[self.df[self.par_name]==par].loc[self.df['instance']==insta].loc[:,s].values #self.df.loc[self.df[self.par_name]==par].loc[:,s].iloc[inst].values

        return tor[0] #if s == ['sol(keuro)'] else tor
        
    def __call__(self, x):
        return self.get_df_feat(x, ['sol(keuro)'])

    def get_res(self, x): 
        return self.get_df_feat(x, ['time(sec)','memAvg(MB)'])

    def get_instance(self, x):
        par = int(round(x[0])) # parametro discreto
        inst = int(round(x[1])) # indice relativo al dataset split
        ret = self.df.loc[self.df[self.par_name]==par].loc[:,['instance']].iloc[inst].values
        assert int(ret) in self.instances
        return ret

class dataFix:
    def __init__(self, df, par_name, inst):
        self.dim = 1
        self.df = df
        self.inst = inst
        self.par_name = par_name
        self.lb = 1 * np.ones(1)
        self.ub = 100 * np.ones(1)
        
        self.olb = 1 * np.ones(1)
        self.oub = 100 * np.ones(1)

    def get_df_feat(self, x, s):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        par = int(round(x[0])) # parametro discreto

        tor = self.df.loc[self.df[self.par_name]==par].loc[:,s].values #self.df.loc[self.df[self.par_name]==par].loc[:,s].iloc[inst].values

        return tor[0] #if s == ['sol(keuro)'] else tor
        
    def __call__(self, x):
        return self.get_df_feat(x, ['sol(keuro)'])

    def get_res(self, x): 
        return self.get_df_feat(x, ['time(sec)','memAvg(MB)'])

    def get_instance(self, x):
        par = int(round(x[0])) # parametro discreto
        inst = int(round(x[1])) # indice relativo al dataset split
        ret = self.df.loc[self.df[self.par_name]==par].loc[:,['instance']].iloc[inst].values
        assert int(ret) in self.instances
        return ret


def constraints_violation(estimates, cs):
    tmp = np.zeros(estimates.shape[0])

    for i in range(len(cs)):
        violations = cs[i] - estimates[:, i]
        violations[violations > 0] = 0.0
        tmp = np.dstack((tmp, violations))

    return np.sum(tmp, axis=2).T

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

def check_split(value):
    ivalue = int(value)
    if not ivalue in [100,20,30,40]:
        raise argparse.ArgumentTypeError("%s must be one of [20,30,40,100]" % value)
    return ivalue

def check_sol(value):
    ivalue = int(value)
    if not ivalue in [0, 10, 20]:
        raise argparse.ArgumentTypeError("%s must be one of [0,10,20]" % value)
    return ivalue


if __name__== "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--data', default='ant', type=str, help='Select the dataset to use, ant for ANTICIPATE, cont for CONTINGENCY')
    parser.add_argument('--fix_instance', default=-1, type=check_zero, help='It fixes the instance to optimize only the parameter. if -1 run optimization on all the data')
    parser.add_argument('--split', default=100, type=check_split, help='Select the GP trained split to use')
    parser.add_argument('--max_time', default=0, type=check_zero, help='Time constraint for single function call, pass 0 to not use constraint')
    parser.add_argument('--max_mem', default=0, type=check_zero, help='Memory constraint for single function call, pass 0 to not use constraint')
    parser.add_argument('--sol_q', default=0, type=check_sol, help='The solution quality expressed as percentage improvement over base solution, 0 for no constraint')
    parser.add_argument('--cum_time', default=0, type=check_zero, help='Cumulative time constraint for function calls, pass 0 to not use constraint')
    parser.add_argument('--cum_mem', default=0, type=check_zero, help='Cumulative memory constraint for function calls, pass 0 to not use constraint')
    parser.add_argument('--batch_size', default=1, type=check_one)
    parser.add_argument('--max_evals', default=200, type=check_zero)
    parser.add_argument('--n_init', default=20, type=check_one)
    parser.add_argument('--trust_regions', default=5, type=check_one)
    parser.add_argument('--rand', action='store_true', help='Use randomized splits')
    parser.add_argument('--load_gp', action='store_true', help='Load pre-trained GPs, the GP is trained on 100 minus the split passed as argument (percentage of the dataset)')
    parser.add_argument('--freeze_gp', action='store_true', help='Freeze the GPs and do not perform training during the algorithm')
    parser.add_argument('--transform', action='store_true', help='Use the GPs fitted with transformations on the observations')
    parser.add_argument('--csv', action='store_true', help='Add results to csv')
    # Parse the argument
    args = parser.parse_args()

    # PATHS
    base_path = "dataset_splits"
    if args.rand:
        base_path = "dataset_splits_rand/"

    if(args.data == 'ant'):
        dataset_name = 'ant/'
        par_name = 'nScenarios'
        df_name = 'anticipate.csv'
    else:
        dataset_name = 'cont/'
        par_name = 'nTraces'
        df_name = 'contingency.csv'

    dir_name = str(100-args.split)+'_'+str(args.split)+'/'
    test_name = 'test_split'

    obj_name = 'objective_state_raw.pth'
    transf = 'raw'

    cs=[args.max_time, args.max_mem] # constraints bounds

    # Load Pre-trained GPs
    gp_dict=None
    if args.load_gp:
        if args.transform: # use transformations on observed values
                obj_name = 'objective_state_transformed.pth'
                transf = 'transformed'
                cs = list(map(bilog, cs)) # transformed constraints bounds

        print('Loading {} trained GPs - path: {}'.format(transf, base_path+dataset_name+dir_name))
        gp_dict={}
        gp_dict['obj'] = torch.load(base_path+dataset_name+dir_name+obj_name)
        for i in range(2):
            name = 'cons_{}_state_{}.pth'.format(i, transf) 
            gp_dict['cons_{}'.format(i)] = torch.load(base_path+dataset_name+dir_name+name)

    frs = 'freezed' if args.freeze_gp else 'unfreezed'
    csv_name = 't2_{}_{}.csv'.format(frs, transf)

    if args.csv: # create csv for results
        try:
            out_df = pd.read_csv(base_path+dataset_name+dir_name+csv_name)
            print('Loading csv: '+base_path+dataset_name+dir_name+csv_name)
        except:
            out_df = pd.DataFrame(columns=['instance', 'Time bound (s)', 'Sol improvement', par_name, 'memAvg(MB)', 'baseline sol(keuro)', 'sol(keuro)'])
 
    # Fixed Instance
    if args.fix_instance == -1:
        df = pd.read_csv('datasets/'+df_name) # always run the algorithm on the full dataset
        base_sol = df.loc[df[par_name]==1].loc[:, ['sol(keuro)']].max().values[0][0]
        f = dataF(df, par_name, 99)
    else: # run on all instances
        df = pd.read_csv('datasets/'+df_name) # always run the algorithm on the full dataset
        # baseline solution, minimum resources
        base_sol = df.loc[df[par_name]==1].loc[df['instance']==args.fix_instance].loc[:, ['sol(keuro)']].values[0][0]
        df = df.loc[df['instance']==args.fix_instance]
        f = dataFix(df, par_name, args.fix_instance)

    if args.trust_regions == 1:
        turbo = Turbo1(
            f=f,  # Handle to objective function
            tcs=[args.cum_time, args.cum_mem], 
            cs=cs, # solution constraint
            lb=f.lb,  # Numpy array specifying input lower bounds
            ub=f.ub,  # Numpy array specifying input upper bounds
            olb=f.olb,  # Numpy array specifying input lower bounds
            oub=f.oub,  # Numpy array specifying input upper bounds
            n_init=20,  # Number of initial bounds from an Latin hypercube design
            max_evals = args.max_evals,  # Maximum number of evaluations
            batch_size=args.batch_size,  # How large batch size TuRBO uses
            verbose=True,  # Print information from each batch
            use_ard=True,  # Set to true if you want to use ARD for the GP kernel
            max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
            n_training_steps=50,  # Number of steps of ADAM to learn the hypers
            min_cuda=1024,  # Run on the CPU for small datasets
            device="cpu",  # "cpu" or "cuda"
            dtype="float64",  # float64 or float32
            gp_dict=gp_dict, # gp state dict if we load trained gp
            freeze_flag=args.freeze_gp, # flag to freeze the gps 
            transform_flag=args.transform # flag to transform observations
        )
    else:
      turbo = TurboM(
        f=f,  # Handle to objective function
        tcs=[args.cum_time, args.cum_mem], 
        cs=cs, # solution constraint
        lb=f.lb,  # Numpy array specifying lower bounds
        ub=f.ub,  # Numpy array specifying upper bounds
        olb=f.olb,  # Numpy array specifying lower bounds
        oub=f.oub,  # Numpy array specifying upper bounds
        n_init=20,  # Number of initial bounds from an Latin hypercube design
        max_evals = args.max_evals,  # Maximum number of evaluations
        n_trust_regions=args.trust_regions,  # Number of trust regions
        batch_size=args.batch_size,  # How large batch size TuRBO uses
        verbose=True,  # Print information from each batch
        use_ard=True,  # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
        n_training_steps=50,  # Number of steps of ADAM to learn the hypers
        min_cuda=1024,  # Run on the CPU for small datasets
        device="cpu",  # "cpu" or "cuda"
        dtype="float64",  # float64 or float32
        gp_dict=gp_dict, # gp state dict if we load trained gp
        freeze_flag=args.freeze_gp,
        transform_flag=args.transform # flag to transform observations
        )

    turbo.optimize()

    X = turbo.X.astype(int)  # Evaluated points
    fX = turbo.fX.astype(float)  # Observed values
    cX = turbo.cX.astype(float)  # Observed resources
    csf = [] # constraints
    
    for i in cs:
        if i==0: # the constraint is not used
            csf.append(np.inf) 
        else:
            csf.append(i)

    satisfactions = [cX[:,i] <= csf[i] for i in range(len(csf))]
    cum_and = np.isfinite(fX).ravel()
    for e in satisfactions:
        cum_and = np.logical_and(cum_and, e)

    # improve solution of desired percentage
    if args.sol_q != 0:
        sol_improvement = (fX <= (base_sol - (args.sol_q*base_sol/100))).ravel()
        cum_and = np.logical_and(cum_and, sol_improvement)

        try:
            subset_idx = np.argmin(fX[cum_and])
            ind_best = np.arange(fX.size)[cum_and.ravel()][subset_idx]

            f_best, x_best, c_best = fX[ind_best][0], X[ind_best, :], cX[ind_best, :]

            print("Best feasible value found:\n\tf(x) = %.3f\nObserved at:\n\t %s = %s\nWith time: %d seconds, and memory: %d Mb" % (f_best, par_name, x_best[0], c_best[0], c_best[1]))
            print("Base solution: %.3f - improvement needed %d%%" % (base_sol, args.sol_q))
            print("with total time: %d seconds, and total memory: %d Mb" % (turbo.currTime, turbo.currMem))
        except:
            total_violations = -(constraints_violation(cX, csf))
            mask = np.isfinite(fX)

            subset_idx = np.argmin(total_violations[mask])
            ind_best = np.arange(fX.size)[mask.ravel()][subset_idx]

            f_best, x_best, c_best = fX[ind_best][0], X[ind_best, :], cX[ind_best, :]

            print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\t %s = %s\nWith time: %d seconds, and memory: %d Mb" % (f_best, par_name, x_best[0], c_best[0], c_best[1]))
            print("Base solution: %.3f - improvement needed %d%%" % (base_sol, args.sol_q))
            print("The value is NOT feasible, but it is the one with minimum total violations with %d evals" % (args.max_evals))
            x_best[0] = '-1'
            f_best = -1
    else:
        f_best, x_best, c_best = base_sol, [1], [0,60.12]

        print("Baseline solution:\n\tf(x) = %.3f\nObserved at:\n\t %s = %s\nWith memory: %d Mb" % (f_best, par_name, x_best[0], c_best[1]))

    insert_row = {
        'instance': args.fix_instance, 
        'Time bound (s)': csf[0], 
        'Sol improvement': args.sol_q, 
        par_name : x_best[0], 
        'memAvg(MB)': c_best[1],
        'baseline sol(keuro)': base_sol, 
        'sol(keuro)': f_best
    }
    
    if args.csv: # create csv for results
        out_df = pd.concat([out_df, pd.DataFrame([insert_row])])
        print('Saving to '+base_path+dataset_name+dir_name+csv_name)
        print(insert_row)
        out_df.to_csv(base_path+dataset_name+dir_name+csv_name, index=False)