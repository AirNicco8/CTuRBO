import numpy as np
import pandas as pd
import re
import random
import argparse
import warnings
from pathlib import Path
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

base_path = "dataset_splits/"

dataset_name = 'ant/'
dataset_name1 = 'cont/'

test_name = 'test_split'
train_name = 'train_split'

df = pd.read_csv('datasets/anticipate.csv')
df1 = pd.read_csv('datasets/contingency.csv')


# Create the parser
parser = argparse.ArgumentParser()
parser.add_argument('--test_split', default=30, type=int)
# Parse the argument
args = parser.parse_args()
ts = args.test_split

split_from_frac = int(ts*len(df)/100)
split_from_frac1 = int(ts*len(df1)/100)

# create split dirs
dir_name = str(100-ts)+'_'+str(ts)+'/'
Path(base_path+dataset_name+dir_name).mkdir(parents=True, exist_ok=True)
Path(base_path+dataset_name1+dir_name).mkdir(parents=True, exist_ok=True)

# split ANT
ind_to_split=random.sample(range(0, 100), int(split_from_frac/100))
mask=[]

for i in range(0,len(df),100):
    mask+=list(np.array(ind_to_split) + i)

tosave = df.iloc[mask]
test_df = df.index.isin(mask)
trsave = df[~test_df]

tosave.to_csv(base_path+dataset_name+dir_name+test_name)
trsave.to_csv(base_path+dataset_name+dir_name+train_name)

# split CONT
ind_to_split1=random.sample(range(0, 100), int(split_from_frac1/100))
mask=[]

for i in range(0,len(df),100):
    mask+=list(np.array(ind_to_split1) + i)

tosave = df1.iloc[mask]
test_df = df1.index.isin(mask)
trsave = df[~test_df]

tosave.to_csv(base_path+dataset_name1+dir_name+test_name)
trsave.to_csv(base_path+dataset_name1+dir_name+train_name)