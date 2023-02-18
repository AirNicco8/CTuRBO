
df = pd.read_csv('ANTICIPATE_dataset.csv')
df1 = pd.read_csv('CONTINGENCY_dataset.csv')

# process instances data
df['PV(kW)'] = df['PV(kW)'].apply(procPVL)
df['Load(kW)'] = df['Load(kW)'].apply(procPVL)

# process instances data
df1['PV(kW)'] = df1['PV(kW)'].apply(procPVL)
df1['Load(kW)'] = df1['Load(kW)'].apply(procPVL)

base_instances_indexes = list(range(100))
new_col = []
for i in range(100):
    new_col += base_instances_indexes

df['instance'] = new_col
df1['instance'] = new_col

df.to_csv('datasets/anticipate.csv')
df1.to_csv('datasets/contingency.csv')