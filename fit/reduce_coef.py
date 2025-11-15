#Author: Ajith Sampath
#Affiliation: University of Geneva

from BeamLab21.lib import *
from BeamLab21 import ROOT_DIR

#read config_fit.yaml file
yamlpath = os.path.join(ROOT_DIR, 'fit', 'config_fit.yaml')

with open(yamlpath, 'r') as file:
    config = yaml.safe_load(file)

in_coef_name = os.path.join(ROOT_DIR,config['in_coef_dir'],config['in_coef_filename']) 

df = pd.read_csv(in_coef_name)
coef = df["coef_full"].to_numpy() 

#reorder the coef as per their corresponding Noll indices
coef = reorder_coef((coef))

#generate Noll indices
j = np.arange(len(coef))

vectorized_NollToQuantum = np.vectorize(NollToQuantum)

#generate quantum indices
n_val,m_val = vectorized_NollToQuantum(j)

coef_jnm = np.column_stack((j,n_val,m_val,coef))

#load into a dataframe with column names
df_coef = pd.DataFrame(coef_jnm, columns=['j','n','m','coef'])

coefs = df_coef.iloc[:, 3].values

# Sort indices by absolute value of coefficient, in descending order
sorted_indices = np.argsort(np.abs(coefs))[::-1]

# Calculate cumulative contribution ratio 
sorted_coefs = coefs[sorted_indices]
cumulative_contrib = np.cumsum(np.abs(sorted_coefs)**2) / np.sum(np.abs(sorted_coefs)**2)

# Find how many coefficients to keep for 100% contribution / removing non-contributing coefs
num_coefs = np.searchsorted(cumulative_contrib, config['percentage_energy']/100) + 1

#print(f"Coefficients are reordered and reduced to most dominant modes with {config['percentage_energy']}% contribution..! ")

# Select indices of top coefficients
selected_indices = sorted_indices[:num_coefs]

# Select corresponding rows from original dataframe, preserving all columns
selected_df = df_coef.iloc[selected_indices]
selected_df = selected_df.sort_index()

out_coef_name = os.path.join(ROOT_DIR,config['out_coef_dir'],'reduced_coef.csv')
selected_df.to_csv(out_coef_name, index=False)

print("New coefficient file with Noll, Quantum indices and reduced Coefficients is written!")
print(f"Check the file at {out_coef_name}")