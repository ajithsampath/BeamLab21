#Author: Ajith Sampath
#Affiliation: University of Geneva

from beamlab21.lib import *
from beamlab21 import ROOT_DIR

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

# --- Compute impact for each coefficient ---
# impact_i = || c_i * X[:, i] ||  (true contribution to model)
impacts = np.linalg.norm(coefs * X, axis=0)

# --- Sort coefficients by impact (descending) ---
sorted_indices = np.argsort(impacts)[::-1]
sorted_impacts = impacts[sorted_indices]

# --- Cumulative impact contribution (0 to 1) ---
cumulative_contrib = np.cumsum(sorted_impacts) / np.sum(sorted_impacts)

# --- Choose number of coefficients needed for desired % impact ---
num_coefs = np.searchsorted(cumulative_contrib, config['percentage_energy']/100) + 1

# --- Select the top-impact coefficients ---
selected_indices = sorted_indices[:num_coefs]

# --- Pull corresponding rows from original dataframe ---
selected_df = df_coef.iloc[selected_indices].sort_index()

# --- Save reduced coefficient file ---
out_coef_name = os.path.join(ROOT_DIR, config['out_coef_dir'], 'reduced_coef.csv')
selected_df.to_csv(out_coef_name, index=False)


print("New coefficient file with Noll, Quantum indices and reduced Coefficients is written!")
print(f"Check the file at {out_coef_name}")