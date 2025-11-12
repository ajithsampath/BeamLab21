#Author: Ajith Sampath
#Affiliation: University of Geneva

from lib import *

with open('config_fit.yaml', 'r') as file:
    config = yaml.safe_load(file)


coef_redname = config['coef_redname']

#reorder the coef as per their corresponding Noll indices
coef = reorder_coef(np.loadtxt(coef_redname))

#generate Noll indices
j = np.arange(len(coef))

vectorized_NollToQuantum = np.vectorize(NollToQuantum)

#generate quantum indices
n_val,m_val = vectorized_NollToQuantum(j)

coef_jnm = np.column_stack((j,n_val,m_val,coef))

#load into a dataframe with column names
df_coef = pd.DataFrame(coef_jnm, columns=['j','n','m','coef'])

coefs = df_coef.iloc[:, 3].values

# Sort indices by absolute value of 4th column, descending
sorted_indices = np.argsort(np.abs(coefs))[::-1]

# Calculate cumulative contribution ratio 
sorted_coefs = coefs[sorted_indices]
cumulative_contrib = np.cumsum(np.abs(sorted_coefs)) / np.sum(np.abs(sorted_coefs))

# Find how many coefficients to keep for 99% contribution
num_coefs = np.searchsorted(cumulative_contrib, 0.99) + 1

# Select indices of top coefficients
selected_indices = sorted_indices[:num_coefs]

# Select corresponding rows from original dataframe, preserving all columns
selected_df = df_coef.iloc[selected_indices]
selected_df = selected_df.sort_index()

selected_df.to_csv('coef_test.csv', index=False)
