from lib import *

with open('config_fit.yaml', 'r') as file:
    config = yaml.safe_load(file)


coef_redname = config['coef_redname']
coef = reorder_coef(np.loadtxt(coef_redname))

j = np.arange(len(coef))

vectorized_NollToQuantum = np.vectorize(NollToQuantum)

n_val,m_val = vectorized_NollToQuantum(j)

coef_jnm = np.column_stack((j,n_val,m_val,coef))


df_coef = pd.DataFrame(coef_jnm, columns=['j','n','m','coef'])

df_coef.to_csv('coef_test.csv', index=False)
