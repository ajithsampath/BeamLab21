from lib import *

with open('config_fit.yaml', 'r') as file:
    config = yaml.safe_load(file)


coef_filename = config['coef_filename']
coef = reorder_coef(coef_filename)

j = np.arange(len(coef))

coef_noll = np.column_stack((j,coef))

