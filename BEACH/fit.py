#Fit gaussian/Zernikes to EMsims or Drone data 
#Returns fit parameters (gaussian parameters or Zernike Coefficients) and model beam.


from lib import *

#read config_fit.yaml file
with open('config_fit.yaml', 'r') as file:
    config = yaml.safe_load(file)

datafile = config['datafile']
freq = config['frequency']
xrange = config['xrange']
yrange = config['yrange']
error = config['error']
N = config['N'] 
minimize_method = config['minimize_method']

output_name = config['output_name']
plot_results = config['plot_results']
save_plots = config['save_plots']
plot_format = config['plot_format']
plot_directory = config['plot_directory']

#Create x and y arrays based on xrange and yrange
x = np.linspace(xrange[0],xrange[1],100)
y = np.linspace(yrange[0],yrange[1],100)   

#Initialize FitBeam class
fitbeam = FitBeam(freq,x,y,error,N)
#Load data
Observed = fitbeam.load_beam(datafile)
#Generate Zernike basis
Zernike_basis = fitbeam.basis_N(params=None)
#Fit data to Zernike basis
fit_params, model_beam = fitbeam.fit_data(Observed,Zernike_basis,minimize_method)   
#Save fit parameters to csv file
np.savetxt(output_name, fit_params, delimiter=",")
#Plot results
if plot_results:
    fitbeam.plot_results(Observed,model_beam,fit_params,save_plots,plot_format,plot_directory)  