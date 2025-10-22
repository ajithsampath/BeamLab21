#Fit gaussian/Zernikes to EMsims or Drone data 
#Returns fit parameters (gaussian parameters or Zernike Coefficients) and model beam.


from lib import *

#read config_fit.yaml file
with open('config_fit.yaml', 'r') as file:
    config = yaml.safe_load(file)
telescope_name = config['Telescope_name']

datafile = config['datafile']
freq = config['frequency']

output_name = config['output_filename']+config['output_format']
plot_results = config['plot_results']
save_plots = config['save_plots']
plot_format = config['plot_format']
plot_directory = config['plot_directory']
init_gparams = config['init_gparams']  # Initial guess for Gaussian sigmas in arcminutes

print("Fitting data from file:", datafile)
print("Telescope:", telescope_name)
print("Frequency channel (MHz):", config['frequency'])

#Initialize GaussianFit class
gfit = GaussianFit(datafile,freq,error_type=config['gaussian_error_type'])
print("Data loaded. Shape of observed data:", gfit.Observed.shape)
#Optimize Gaussian fit
print("Starting Gaussian fit optimization...")
x,y,freq_arr,freq,sigx_gopt,sigy_gopt,gExpected,data,_= gfit.optimize_Gauss([init_gparams],minimize_method = config['gminimize_method'],xtol=config['gtol'],maxiter=config['gmaxiter'])

#Generate Zernike basis
ztfit = ZernikeFit(x,y,freq_arr,freq,data,config['N'],error_type=config['zernike_error_type'],normalize_data=config['normalize_data'])

#Fit data to Zernike basis
init_ztparams = [sigx_gopt,sigy_gopt] # Use optimized Gaussian sigmas as initial guess for Zernike fit

fit_params, model_beam = ztfit.optimize_ZT(init_ztparams,minimize_method=config['ztminimize_method'],xtol=config['zttol'],maxiter=config['ztmaxiter'])



#Save fit parameters to csv file
np.savetxt(output_name, fit_params, delimiter=",")
#Plot results
if plot_results:
    ztfit.plot_results(gfit.Observed,model_beam,fit_params,save_plots,plot_format,plot_directory)  