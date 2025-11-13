#Author: Ajith Sampath
#Affiliation: University of Geneva

#Fit gaussian/Zernikes to EMsims or Drone data 
#Returns fit parameters (gaussian parameters or Zernike Coefficients) and model beam.

from BeamLab21.lib import *
from BeamLab21 import ROOT_DIR

#read config_fit.yaml file
yamlpath = os.path.join(ROOT_DIR, 'fit', 'config_fit.yaml')

with open(yamlpath, 'r') as file:
    config = yaml.safe_load(file)

#read in necessary parameters    
telescope_name = config['Telescope_name']

datafile = os.path.join(ROOT_DIR,config["data_dir"],config['datafile'])
freq = config['frequency']
fac = config['fac']

plot_results = config['plot_results']
plot_format = config['plot_format']
plot_directory = os.path.join(ROOT_DIR,config['plot_directory'])
init_gparams = np.array(config['init_gparams'])  # Initial guess for Gaussian sigmas in arcminutes

save_outputs = config['save_outputs']
output_dir = config['output_dir']
output_name = config['output_filename']+config['output_format']

save_coef = config["save_coef"]
coef_filename = config["out_coef_filename"]

print("Fitting data from file:", datafile)
print("Telescope:", telescope_name)
print("Frequency channel (MHz):", config['frequency'])

#Initialize GaussianFit class
gfit = GaussianFit(datafile,freq,error_type=config['gaussian_error_type'],normalize_data=config['normalize_data'],coord_type=config['coord_type'])
print("Data loaded. Shape of observed data:", gfit.data.shape)
print("Fitting a 2D Gaussian to calculate beam width...\n")
#Optimize Gaussian fit
print("Starting Gaussian fit optimization...\n")

x,y,xo,yo,freq_arr,freq,sigx_gopt,sigy_gopt,gExpected,data,_= gfit.optimize_Gauss(init_gparams,minimize_method = config['gminimize_method'],xtol=config['gtol'],verbose=config['gverbose'])
print("Gaussian fit completed. Optimized sigx:", sigx_gopt, "sigy:", sigy_gopt)

#Generate Zernike basis
print("Generating Zernike basis...\n")
ztfit = ZernikeFit(x,y,xo,yo,freq_arr,freq,data,config['N'],error_type=config['zernike_error_type'],normalize_data=config['normalize_data'],coord_type=config['coord_type'])

#Fit data to Zernike basis
init_ztparams = [sigx_gopt,sigy_gopt] # Use optimized Gaussian sigmas as initial guess for Zernike fit

if config['skip_minimise']:
    sigx,sigy,coef,model_beam= ztfit.NO_optimize_ZT(init_ztparams,fac)

else:
    print("Starting Zernike fit optimization by varying scaling parameters...\n")

    sigx,sigy,coef,model_beam,optfun = ztfit.optimize_ZT(init_ztparams,minimize_method=config['ztminimize_method'],xtol=config['zttol'],maxiter=config['ztmaxiter'])
    print("Zernike fit completed. Fit parameters:", coef)


#Save fit parameters to csv file
if save_coef:
    cf = pd.DataFrame({"coef_full": coef})
    coef_filename = os.path.join(ROOT_DIR,output_dir,coef_filename)
    cf.to_csv(coef_filename, index=False)
    print(f"Coefficients are saved in {coef_filename}!\n")
    df = pd.DataFrame({"freq(MHz)": [freq], "sigx": [sigx], "sigy": [sigy]})
    os.path.join(ROOT_DIR,output_dir)
    sp_filename = os.path.join(ROOT_DIR,output_dir,"scaleparameters.csv")
    df.to_csv(sp_filename, index=False)
    print(f"Scaling parameters are saved in {sp_filename}!\n")
#Plot results
if plot_results:
    ztfit.plot_results_cart(gfit.data,model_beam,freq,config['N'],x,y,plot_format,plot_directory) 
    print("Plotted and saved...!!!\n")
else:
    print("The results are not plotted and hence not saved..!!!\n")

if save_outputs:
    np.savez(os.path.join(ROOT_DIR,output_dir,output_name),x=x,y=y,xo=xo,yo=yo,model=model_beam)
else:
    print("Fitted model is not saved! Set save_outputs parameters to True in the config_fit.yaml file :)\n")


print("Decomposing/Fitting the beam for a single given frequency is done!!\n")
