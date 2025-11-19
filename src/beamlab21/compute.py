#Author: Ajith Sampath
#Affiliation: University of Geneva

#Compute HIRAX beam from Zernike basis functions using coefficients from csv file or from Gaussian parameters in yaml file
#Return HIRAX beam model

from beamlab21.lib import *



def run(config_path):
    project_root = get_project_root()
    config = load_config(config_path, context={"frequency": 400})

    freq = config['frequency']
    c = 3e8
    wvl = c/(freq*1e6)
    Deff = 4.6

    angextent = config['pixels']*config['angular_res']
    x = np.linspace(-angextent/2,angextent/2,config['pixels'])
    y = x
    dtype = config['dtype']
    


    if config['gen_gaussian_model']:
        amp = config['gaussian_amp']
        lambdabyDcoef = config['gaussian_lamdabyD_coef']
        sigx = np.rad2deg(lambdabyDcoef*(wvl/Deff))
        sigy = sigx
        xo,yo = config['gaussian_offset_x'],config['gaussian_offset_y']
        tilt = config['gaussian_rotation']
        gparams = amp, sigx,sigy,xo,yo,tilt
        gmodel = twoD_Gaussian(x,y,gparams)

        if config['add_noise2gaussian']:
            np.random.seed(config['gaussian_noise_random_seed']) 
            gnoise = np.random.normal(loc=config['gaussian_noise_mean'], scale=config['gaussian_noise_sigma'], size=gmodel.shape)
        else:
            gnoise = 0.0
            print("No random noise added to the Gaussian Model...\n")

        gmodel = (gmodel + gnoise).reshape(config['pixels'],config['pixels'])

        if config['save_gaussian_model']:
            goutput_dir = os.path.join(project_root,config['goutput_dir'])
            goutput_name = config['goutput_name']
            os.makedirs(os.path.join(project_root,goutput_dir), exist_ok=True) 
            np.savez(os.path.join(project_root,goutput_dir,goutput_name),x=x,y=y,data=gmodel)
        else:
            print("Computed Gaussian model is not saved! Set save_gaussian_model parameters to True in the config_compute.yaml file :)\n")

    elif config['gen_zernike_model']:
        scaleparamfile = os.path.join(project_root,config['scaleparam_dir'],config['scaleparam_file']) 
        coeffile = os.path.join(project_root,config['coef_dir'],config['coef_file']) 
        ztgen = GenZTBeam(freq,x,y,dtype)
        ztsp_df = pd.read_csv(scaleparamfile)
        sigx,sigy = ztsp_df['sigx'].to_numpy(),ztsp_df['sigy'].to_numpy()
        ztgen.load_coef(coeffile)
        ztgen.basisfunc(sigx,sigy)
        ztmodel = np.dot(ztgen.Basis.T, ztgen.coef)

        if config['add_noise2zernike']:
            np.random.seed(config['zernike_noise_random_seed']) 
            znoise = np.random.normal(loc=config['zernike_noise_mean'], scale=config['zernike_noise_sigma'], size=ztmodel.shape)
        else:
            znoise = 0.0
            print("No random noise added to the Gaussian Model...\n")

        ztmodel = (ztmodel + znoise).reshape(config['pixels'],config['pixels'])
        
        if config['save_zernike_model']:        
            zoutput_dir = config['zoutput_dir']
            zoutput_name = config['zoutput_name']+config['zoutput_format']
            os.makedirs(os.path.join(project_root,zoutput_dir), exist_ok=True)
            np.savez(os.path.join(project_root,zoutput_dir,zoutput_name),x=x,y=y,data=ztmodel)
        else:
            print("Computed Zernike model is not saved! Set save_zernike_model parameters to True in the config_compute.yaml file :)\n")

    else: 
        print("Set one out of the two gen_gaussian_model and gen_zernike_model parameters to be true!\n")
        print("Exiting without any computing ... :(\n")
        sys.exit()


def main():
    #read config_fit.yaml file
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/config_compute.yaml'
    run(config_path)

if __name__ == "__main__":
    main()
