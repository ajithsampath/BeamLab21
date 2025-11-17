#Author: Ajith Sampath
#Affiliation: University of Geneva

#Compute HIRAX beam from Zernike basis functions using coefficients from csv file or from Gaussian parameters in yaml file
#Return HIRAX beam model

from beamlab21.lib import *
from beamlab21 import ROOT_DIR

#read config_compute.yaml file
if len(sys.argv) < 2:
    print("Usage: python -m beamlab21.fit <config_path>")
    sys.exit(1)

config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/config_compute.yaml'
config = load_config(config_path)

print("Loaded config:", config)

freq = config['freq']
c = 3e8
wvl = c/(freq*1e6)
Deff = 4.6

angextent = config['pixels']*config['angular_res']
x = np.linspace(-angextent/2,angextent/2,config['pixels'])
y = x
dtype = config['dtype']
scaleparamfile = os.path.join(ROOT_DIR,config['scaleparam_dir'],config['scaleparam_file']) 
coeffile = os.path.join(ROOT_DIR,config['coef_dir'],config['coef_file']) 


if config['gen_gaussian_model'] and config['gen_zernike_model']:
    print("Set either gen_gaussian_model or gen_zernike_model to be true not both!!\n")
    print("Exiting for now...! Hope you reset the parameters and come back :) \n")
    exit
elif config['gen_gaussian_model']:
    amp = config['gaussian_amp']
    lambdabyDcoef = config['gaussian_lamdabyD_coef']
    sigx = np.rad2deg(lambdabyDcoef*(wvl/Deff))
    sigy = sigx
    print(sigx,sigy)
    xo,yo = config['gaussian_offset_x'],config['gaussian_offset_y']
    tilt = config['gaussian_rotation']
    gparams = amp, sigx,sigy,xo,yo,tilt
    gmodel = twoD_Gaussian(x,y,gparams)

    if config['add_noise2gaussian']:
        np.random.seed(config['gaussian_noise_random_seed']) 
        gnoise = np.random.normal(loc=config['gaussian_noise_mean'], scale=config['gaussian_noise_sigma'], size=gmodel.shape)
    else:
        gnoise = 0.0

    gmodel = (gmodel + gnoise).reshape(config['pixels'],config['pixels'])

    np.savez(f"GaussianMainLobeModel_{freq}MHz.npz",x=x,y=y,data=gmodel)
    plt.imshow(gmodel)
    plt.savefig("testgauss.png") 

elif config['gen_zernike_model']:
    ztgen = GenZTBeam(freq,x,y,dtype)
    ztsp_df = pd.read_csv(scaleparamfile)
    sigx,sigy = ztsp_df['sigx'].to_numpy(),ztsp_df['sigy'].to_numpy()
    ztgen.load_coef(coeffile)
    ztgen.basisfunc(sigx,sigy)
    ztmodel = np.dot(ztgen.Basis.T, ztgen.coef)
    plt.imshow(np.log10(ztmodel.reshape(256,256)))
    plt.savefig("testzt.png")     


else: 
    print("Set one out of the two gen_gaussian_model and gen_zernike_model parameters to be true!\n")
    print("Exiting without any computing ... :(\n")
    exit