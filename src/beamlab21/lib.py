#Author: Ajith Sampath
#Affiliation: University of Geneva
#Project: HIRAX Beam package

#lib.py file for all the helper functions and classes

import numpy as np
from scipy.special import jn
import matplotlib.pyplot as plt
from scipy.optimize import fmin,minimize, OptimizeWarning
import scipy as sp
from scipy.interpolate import interp1d
import time
import yaml
import os
import h5py 
import pandas as pd
from astropy.io import fits
from tqdm import tqdm
import sys

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_config(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def NollToQuantum(j):
    n=int(np.ceil((-3+np.sqrt(9+(8*j)))/2))
    m=int((2*j)-(n*(n+2)))
    return (n, m);

def QuantumToNoll(n,m):
    j=int((n*(n+2)+m)/2)
    return j;


def find_min_full_N_for_Nprime(Nprime, NollToQuantum):
    count = 0
    j = 0
    while True:
        n, m = NollToQuantum(j)
        if n >= 0 and n >= abs(m) and (n - abs(m)) % 2 == 0 and m >= 0:
            count += 1
            if count == Nprime:
                return j + 1  # +1 because j is 0-based index
        j += 1

def twoD_Gaussian(x,y,params):
    amp,sigx,sigy,xo, yo, tilt = params
    xo = float(xo)
    yo = float(yo)
    tilt = np.radians(tilt)
    x,y = np.meshgrid(x,y)
    a = (np.cos(tilt)**2)/(2*sigx**2) + (np.sin(tilt)**2)/(2*sigy**2)
    b = -(np.sin(2*tilt))/(4*sigx**2) + (np.sin(2*tilt))/(4*sigy**2)
    c = (np.sin(tilt)**2)/(2*sigx**2) + (np.cos(tilt)**2)/(2*sigy**2)
    
    return amp*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)));

        
def reorder_coef(coef):
    
    neg_m_list = []
    for j in range(coef.shape[0]):
        [n,m]=NollToQuantum(j)
        if m<0:
            neg_m_list.append(j)
    for g in neg_m_list:
        coef=np.delete(coef, -1)
        coef=np.insert(coef,g,0.0)
    return coef;

def compute_impacts(coeffs, X):
    impacts = np.linalg.norm(X * coeffs[:, None], axis=1)
    return impacts

def load_beam(datafile):
    if datafile.endswith('.fits'):
        hdul = fits.open(datafile)
        data = hdul[0].data
        hdul.close()
    elif datafile.endswith('.npy'):
        data = np.load(datafile)
    elif datafile.endswith('.npz'):
        data = np.load(datafile)['data']
        data[np.isnan(data)] = 0.0

        x = np.load(datafile)['x']
        y = np.load(datafile)['y']
        freq_arr = (np.load(datafile)['freq'])*1e3
        nchan = freq_arr.shape[0]
        if np.load(datafile).keys().__contains__('error'):
            error = np.load(datafile)['error']
        else:
            error = None

    elif datafile.endswith('.txt'):           
        data = np.loadtxt(datafile)
    elif datafile.endswith('.h5') or datafile.endswith('.hdf5'):
        with h5py.File(datafile, 'r') as f:
            data = f['dataset_name'][:]  # replace 'dataset_name' with actual dataset name
    elif datafile.endswith('.csv'):
        df = pd.read_csv(datafile)
        data = df.values  # assuming the entire CSV is the data
    else:
        raise ValueError("Unsupported file format. Please use .fits, .h5/.hdf5, or .csv files.")
    
    return x,y,freq_arr,nchan,error,data;


#Gaussian Fit class


class GaussianFit:
    def __init__(self, datafile, freq, error_type='uniform',
                 normalize_data=True, coord_type='auto'):
        """
        Routine to load data from fits/hdf5/csv file.
        coord_type : 'cartesian', 'polar', or 'auto'
        """
        self.freq = freq
        self.x, self.y, self.freq_arr, self.nchan, self.error, self.data_cube = \
            load_beam(datafile)
        # Auto-detect coordinate system if requested
        if coord_type == 'auto':
            # Heuristic: if range of x ~ [0, 2π] and min(y) >= 0 => polar (r, θ)
            if np.all(self.x >= 0) and np.all((self.y >= 0) & (self.y <= 2*np.pi)):
                coord_type = 'polar'
            else:
                coord_type = 'cartesian'
        self.coord_type = coord_type

        # Convert polar → Cartesian for fitting, if needed
        if self.coord_type == 'polar':
            r, theta = self.x, self.y
            self.x = r * np.cos(theta)
            self.y = r * np.sin(theta)

        dfreq = np.abs(self.freq_arr[1] - self.freq_arr[0])
        self.data = self.data_cube[int((freq - 400) / dfreq)]
        if normalize_data:
            self.data = self.data/np.max(self.data)
        else:
            self.data = self.data  
        if self.error is None:
            if error_type == 'uniform':
                self.error = np.ones_like(self.data)
                print("Proceeding with uniform error array...\n")
            else:
                raise ValueError(
                    "Error array not found. Provide error array or set error_type='uniform'.\n"
                )

    def callback(self, xk):
        if hasattr(self, "pbar") and self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_postfix({"Chi-square": f"{xk[0]:.4f}"})

    def g_chisq(self, params):
        """Routine to compute chisq for Gaussian fit"""
        self.gExpected = twoD_Gaussian(self.x, self.y, params)
        Expected = self.gExpected.flatten()
        data = self.data.flatten()
        k = len(params)
        chi = np.vdot((data - Expected) / self.error.flatten(),
                      (data - Expected) / self.error.flatten()) / (len(data) - k - 2)
        return np.abs(chi)

    def optimize_Gauss(self, init_gparams, minimize_method='Nelder-Mead',
                       xtol=1e-8, verbose=True):
        """Optimize Gaussian fit by minimizing chisq"""
        self.init_gparams = init_gparams
        print("Initial Gaussian parameters:", self.init_gparams)
        self.pbar = tqdm(desc="Optimizing", unit="iter", dynamic_ncols=True)
        self.gopt = minimize(self.g_chisq, self.init_gparams,
                             callback=self.callback,
                             method=minimize_method,
                             options={'disp': verbose})
        self.pbar.close()
        # Ignore "max iterations exceeded" silently
        if getattr(self.gopt, "status", None) == 1:
            pass

        _, self.sigx_gopt, self.sigy_gopt, self.xo, self.yo, _ = self.gopt.x
        return (self.x, self.y, self.xo, self.yo, self.freq_arr,
                self.freq, self.sigx_gopt, self.sigy_gopt,
                self.gExpected, self.data, self.gopt.fun)
    

#Zernike Transform fit
class ZernikeFit:
    def __init__(self, x, y, xo, yo, freq_arr, freq, data, N,
                 error_type='proportional', normalize_data=True,
                 coord_type='auto'):
        """
        Parameters:
        -----------
        x, y : array-like
            Coordinates. Interpreted as (x,y) if coord_type='cartesian',
            or as (r, θ) if coord_type='polar'.
        coord_type : 'auto' (default), 'cartesian', or 'polar'
            Coordinate system type.
        """
        self.pbar = None
        self.coord_type = coord_type
        self.project_root=get_project_root()

        # Auto-detect coordinate system if needed
        if self.coord_type == 'auto':
            if np.all(x >= 0) and np.all((y >= 0) & (y <= 2 * np.pi)):
                self.coord_type = 'polar'
            else:
                self.coord_type = 'cartesian'

        if self.coord_type == 'polar':
            self.r = x
            self.theta = y
            self.x = None
            self.y = None
        else:
            self.x = x
            self.y = y
            self.r = None
            self.theta = None

        self.xo = xo
        self.yo = yo
        self.freq_arr = freq_arr
        self.freq = freq
        self.data = data
        self.N = N
        self.N_full = find_min_full_N_for_Nprime(self.N,NollToQuantum)

        if normalize_data:
            self.data = self.data / np.max(self.data)
            print("Data normalized to maximum value.\n")
        else:
            print("Data not normalized.\n")

        if error_type == 'proportional':
            self.error = np.abs(self.data) * 0.1
        elif error_type == 'uniform':
            self.error = np.ones_like(self.data)
        else:
            raise ValueError("Unsupported error type. Use 'proportional' or 'uniform'.\n")

    def basis_N(self, params):
        """Generate Zernike basis from Bessel functions to fit the data."""
        self.sigx, self.sigy = params

        if self.coord_type == 'cartesian':
            xm, ym = np.meshgrid(self.x / self.sigx, self.y / self.sigy)
            rm = np.hypot(xm, ym)
            rm[rm == 0] = 1e-10
            thetam = np.arctan2(ym, xm)

        elif self.coord_type == 'polar':
            # Scale radius and keep angle as is
            rm = self.r / self.sigx  # adjust scaling if needed
            thetam = self.theta
            rm[rm == 0] = 1e-10

        self.Basis = np.zeros((int(self.N_full), int(len(rm.flatten()))), dtype="float32")
        count = 0
        print(f"Constructing the full basis set for the given N={self.N} and scaling parameters = [{np.round(self.sigx,2),np.round(self.sigy,2)}]...")
        with tqdm(total=100, bar_format='{l_bar}{bar}| [{elapsed}] {postfix}') as pbar:
            for j in range(0, self.N_full):
                n, m = NollToQuantum(j)
                if n >= 0 and n >= abs(m) and (n - abs(m)) % 2 == 0 and m >= 0:
                    Bes = (jn(n + 1, rm)) / rm
                    nc = np.abs(((2 * n + 1) * (2 * n + 3) * (2 * n + 5)) / (-1) ** n) ** 0.5
                    temp = np.real(
                        (nc * (np.exp(1j * m * thetam)) / ((1j ** m) * 2 * np.pi) * (-1) ** ((n - m) // 2) * Bes)
                    )
                    self.Basis[count] = temp.flatten()
                    count += 1
                pbar.update(100 / self.N_full)
                pct = round(pbar.n, 1)
                pbar.set_postfix_str(f'{pct}%')

    def zt_chisq(self, params):
        """Compute chi-squared for Zernike fit."""
        self.basis_N(params)
        w = 1 / self.error.flatten() ** 2
        Bw = self.Basis.T * np.sqrt(w[:, np.newaxis])
        Cw = self.data.flatten() * np.sqrt(w)
        self.coef, _, _, _ = sp.linalg.lstsq(Bw, Cw)
        self.Expected = np.dot(self.Basis.T, self.coef).reshape(self.data.shape)
        Expected = self.Expected.flatten()
        data = self.data.flatten()
        k = len(self.coef)
        resid = (data - Expected) / np.sqrt(self.error.flatten())
        chi_red = np.vdot(resid, resid).real / (len(data) - k)
        return np.abs(chi_red)

    def callback(self, xk):
        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_postfix({"Chi-square": f"{xk[0]:.4f}"})

    def optimize_ZT(self, init_ztparams, minimize_method='Nelder-Mead', xtol=1e-8, maxiter=100):
        """Optimize Zernike fit by minimizing chi-squared."""
        self.init_ztparams = init_ztparams
        print("Is this a HPC machine? If so, good choice :). If not run this script with 'skip_minimize' set to True.\n")
        self.pbar = tqdm(desc="Optimizing", unit="iter", dynamic_ncols=True)
        self.ztopt = minimize(self.zt_chisq, self.init_ztparams,
                              callback=self.callback,
                              method=minimize_method,
                              options={'disp': True, 'maxiter': maxiter, 'xtol': xtol})
        self.pbar.close() 
        self.sigx_ztopt, self.sigy_ztopt = self.ztopt.x
        return self.sigx_ztopt, self.sigy_ztopt, self.coef, self.Expected, self.ztopt.fun

    def NO_optimize_ZT(self, init_ztparams,fac):
        """Estimate ZT scaling parameter skipping optimization."""
        print("Skipping scaling parameter optimization for Zernike basis.....\n")
        print("Good choice if you are running in a laptop :) \n")
        self.init_ztparams = [init_ztparams[0] / fac, init_ztparams[1] / fac]
        self.zt_chisq(self.init_ztparams)
        print("No optimization done for scaling parameter... and the ZT model is constructed using Gaussian sigma!!\n")
        return self.init_ztparams[0], self.init_ztparams[1], self.coef, np.abs(self.Expected)

    def plot_results_cart(self, data, model, freq, N, x, y, plot_format, plot_directory,plot_cmap):
        ms = 1.5
        vmin = None
        vmax = None
        residue = data - model
        extent = [x.min(), x.max(), y.min(), y.max()]
        plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)

        z1_plot = ax1.imshow(np.log(data), cmap=plot_cmap, vmin=vmin, vmax=vmax, extent=extent)
        ax1.grid(False)
        plt.colorbar(z1_plot, ax=ax1, fraction=0.047)
        ax1.set_title("Simulated CST beam")

        z2_plot = ax2.imshow(np.log(model), cmap=plot_cmap, vmin=vmin, vmax=vmax, extent=extent)
        ax2.grid(False)
        plt.colorbar(z2_plot, ax=ax2, fraction=0.047)
        ax2.set_title(f"Fit with {N} basis functions")
        ax2.get_yaxis().set_visible(False)

        z3_plot = ax3.imshow((residue/model)*100, cmap='seismic', vmin=vmin, vmax=vmax, extent=extent)
        ax3.grid(False)
        plt.colorbar(z3_plot, ax=ax3, fraction=0.047)
        ax3.set_title("Percentage Residuals")
        ax3.get_yaxis().set_visible(False)

        plt.tight_layout()
        plotname = f"ZernikeFitResults_{freq}MHz with N={N}{plot_format}"
        print("Making the plot.....\n")
        if not os.path.exists(os.path.join(self.project_root,plot_directory)):
            os.makedirs(os.path.join(self.project_root,plot_directory))
        plt.savefig(os.path.join(self.project_root, plot_directory, plotname), bbox_inches='tight', dpi=300)
        plt.clf()
        plt.close('all')


    def plot_results_polar(self, data, model, freq, N, rho, phi, plot_format, plot_directory):
        ms = 1.5
        vmin = None
        vmax = None
        residue = data - model
        plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(131, projection='polar')
        ax2 = plt.subplot(132, projection='polar')
        ax3 = plt.subplot(133, projection='polar')

        z1_plot = ax1.scatter(phi, rho, c=np.log(data), cmap='inferno', s=ms, vmin=vmin, vmax=vmax)
        ax1.grid(False)
        plt.colorbar(z1_plot, ax=ax1, fraction=0.047)
        ax1.set_title("Simulated CST beam")

        z2_plot = ax2.scatter(phi, rho, c=np.log(model), cmap='inferno', s=ms, vmin=vmin, vmax=vmax)
        ax2.grid(False)
        plt.colorbar(z2_plot, ax=ax2, fraction=0.047)
        ax2.set_title(f"Fit with {N} basis functions")
        ax2.get_yaxis().set_visible(False)

        z3_plot = ax3.scatter(phi, rho, c=(residue/model)*100, cmap='seismic', s=ms, vmin=vmin, vmax=vmax)
        ax3.grid(False)
        plt.colorbar(z3_plot, ax=ax3, fraction=0.047)
        ax3.set_title("Percentage Residual")
        ax3.get_yaxis().set_visible(False)

        plt.tight_layout()
        
        plotname = f"ZernikeFitResults_{freq}MHz with N={N}{plot_format}"
        print("Making the plot.....\n")
        if not os.path.exists(os.path.join(self.project_root,plot_directory)):
            os.makedirs(os.path.join(self.project_root,plot_directory))
        plt.savefig(os.path.join(self.project_root, plot_directory, plotname), bbox_inches='tight', dpi=300)
        plt.clf()
        plt.close('all')


    def process_and_reduce_coefficients(self, output_dir, coef_name, reduce=False, percentage_energy=100.0):
        coef_reordered = reorder_coef(self.coef)

        j = np.arange(len(coef_reordered))
        vectorized_NollToQuantum = np.vectorize(NollToQuantum)

        #generate quantum indices
        n_val,m_val = vectorized_NollToQuantum(j)

        coef_jnm = np.column_stack((j, n_val, m_val, coef_reordered))

        if reduce:
            impacts = np.linalg.norm(self.Basis * coef_reordered[:, None], axis=1)
            sorted_idx = np.argsort(impacts)[::-1]
            sorted_impacts = impacts[sorted_idx]
            cumulative = np.cumsum(sorted_impacts) / np.sum(sorted_impacts)
            target_fraction = percentage_energy / 100.0
            num_keep = np.searchsorted(cumulative, target_fraction) + 1
            keep_idx = sorted_idx[:num_keep]

            coef_jnm = coef_jnm[keep_idx]
            keep_mask = np.zeros(len(coef_reordered), dtype=bool)
            keep_mask[keep_idx] = True
            # Save reduced coefficients CSV
            if not os.path.exists(os.path.join(self.project_root,output_dir)):
                os.makedirs(os.path.join(self.project_root,output_dir))
            reduced_path = os.path.join(self.project_root, output_dir, 'coef_reduced.csv')
            os.makedirs(os.path.dirname(reduced_path), exist_ok=True)
            df_reduced = pd.DataFrame(coef_jnm, columns=['j', 'n', 'm', 'coef'])
            df_reduced.to_csv(reduced_path, index=False)

        else:
            # Save full coefficients CSV
            full_path = os.path.join(self.project_root, output_dir, coef_name)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            df = pd.DataFrame({"coef_full": self.coef})
            df.to_csv(full_path, index=False)
            keep_mask = np.ones(len(coef_reordered), dtype=bool)

        return None


#Generative Beam class

class GenZTBeam:
    def __init__(self,freq,x,y,dtype):
        '''Initializer to get all the parameters'''
        self.freq = freq
        self.x = x
        self.y = y
        self.dtype = dtype

    def load_coef(self,coeffile):
        '''Routine to load coefficients from csv file'''
        if coeffile.endswith('.csv'):
            df = pd.read_csv(coeffile)
            self.coef = df['coef'].values  # assuming the entire CSV is the data
            self.j = (df['j'].to_numpy()).astype(int)
            self.n = (df['n'].to_numpy()).astype(int)
            self.m = (df['m'].to_numpy()).astype(int)
        else:
            raise ValueError("Unsupported file format. Please use .csv files for Coefficients.\n")
        
        return None; 
        
    def basisfunc(self,sigx,sigy):
        '''Routine to generate Zernike Transforms (basis) from Bessel function of first kind to generate beam from coefficients'''
        self.sigx,self.sigy = sigx,sigy
        xm,ym = np.meshgrid(self.x/self.sigx,self.y/self.sigy)
        rm = np.hypot(xm,ym)
        rm[rm==0] = 1e-10
        thetam = np.arctan2(ym,xm)

        self.Basis = np.zeros((len(self.coef),len(rm.flatten())),dtype=self.dtype)
        print(f"Constructing the basis set for the given coefficients and scale parameters...")
        with tqdm(total=100, bar_format='{l_bar}{bar}| [{elapsed}] {postfix}') as pbar:
            for id in range(self.coef.shape[0]):
                n=self.n[id]
                m=self.m[id]
                Bes=(jn(n+1,rm))/rm
                nc = np.abs((((2*n+1)*(2*n+3)*(2*n+5))/(-1)**n))**0.5
                #print(nc)
                temp=np.real((nc*(np.exp(1j*m*thetam))/((1j**m)*2*np.pi) *(-1)**((n-m)/2) *Bes))
                #temp=temp/np.max(temp)
                self.Basis[id]=temp.flatten()
                pbar.update(100 / self.coef.shape[0])
                pct = round(pbar.n, 1)
                pbar.set_postfix_str(f'{pct}%')
        return self.Basis;