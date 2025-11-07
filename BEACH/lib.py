#Author: Ajith Sampath
#Affiliation: University of Geneva
#Project: BEACH - HIRAX Beam package

#lib.py file for all the helper functions

import numpy as np
from scipy.special import jn
import matplotlib.pyplot as plt
from scipy.optimize import fmin,minimize
import scipy as sp
from scipy.interpolate import interp1d
import time
import yaml
import os
import h5py 
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

def NollToQuantum(j):
    n=int(np.ceil((-3+np.sqrt(9+(8*j)))/2))
    m=int((2*j)-(n*(n+2)))
    return (n, m);

def QuantumToNoll(n,m):
    j=int((n*(n+2)+m)/2)
    return j;
    
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


#Gaussian Fit class

class GaussianFit:
    def __init__(self,datafile,freq,error_type='uniform'):
        '''Routine to load data from fits/hdf5/csv file'''
        self.freq = freq
        self.load_beam(datafile,error_type)


    def load_beam(self,datafile,error_type='uniform'):
        if datafile.endswith('.fits'):
            hdul = fits.open(datafile)
            self.data = hdul[0].data
            hdul.close()
        elif datafile.endswith('.npy'):
            self.data = np.load(datafile)
        elif datafile.endswith('.npz'):
            self.data = np.load(datafile)['fiducial'][int((self.freq-400)/50)]
            self.data[np.isnan(self.data)] = 0.0
            self.x = np.load(datafile)['x']
            self.y = np.load(datafile)['y']
            self.freq_arr = np.load(datafile)['freq']
            self.nchan = self.freq_arr.shape[0]
            if np.load(datafile).keys().__contains__('error'):
                self.error = np.load(datafile)['error']
            else:
                if error_type=='uniform':
                    self.error = np.ones_like(self.data)
                else:
                    raise ValueError("Error array not found in .npz file. Please provide error array or set error_type to 'uniform'.")

        elif datafile.endswith('.txt'):           
            self.data = np.loadtxt(datafile)
        elif datafile.endswith('.h5') or datafile.endswith('.hdf5'):
            with h5py.File(datafile, 'r') as f:
                self.data = f['dataset_name'][:]  # replace 'dataset_name' with actual dataset name
        elif datafile.endswith('.csv'):
            df = pd.read_csv(datafile)
            self.data = df.values  # assuming the entire CSV is the data
        else:
            raise ValueError("Unsupported file format. Please use .fits, .h5/.hdf5, or .csv files.")
        return self.data;
      
    def g_chisq(self,params):
        '''Routine to compute chisq for Gaussian fit'''
        self.gExpected = twoD_Gaussian(self.x,self.y,params)
        Expected = self.gExpected.flatten()
        data = self.data.flatten()
        k=len(params)
        chi = np.vdot((data - Expected)/(self.error.flatten()),(data - Expected)/(self.error.flatten()))/(len(data)-k-2)
        chi2=np.abs(chi)
        print("Current Gaussian parameters:", params, "Chisq:", chi2)
        return (chi2);

    def optimize_Gauss(self,init_gparams,minimize_method = 'Nelder-Mead',xtol=1e-8,maxiter=100,verbose=True):
        '''Routine to optimize Gaussian fit by minimizing chisq'''
        self.init_gparams = init_gparams
        print("Initial Gaussian parameters:", self.init_gparams)
        self.gopt = minimize(self.g_chisq, self.init_gparams, method=minimize_method, options={ 'maxiter': maxiter, 'disp': verbose})
        _,self.sigx_gopt,self.sigy_gopt,self.xo,self.yo,_ = self.gopt.x       
        return self.x,self.y,self.xo,self.yo,self.freq_arr,self.freq,self.sigx_gopt,self.sigy_gopt,self.gExpected,self.data,self.gopt.fun;


#Zernike Transform Fit class

class ZernikeFit:
    def __init__(self,x,y,xo,yo,freq_arr,freq,data,N,error_type='proportional',normalize_data=True):
        self.pbar = None
        self.x = x
        self.y = y
        self.xo = xo
        self.yo = yo
        self.freq_arr = freq_arr
        self.freq = freq
        self.data = data
        self.N = N
        if normalize_data:
            self.data = self.data/np.max(self.data)
            print("Data normalized to maximum value.")
        else:
            print("Data not normalized.")
            pass
        if error_type=='proportional':
            self.error = np.abs(self.data)*0.05 + 1e-3
        elif error_type=='uniform':
            self.error = np.ones_like(self.data)
        else:
            raise ValueError("Unsupported error type. Please use 'proportional' or 'uniform'.")
        

    def basis_N(self,params):
        '''Routine to generate Zernike Transforms (basis) from Bessel function of first kind to fit a data set'''
        self.sigx,self.sigy = params
        xm,ym = np.meshgrid(self.x/self.sigx,self.y/self.sigy)
        rm = np.hypot(xm,ym)
        rm[rm==0] = 1e-10
        #rho = rho/sig
        thetam = np.arctan2(ym,xm)
        self.Basis = np.zeros((int(self.N),int(len(rm.flatten()))),dtype="float32")
        count = 0
        for j in range(0,self.N):
            n,m = NollToQuantum(j)
            if n>=0 and n>=abs(m) and (n-abs(m))%2==0:
                Bes=(jn(n+1,rm))/rm
                nc = (((2*n+1)*(2*n+3)*(2*n+5))/(-1)**n)**0.5
                temp=np.real((nc*(np.exp(1j*m*thetam))/((1j**m)*2*np.pi) *(-1)**((n-m)/2) *Bes))
                self.Basis[count]=temp.flatten()
                count+=1            
    def zt_chisq(self,params):
        '''Routine to compute chisq for Zernike fit'''
        self.basis_N(params)
        w = 1/self.error.flatten()**2
        Bw = self.Basis.T*np.sqrt(w[:,np.newaxis])
        Cw = self.data.flatten()*np.sqrt(w)
        self.coef,_,_,_ = sp.linalg.lstsq(Bw,Cw)
        self.Expected = np.dot(self.Basis.T,self.coef).reshape(self.data.shape)
        Expected = self.Expected.flatten()
        data = self.data.flatten()
        k = len(self.coef)
        resid = (data - Expected) / np.sqrt(self.error.flatten())
        chi_red = np.vdot(resid, resid).real / (len(data) - k)
        chi2=np.abs(chi_red)
        return (chi2);

    def callback(self, xk):
        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_postfix({"Chi-square": f"{xk[0]:.4f}"})
        
    def optimize_ZT(self,init_ztparams,minimize_method='Nelder-Mead',xtol=1e-8,maxiter=100):
        '''Routine to optimize Zernike fit by minimizing chisq'''
        self.init_ztparams = init_ztparams
        self.pbar = tqdm(desc="Optimizing", unit="iter", dynamic_ncols=True)
        self.ztopt = minimize(self.zt_chisq, self.init_ztparams, callback=self.callback,method=minimize_method, options={ 'disp': True, 'maxiter': maxiter})
        self.sigx_ztopt,self.sigy_ztopt = self.ztopt.x         
        return self.sigx_ztopt,self.sigy_ztopt,self.coef,self.Expected,self.ztopt.fun;

    def plot_results_cart(self,data,model,freq,N,x,y,plot_format,plot_directory):
        ms=1.5
        vmin=None
        vmax=None
        residue = data - model
        extent = [x.min(), x.max(), y.min(), y.max()]
        #plot 2D cst,fit and residue and x and y Cuts
        plt.figure(figsize=(12,6))
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)
        
        z1_plot=ax1.imshow(np.log(data),cmap='inferno',vmin=vmin,vmax=vmax,extent=extent)
        ax1.grid(False)
        plt.colorbar(z1_plot,ax=ax1,fraction=0.047)
        ax1.set_title("Simulated CST beam")
        z2_plot=ax2.imshow(np.log(model),cmap='inferno',vmin=vmin,vmax=vmax,extent=extent)
        ax2.grid(False)
        plt.colorbar(z2_plot,ax=ax2,fraction=0.047)
        ax2.set_title("Fit with "+str(N)+" basis functions")
        ax2.get_yaxis().set_visible(False)
        z3_plot=ax3.imshow(residue,cmap='seismic',vmin=vmin,vmax=vmax,extent=extent)
        ax3.grid(False)
        plt.colorbar(z3_plot,ax=ax3,fraction=0.047)
        ax3.set_title("Abs Residue")
        ax3.get_yaxis().set_visible(False)
        plt.tight_layout()
        plotname = "BeamFitResults_"+str(freq)+"MHz with N="+str(N)+plot_format
        new_dir = os.path.join(os.getcwd(), plot_directory)
        os.makedirs(new_dir, exist_ok=True)
        print("Making the plot.....")
        plt.savefig(os.path.join(new_dir, plotname), bbox_inches='tight', dpi=300)
        plt.clf()


    def plot_results_polar(self,data,model,freq,N,rho,phi ,plot_format,plot_directory):
        ms=1.5
        vmin=None
        vmax=None
        residue = data - model

        #plot 2D cst,fit and residue and x and y Cuts
        plt.figure(figsize=(12,6))
        ax1 = plt.subplot(131,projection='polar')
        ax2 = plt.subplot(132,projection='polar')
        ax3 = plt.subplot(133,projection='polar')
        
        z1_plot=ax1.scatter(phi,rho,c=np.log(data),cmap='inferno',s=ms,vmin=vmin,vmax=vmax)
        ax1.grid(False)
        plt.colorbar(z1_plot,ax=ax1,fraction=0.047)
        ax1.set_title("Simulated CST beam")
        z2_plot=ax2.scatter(phi,rho,c=np.log(model),cmap='inferno',s=ms,vmin=vmin,vmax=vmax)
        ax2.grid(False)
        plt.colorbar(z2_plot,ax=ax2,fraction=0.047)
        ax2.set_title("Fit with "+str(N)+" basis functions")
        ax2.get_yaxis().set_visible(False)
        z3_plot=ax3.scatter(phi,rho,c=(residue),cmap='seismic',s=ms,vmin=vmin,vmax=vmax)
        ax3.grid(False)
        plt.colorbar(z3_plot,ax=ax3,fraction=0.047)
        ax3.set_title("Abs Residue")
        ax3.get_yaxis().set_visible(False)
        plt.tight_layout()
        plotname = "BeamFitResults_"+str(freq)+"MHz with N="+str(N)+plot_format
        new_dir = os.path.join(os.getcwd(), plot_directory)
        os.makedirs(new_dir, exist_ok=True)
        print("Making the plot.....")
        plt.savefig(os.path.join(new_dir, plotname), bbox_inches='tight', dpi=300)
        plt.clf()


#Generative Beam class

class GenBeam:
    def __init__(self,freq,x,y):
        '''Initializer to get all the parameters'''
        self.freq = freq
        self.x = x
        self.y = y    
    def load_coef(self,coeffile):
        '''Routine to load coefficients from csv file'''
        if coeffile.endswith('.csv'):
            df = pd.read_csv(coeffile)
            self.coef = df['Coefficient'].values  # assuming the entire CSV is the data
        elif coeffile.endswith('.npy'):
            self.coef = np.load(coeffile)
        else:
            raise ValueError("Unsupported file format. Please use .csv or .npy/.npz files.")
        return self.coef; 
        
    def basis_j(self,params):
        '''Routine to generate Zernike Transforms (basis) from Bessel function of first kind to generate beam from coefficients'''
        self.sigx,self.sigy = params
        xm,ym = np.meshgrid(self.x/self.sigx,self.y/self.sigy)
        rm = np.hypot(xm,ym)
        rm[rm==0] = 1e-10
        thetam = np.arctan2(ym,xm)

        self.Basis = np.zeros((len(self.coef),len(rm.flatten())),dtype="float64")
        count = 0
        for j in range(0,self.coef.shape[0]):
            n,m = NollToQuantum(j)
            if n>=0 and n>=abs(m) and (n-abs(m))%2==0:
                Bes=(jn(n+1,rm))/rm
                nc = (((2*n+1)*(2*n+3)*(2*n+5))/(-1)**n)**0.5
                temp=np.real((nc*(np.exp(1j*m*thetam))/((1j**m)*2*np.pi) *(-1)**((n-m)/2) *Bes))
                self.Basis[count]=temp.flatten()
                count+=1
        return self.Basis;





