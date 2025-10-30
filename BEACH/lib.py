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
            self.Observed = hdul[0].data
            hdul.close()
        elif datafile.endswith('.npy'):
            self.Observed = np.load(datafile)
        elif datafile.endswith('.npz'):
            self.Observed = np.load(datafile)['fiducial'][int((self.freq-400)/50)]
            self.Observed[np.isnan(self.Observed)] = 0.0
            self.x = np.load(datafile)['x']
            self.y = np.load(datafile)['y']
            self.freq_arr = np.load(datafile)['freq']
            self.nchan = self.freq_arr.shape[0]
            if np.load(datafile).keys().__contains__('error'):
                self.error = np.load(datafile)['error']
            else:
                if error_type=='uniform':
                    self.error = np.ones_like(self.Observed)
                else:
                    raise ValueError("Error array not found in .npz file. Please provide error array or set error_type to 'uniform'.")

        elif datafile.endswith('.txt'):           
            self.Observed = np.loadtxt(datafile)
        elif datafile.endswith('.h5') or datafile.endswith('.hdf5'):
            with h5py.File(datafile, 'r') as f:
                self.Observed = f['dataset_name'][:]  # replace 'dataset_name' with actual dataset name
        elif datafile.endswith('.csv'):
            df = pd.read_csv(datafile)
            self.Observed = df.values  # assuming the entire CSV is the data
        else:
            raise ValueError("Unsupported file format. Please use .fits, .h5/.hdf5, or .csv files.")
        return self.Observed;
      
    def g_chisq(self,params):
        '''Routine to compute chisq for Gaussian fit'''
        self.gExpected = twoD_Gaussian(self.x,self.y,params)
        Expected = self.gExpected.flatten()
        Observed = self.Observed.flatten()
        k=len(params)
        chi = np.vdot((Observed - Expected)/(self.error.flatten()),(Observed - Expected)/(self.error.flatten()))/(len(Observed)-k-2)
        chi2=np.abs(chi)
        print("Current Gaussian parameters:", params, "Chisq:", chi2)
        return (chi2);

    def optimize_Gauss(self,init_gparams,minimize_method = 'Nelder-Mead',xtol=1e-8,maxiter=100,verbose=True):
        '''Routine to optimize Gaussian fit by minimizing chisq'''
        self.init_gparams = init_gparams
        print("Initial Gaussian parameters:", self.init_gparams)
        self.gopt = minimize(self.g_chisq, self.init_gparams, method=minimize_method, options={ 'maxiter': maxiter, 'disp': verbose})
        _,self.sigx_gopt,self.sigy_gopt,self.xo,self.yo,_ = self.gopt.x       
        return self.x,self.y,self.xo,self.yo,self.freq_arr,self.freq,self.sigx_gopt,self.sigy_gopt,self.gExpected,self.Observed,self.gopt.fun;


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
        self.Observed = data
        self.N = N
        if normalize_data:
            self.Observed = self.Observed/np.max(self.Observed)
            print("Data normalized to maximum value.")
        else:
            print("Data not normalized.")
            pass
        if error_type=='proportional':
            self.error = np.abs(self.Observed)*0.05 + 1e-3
        elif error_type=='uniform':
            self.error = np.ones_like(self.Observed)
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
        Cw = self.Observed.flatten()*np.sqrt(w)
        self.coef,_,_,_ = sp.linalg.lstsq(Bw,Cw)
        self.Expected = np.dot(self.Basis.T,self.coef).reshape(self.Observed.shape)
        Expected = self.Expected.flatten()
        Observed = self.Observed.flatten()
        k = len(self.coef)
        resid = (Observed - Expected) / np.sqrt(self.error.flatten())
        chi_red = np.vdot(resid, resid).real / (len(Observed) - k)
        chi2=np.abs(chi_red)
        return (chi2);

    def callback(self, xk):
        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_postfix({"x": f"{xk[0]:.4f}"})
        
    def optimize_ZT(self,init_ztparams,minimize_method='Nelder-Mead',xtol=1e-8,maxiter=100):
        '''Routine to optimize Zernike fit by minimizing chisq'''
        self.init_ztparams = init_ztparams
        self.pbar = tqdm(desc="Optimizing", unit="iter", dynamic_ncols=True)
        self.ztopt = minimize(self.zt_chisq, self.init_ztparams, callback=self.callback,method=minimize_method, options={ 'disp': True, 'maxiter': maxiter})
        self.sigx_ztopt,self.sigy_ztopt = self.ztopt.x         
        return self.sigx_ztopt,self.sigy_ztopt,self.coef,self.Expected,self.ztopt.fun;



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
        elif coeffile.endswith('.npz'):
            self.coef = np.load(coeffile)['arr_0']
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





