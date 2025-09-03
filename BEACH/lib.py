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



def NollToQuantum(j):
    n=int(np.ceil((-3+np.sqrt(9+(8*j)))/2))

    m=int((2*j)-(n*(n+2)))

    return (n, m);


def QuantumToNoll(n,m):
    
    j=int((n*(n+2)+m)/2)
    return j;
    
def twoD_Gaussian(x,y,amp,sigx,sigy,xo, yo, tilt, offset):
    xo = float(xo)
    yo = float(yo)
    tilt = np.radians(tilt)
    x,y = np.meshgrid(x,y)
    a = (np.cos(tilt)**2)/(2*sigx**2) + (np.sin(tilt)**2)/(2*sigy**2)
    b = -(np.sin(2*tilt))/(4*sigx**2) + (np.sin(2*tilt))/(4*sigy**2)
    c = (np.sin(tilt)**2)/(2*sigx**2) + (np.cos(tilt)**2)/(2*sigy**2)
    
    return offset + amp*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)));

def chisq(Observed,Expected,error,k):
    Expected = Expected.flatten()
    Observed = Observed.flatten()
    chi = np.vdot((Observed - Expected)/(error),(Observed - Expected)/(error))/(len(Observed) -k-2)
    chi2=np.abs(chi)
    return (chi2);



class FitBeam:
    def __init__(self,freq,x,y,Observed,error,obs_frac,N):
        self.freq = freq
        self.x = x
        self.y = y
        self.Observed = Observed
        self.error = error
        self.obs_frac = obs_frac
        self.N = N
      
    def basis_N(self,params):
        '''Routine to generate Zernike Transforms (basis) from Bessel function of first kind to fit a data set'''
        self.sigx,self.sigy = params
        xm,ym = np.meshgrid(self.x/self.sigx,self.y/self.sigy)
        rm = np.hypot(xm,ym)
        rm[rm==0] = 1e-10
        #rho = rho/sig
        thetam = np.arctan2(ym,xm)

        self.Basis = np.zeros((self.N,len(rm.flatten())),dtype="float64")
        count = 0
        for j in range(0,self.N):
            n,m = NollToQuantum(j)
            if n>=0 and n>=abs(m) and (n-abs(m))%2==0:
                Bes=(jn(n+1,rm))/rm
                nc = (((2*n+1)*(2*n+3)*(2*n+5))/(-1)**n)**0.5
                temp=np.real((nc*(np.exp(1j*m*thetam))/((1j**m)*2*np.pi) *(-1)**((n-m)/2) *Bes))
                self.Basis[count]=temp.flatten()
                count+=1
        return self.Basis;

    def optimize_ZT(self,init_params):
        '''Routine to optimize Zernike fit by minimizing chisq'''
        self.init_params = init_params
        w = 1/self.error
        Bw = self.Basis*np.sqrt(w[:,np.newaxis])
        Cw = self.Observed*np.sqrt(w)
        self.coef,_,_,_ = sp.linalg.lstsq(Bw,Cw)
        self.Expected = np.dot(self.Basis.T,self.coef).reshape(self.Observed.shape)
        self.opt = minimize(chisq, self.init_params, args=(self.Observed,self.Expected,self.error,len(self.coef)), method='Nelder-Mead', options={'xatol': 1e-8, 'disp': True})
        self.sigx_opt,self.sigy_opt = self.opt.x         
        return self.sigx_opt,self.sigy_opt,self.coef,self.Expected,self.opt.fun;


class GenBeam:
    def __init__(self,freq,x,y,coef):
        self.freq = freq
        self.x = x
        self.y = y
        self.coef = coef
        
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





