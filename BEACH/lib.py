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

def basis_fit_N(params,x,y,N):
    '''Routine to generate Zernike Transforms (basis) from Bessel function of first kind to fit a data set'''
    sigx,sigy = params
    x,y = np.meshgrid(x/sigx,y/sigy)
    r = np.hypot(x,y)
    r[r==0] = 1e-10
    #rho = rho/sig
    theta = np.arctan2(y,x)

    z1 = np.zeros((N,len(r.flatten())),dtype="float64")
    count = 0
    for j in range(0,N):
        n,m = NollToQuantum(j)
        if n>=0 and n>=abs(m) and (n-abs(m))%2==0:
            Bes=(jn(n+1,r))/r
            nc = (((2*n+1)*(2*n+3)*(2*n+5))/(-1)**n)**0.5
            temp=np.real((nc*(np.exp(1j*m*theta))/((1j**m)*2*np.pi) *(-1)**((n-m)/2) *Bes))
            z1[count]=temp.flatten()
            count+=1
    return z1;

def basis_gen_j(params,x,y,coef):
    '''Routine to generate Zernike Transforms (basis) from Bessel function of first kind to generate beam from coefficients'''
    sigx,sigy = params
    x,y = np.meshgrid(x/sigx,y/sigy)
    r = np.hypot(x,y)
    r[r==0] = 1e-10
    theta = np.arctan2(y,x)
    z1 = np.zeros((len(coef),len(r.flatten())),dtype="float64")
    count = 0
    for j in coef[:,0]:
        n,m = NollToQuantum(j)
        if n>=0 and n>=abs(m) and (n-abs(m))%2==0:
            Bes=(jn(n+1,r))/r
            nc = (((2*n+1)*(2*n+3)*(2*n+5))/(-1)**n)**0.5
            temp=np.real((nc*(np.exp(1j*m*theta))/((1j**m)*2*np.pi) *(-1)**((n-m)/2) *Bes))
            z1[count]=temp.flatten()
            count+=1
        
    return z1;


def chisq_ZT(Basis,Observed,error,params,N,obs_frac):
    sigx,sigy = params
    Basis = basis_fit_N(params,x,y,N)
    w = 1/((obs_frac*Observed)+10**-5)
    Bw = Basis*np.sqrt(w[:,np.newaxis])
    Cw = Observed*np.sqrt(w)
    coef,_,_,_ = sp.linalg.lstsq(Bw,Cw)
    Expected = np.dot(Basis,coef)
    chi = np.vdot((Observed - Expected)/(error),(Observed - Expected)/(error))/(len(Observed) -len(coef)-2)
    chi2=np.abs(chi)
    return (chi2);
 
def chisq_Gauss(params,x,y,Observed,error):
    amp,sigx,sigy,xo,yo,tilt,offset = params
    Expected = twoD_Gaussian(x,y,amp,sigx,sigy,xo,yo,tilt,offset).flatten()
    chi = np.vdot((Observed - Expected)/(error),(Observed - Expected)/(error))/(len(Observed) -9)
    chi2=np.abs(chi)
    return (chi2);