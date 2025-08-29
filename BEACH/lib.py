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
    

def basis_cart(sigx,sigy,x,y,N):
    '''Routine to generate Zernike Transforms (basis) from Bessel function of first kind'''
    x,y = np.meshgrid(x/sigx,y/sigy)
    r = np.hypot(x,y)
    r[r==0] = 1e-10
    #rho = rho/sig
    theta = np.arctan2(y,x)

    z1 = np.zeros((N,len(r.flatten())),dtype=np.complex128)

    count = 0

    for j in range(0,N):
        n,m = NollToQuantum(j)
        if n>=0 and n>=abs(m) and (n-abs(m))%2==0:
            Bes=(jn(n+1,r))/r
            nc = (((2*n+1)*(2*n+3)*(2*n+5))/(-1)**n)**0.5
            temp=((nc*(np.exp(1j*m*theta))/((1j**m)*2*np.pi) *(-1)**((n-m)/2) *Bes))
            z1[count]=temp.flatten()
            count+=1
        
    return z1;



def twoD_Gaussian(x,y,amp,sigx,sigy,xo, yo, tilt, offset):
    xo = float(xo)
    yo = float(yo)
 
    x,y = np.meshgrid(x,y)
    a = (np.cos(tilt)**2)/(2*sigx**2) + (np.sin(tilt)**2)/(2*sigy**2)
    b = -(np.sin(2*tilt))/(4*sigx**2) + (np.sin(2*tilt))/(4*sigy**2)
    c = (np.sin(tilt)**2)/(2*sigx**2) + (np.cos(tilt)**2)/(2*sigy**2)
    
    return offset + amp*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)));





