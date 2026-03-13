# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 13:17:05 2021

@author: Alejandra Hernandez
"""

import numpy as np
from propagator import propagator
import matplotlib.pyplot as plt

# define parameters

dt = 0.004 
tEnd = 5
Nt = 1250
t = np.linspace(0, tEnd-dt, Nt)
Nz = 200 
Nx = 400
vp = 1500*np.ones((Nx,Nz))
dz = 20
dx = dz
Ix0 = 100
Iz0 = 5
frec = 5

# source

a = (np.pi*5)**2
t0 = 1
g = (-2*a*(t-t0)*np.exp(-a*(t-t0)**2)).T

# compute numeral wave propagation

Pt, P, d2P_dt2 = propagator(vp, g, Ix0, Iz0, dx, dz, dt, 10e3, frec)

# wave propagation movie

for i in range(0, P.shape[-1], 50):
    plt.imshow(P[:,:,i].T)
    plt.title('Iteration ' + str(i))
    plt.pause(0.001)