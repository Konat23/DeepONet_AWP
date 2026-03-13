# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 09:06:17 2021

@author: Alejandra Hernandez
"""

import numpy as np
import matplotlib.pyplot as plt
from ricker import ricker
from numba import jit

Tout = 1250 # Propagation time
Nx = 400 # Number of point in the x direction
Nz = 200 # Number of point in the z direction
c = 1500 # Background velocity
dh = 20 # spatial step
dt = dh/(c*np.sqrt(2))
G = (c*dt)/dh # Courant parameter
tEnd = 5
t = np.linspace(0, tEnd-dt, Tout)
fq = 5

P1 = np.zeros((Nx, Nz)) # frente de onda pasado
P2 = np.zeros((Nx, Nz)) # frente de onda presente
P3 = np.zeros((Nx, Nz)) # frente de onda futuro

video = np.zeros((Nx, Nz, Tout))

# fuente temporal

fq = 5 #Hz
Sx = 100
Sz = 3
fuente = np.zeros(Tout)

for tstep in range(Tout):
    fuente[tstep] = ricker(tstep, dt, fq)

def propagation():
    for t in range(Tout):
        
        for i in np.arange(1, Nx-1):
            for j in np.arange(1, Nz-1):
                P3[i,j] = (2-4*G*G)*P2[i,j]+G*G*(P2[i+1,j]+P2[i-1,j]+P2[i,j+1]+P2[i,j-1])-P1[i,j]
                    
        for i in np.arange(1, Nx-1):
            for j in np.arange(1, Nz-1):
                P1[i,j] = P2[i,j]
                P2[i,j] = P3[i,j]
            #P1 = P2.copy()
            #P2 = P3.copy()
            
        P2[Sx, Sz] = P2[Sx, Sz] + fuente[t]
                    
        for i in range(Nx):
            for j in range(Nz):
                video[i,j,t] = P3[i,j]
    return video

video = propagation()

for i in range(0, video.shape[-1], 5):
    plt.imshow(video[:,:,i].T)
    plt.title('Iteration ' + str(i))
    plt.pause(0.001)