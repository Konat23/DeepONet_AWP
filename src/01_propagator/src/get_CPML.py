# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 14:52:09 2021

@author: Alejandra Hernandez
"""

import numpy as np
from numba import jit

@jit
def get_CPML(CPMLimit, R, Vcpml, Nx, Nz, dx, dz, dt, frec):
    D_pml_x = CPMLimit*dx
    D_pml_z = CPMLimit*dz
    
    d0_x = -3/(2*D_pml_x)*np.log(R)
    d0_z = -3/(2*D_pml_z)*np.log(R)
    
    x = np.zeros((CPMLimit+1))
    z = np.zeros((CPMLimit+1))
    alpha_x = np.zeros((CPMLimit+1))
    alpha_z = np.zeros((CPMLimit+1))
    x_half = np.zeros((CPMLimit+1))
    z_half = np.zeros((CPMLimit+1))
    alpha_x_half = np.zeros((CPMLimit+1))
    alpha_z_half = np.zeros((CPMLimit+1))
    
    for j in range(CPMLimit+1):
        x[j] = (CPMLimit - j )*dx
        z[j] = (CPMLimit - j )*dz
        alpha_x[j] = np.pi*frec*(D_pml_x - x[j])/D_pml_x
        alpha_z[j] = np.pi*frec*(D_pml_z - z[j])/D_pml_z
        x_half[j] = (CPMLimit - j )*dx - dx/2
        z_half[j] = (CPMLimit - j )*dz - dz/2
        alpha_x_half[j] = np.pi*frec*(D_pml_x - x_half[j])/D_pml_x
        alpha_z_half[j] = np.pi*frec*(D_pml_z - z_half[j])/D_pml_z
    
    d_z = np.zeros((Nz))
    b_z = np.zeros((Nz))
    a_z = np.zeros((Nz))
    d_z_half = np.zeros((Nz))
    b_z_half = np.zeros((Nz))
    a_z_half = np.zeros((Nz))
    
    # Bottom side
    for j in range(Nz-CPMLimit-1, Nz):
         d_z[j] = d0_z*Vcpml*((z[Nz-j-1]/D_pml_z)**2)
         b_z[j] = np.exp(-(d_z[j] + alpha_z[Nz-j-1])*dt)
         a_z[j] = d_z[j]/(d_z[j] + alpha_z[Nz-j-1])*(b_z[j]-1)
         if j + 1== Nz-CPMLimit:
              d_z_half[j] = 0
              b_z_half[j] = 0
              a_z_half[j] = 0     
         else:
             d_z_half[j] = d0_z*Vcpml*((z_half[Nz-j-1]/D_pml_z)**2)
             b_z_half[j] = np.exp(-(d_z_half[j] + alpha_z_half[Nz-j-1])*dt)
             a_z_half[j] = d_z_half[j]/(d_z_half[j] + alpha_z_half[Nz-j-1])*(b_z_half[j]-1)
    
    d_x = np.zeros((Nx))
    b_x = np.zeros((Nx))
    a_x = np.zeros((Nx))
    d_x_half = np.zeros((Nx))
    b_x_half = np.zeros((Nx))
    a_x_half = np.zeros((Nx))
             
    # left side
    for i in range(CPMLimit+1):
        d_x[i] = d0_x*Vcpml*((x[i]/D_pml_x)**2)
        b_x[i] = np.exp(-(d_x[i] + alpha_x[i])*dt)
        a_x[i] = d_x[i]/(d_x[i] + alpha_x[i])*(b_x[i] - 1)
        if i + 1 == CPMLimit+1:
            d_x_half[i] = 0
            b_x_half[i] = 0
            a_x_half[i] = 0
        else:
            d_x_half[i] = d0_x*Vcpml*((x_half[i]/D_pml_x)**2)
            b_x_half[i] = np.exp(-(d_x_half[i] + alpha_x_half[i])*dt)
            a_x_half[i] = d_x_half[i]/(d_x_half[i] + alpha_x_half[i])*(b_x_half[i] - 1)
            
    # right side
    
    for i in range(Nx-CPMLimit-1, Nx):
        d_x[i] = d0_x*Vcpml*((x[Nx-i-1]/D_pml_x)**2)
        b_x[i] = np.exp(-(d_x[i] + alpha_x[Nx-i-1])*dt)
        a_x[i] = d_x[i]/(d_x[i] + alpha_x[Nx-i-1])*(b_x[i] - 1)
        if i + 1== Nx-CPMLimit:
            d_x_half[i] = 0
            b_x_half[i] = 0
            a_x_half[i] = 0
        else:
            d_x_half[i] = d0_x*Vcpml*((x_half[Nx-i-1]/D_pml_x)**2)
            b_x_half[i] = np.exp(-(d_x_half[i] + alpha_x_half[Nx-i-1])*dt)
            a_x_half[i] = d_x_half[i]/(d_x_half[i] + alpha_x_half[Nx-i-1])*(b_x_half[i] - 1)
            
    return a_x, a_x_half, b_x, b_x_half, a_z, a_z_half, b_z, b_z_half