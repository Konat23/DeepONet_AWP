# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 09:08:06 2021

@author: Alejandra Hernandez
"""

import numpy as np

def ricker(tstep, dt, fq):
    t = dt*tstep - 1/fq
    f = (1-2*(np.pi*fq*t)**2)*np.exp(-np.pi*np.pi*(fq*t)**2)
    return f
