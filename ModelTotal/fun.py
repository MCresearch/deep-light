# !/usr/bin/env python
# -*- encoding: utf-8 -*- #
 
# ----------------------------------------------------------------------------------------------------------
# File Name:        fun.py
# Author:           Xianyuer
# Version:          1.0
# Created:          2023/03/22 19:07:52
# Description:      Main Function: Functions used for propagation calculation
#                   Cross Reference: NONE
# Function List:    prop1(n_grid,n1,dz,kp,aa,h) -- Calculate the corresponding coefficient in the inverse 
#                                                  space
#                   evol1(n_grid, h, img0_)     -- Propagation: Multiply the amplitude and phase
#                   fftt(img0)                  -- FFT for 2D signal
#                   fftt2(img0)                 -- Inverse FFT for 2D signal
# Input List:
#         <name>       <type>        <description>
#         __NONE__     ---           ---
# Output List:
#         <name>       <type>        <description>
#         __NONE__     ---           ---
# History: 
#         <author>     <version>     <time>			<description>
#         Xianyuer     1.0           2023/03/22		Creat the file
# ----------------------------------------------------------------------------------------------------------

#------------ Load Package ------------#

#import numba
#from numba import jit
import numpy as np
import torch


#------------ Define Functions ------------#

# Calculate the corresponding coefficient in the inverse space
def prop1(n_grid,n1,dz,kp,aa,h):
    tt = 0.0
    t0 = 0.0
    j1 = 0
    tt = dz / (2 * kp) * pow((2 * np.pi / aa), 2)
    for j in range(n_grid):
        j1 = j + 1 - n1
        t0 = tt * j1 * j1
        h[j] = -t0
    return


# Propagation: Multiply the amplitude and phase
def evol1(n_grid, h, img0_):
    for i in range(n_grid):
        for j in range(n_grid):
            img0_[i,j] = img0_[i,j]*torch.exp(1j*h[i])
            img0_[i,j] = img0_[i,j]*torch.exp(1j*h[j])
    return


# FFT for 2D signal
def fftt(img0):
    img0_ = np.fft.fft2(img0)
    return img0_

# Inverse FFT for 2D signal
def fftt2(img0):
    img0_ = np.fft.ifft2(img0)
    return img0_ 