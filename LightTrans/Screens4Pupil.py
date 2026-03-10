#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This makes quilting pattern in 2D

@author: rfrazin
"""

import numpy as np

#1D normal fcn.
#x - spatial position
#fwhm - full width half max (set to 2.35*stdev)
#cen - center position
#Normalize - True if you want to divide by sqrt(2*pi)*sigma
def Norm1D(x, fwhm, cen, Normalize=False):
    arg = -1*(x-cen)*(x-cen)/(0.3621*fwhm*fwhm)
    f = np.exp(arg)
    if Normalize:
        f /= np.sqrt(2*np.pi)*(fwhm*2.35)
    return(f)

def QuiltScreen(filename):
    ppc = 64  # pixel per cell
    nc = 16   # number of cells
    fwhm =3   # full width half max of Gaussians
    filename = "quilt_1024x1024.txt"
    boxquilt = False
    spikequilt = True

    s = np.linspace(0, ppc-1,ppc)
    sx, sy = np.meshgrid(s, s)
    
    if boxquilt:
        qx = Norm1D(sx, fwhm, 0) + Norm1D(sx, fwhm, ppc-1)
        qy = Norm1D(sy, fwhm, 0) + Norm1D(sy, fwhm, ppc-1)
        qq = np.dstack((qx,qy))
        q = np.max(qq,axis=2)
    if spikequilt:
        qx = Norm1D(sx, fwhm, ppc//2-1)
        qy = Norm1D(sy, fwhm, ppc//2-1)
        q = qx*qy
    quilt = np.kron(np.ones((nc,nc)),q)
    quilt = 1 - quilt    
    np.savetxt(filename, quilt, fmt='%1.6e',delimiter=' ', newline='\n')
    return(quilt)

#this makes an amplitude cosine screen at an angle relative to the x-axis
def CosineAmpScreen(filename):
    npix = 1024
    angle = 80*np.pi/180 # angle of pattern
    fracamp = 0.1
    ncycles = 105 #spatial frequency in cycles/pupil
    x = np.linspace(-np.pi, np.pi*(npix-1)/npix,npix)
    x, y = np.meshgrid(x,x)
    xn = np.cos(  angle)*x + np.sin(angle)*y   #new coord grid
    yn = np.sin(- angle)*x + np.cos(angle)*y
    screen = 1. + fracamp*np.cos(ncycles*xn)
    np.savetxt(filename, screen, fmt='%1.6e',delimiter=' ', newline='\n')
    
    return(screen)

