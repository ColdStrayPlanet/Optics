# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 13:18:15 2023
@author: rfrazin
This makes masks for the use as stored functions in LightTrans
"""

import numpy as np

#use this to save a 2D array to a text file
savetxt = lambda fname, g: np.savetxt(fname,g,fmt='%1.6e',delimiter=' ', newline='\n')

#The sides of square are equidistant under this measure
def L0(x,y):
    return max((np.abs(x),np.abs(y)))

#This makes a sqaure opaque occulter with a quadratic border to ease the transition
#npix - the output array is npix X npix
#cs - opaque center is cs X cs
#bw - the pixel width of the border
def SquareOcculterWithBorder(npix,cs,bw):
    ic = npix//2
    s = np.linspace(-ic, -ic + npix, npix+1)
    sx, sy = np.meshgrid(s,s)
    g = np.ones((npix,npix))
    
    cs2 = cs//2  
    for ky in range(npix):
        for kx in range(npix):
            d = L0(sx[ky,kx], sy[ky,kx])
            if d <= cs2:
                g[ky,kx] = 0.
            elif d <= cs2 + bw:
                dp = d - cs2
                g[ky,kx] = dp*dp/(bw*bw)
    return g



"""
This writes a vortex phase over a square aperture.
Various format options are available.
fn - output filename (.txt suffix needed)
charge = how many cyles of 2pi in a circle?
phi0 = phase at the x-axis (radians)
sz - output array size is sz-by-sz
fmt - format options 'phasor', 'angle' (radians)
  if 'phasor' the output is two files, with 'real' and 'imag' in the filenames
"""
def WriteVortexPhase(fn='phase.txt', charge=1, phi0=0., sz=512, fmt='angle'):
    assert fmt == 'angle' or fmt == 'phasor'
    delim = ' '
    newln = '\n'
    if fmt == 'angle':
        g = np.zeros((sz, sz))
    else:
        g = np.zeros((sz, sz)).astype('complex')
    x = np.linspace(-1,1, sz)  # 1D spatial coordinate
    for kx in range(sz):
        for ky in range(sz):
            phi = phi0 + charge*np.arctan2(x[ky], x[kx])
            if fmt == 'phasor':
                g[ky, kx] = np.exp(1j*phi)
                gr = np.real(g)
                gi = np.imag(g)
            else:
                g[ky, kx] = phi
    if fmt == 'angle':
        np.savetxt(fn, g, fmt='%1.6e', delimiter=delim, newline=newln)
    else:
        fnr = 'real_' + fn
        fni = 'imag_' + fn
        np.savetxt(fnr, gr, fmt='%1.6e', delimiter=delim, newline=newln)
        np.savetxt(fni, gi, fmt='%1.6e', delimiter=delim, newline=newln)
    return(None)

"""
This writes a phase screen that place a pi phase shift in the inner circle
  and has unity transmittance outside of the inner circle.
fn - output filename - should include .txt suffix
sz - full array is sz-by-sz
d  - inner circle has a diam of d pixels
"""
def WritePiPhaseShift(fn='PiPhase.txt', sz=1024, d=129):
    delim = ' '
    newln = '\n'
    fmt = '%1.1e'
    assert sz > d
    g = np.ones((sz, sz))
    ll = np.linspace(-1,1,sz)  # 1D spatial coordinate
    rmax = d/sz
    for kx in range(sz):
        x = ll[kx]
        for ky in range(sz):
            y = ll[ky]
            r = np.sqrt(x*x + y*y)
            if r <= rmax:
                g[ky, kx] = -1.0
    np.savetxt(fn, g, fmt=fmt, delimiter=delim, newline=newln)
    return(None)

"""
This carries out Zernike phase mask functions.
The idea is that the centeral portion of the image,
 as a function of r, is multiplied by -1, and then 
 the entire image is summed.
cs - the image summed as function of r
q  - the radius values
  
"""

def CircleSum(img):
    assert img.ndim == 2
    nr = img.shape[0]; nc = img.shape[1]
    assert nr == nc
    s = np.linspace(-1/2, 1/2, nr)  # linear coordinate
    q = np.linspace(0, 1/2, int(nr/2 - 1) ) # admissible radius values
    cs = np.zeros((int(nr/2 - 1), )).astype('complex')  #  output array
    r = np.zeros((nr,nr))  # radius of all pixels 
    for ky in range(nr):
        for kx in range(nr):
            r[ky, kx] = np.sqrt( s[ky]*s[ky] + s[kx]*s[kx]  )
    for kq in range(len(q)):
        mask = np.ones((nr, nr))
        p = np.where(r < q[kq])
        for m in range(len(p[0])):
            mask[p[0][m], p[1][m]] = -1
        prod = mask*img
        cs[kq] = np.sum(prod)
    return([cs, q])