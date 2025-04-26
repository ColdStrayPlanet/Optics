#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 17:01:39 2025
@author: Richard Frazin

This performs various functions to test concepts for
representing DMs

padding is used to deal effects that arise due to the splines not
being able to respresent non zero values at the edges of the spline grid.

"""

import numpy as np
import matplotlib.pyplot as plt
import Bspline3 as BS

#%%

x = np.linspace(-0.5, 0.5, 221)  # ordinate (hi res)

#set up a hi-res padded spline to fit to the unit-amplitude low-res phasor
NKhi_nom = 69  # nominal, i.e., not padded number of knots
NKhi = NKhi_nom + 2  # padded number of knots
deltahi = (x.max() - x.min())/NKhi_nom
Xhi_min = x.min() - deltahi/2
Shi = BS.UnivariateCubicSpline(x, NKhi, Xmin=Xhi_min)

#set up a low-res grid for the unit-amplitude phasor (the DM phase sheet)
NKlo = NKhi_nom//3
deltalo = (x.max() - x.min())/NKlo
Xlo_min = x.min() + deltalo/2
Slo = BS.UnivariateCubicSpline(x, NKlo, Xmin=Xlo_min)

#%%

clo = np.pi*(np.random.rand(NKlo) - 0.5)  #DM command
zlo = Slo.SplInterp(clo)  #DM phase sheet
ezlo = np.exp(1j*zlo)  # DM phasor (unit amplitude)
ezlor = np.real(ezlo); ezloi = np.imag(ezlo)

chi = Shi.GetSplCoefs(ezlo)  # fit hi res splines to DM phase
zhi = Shi.SplInterp(chi)  # estimate of DM phasor
zhir = np.real(zhi);  zhii = np.imag(zhi);

#%%
#amplitude comparison
plt.figure(); plt.plot(x, np.abs(ezlo),'ko-',x,np.abs(zhi),'rx:'); plt.title('amplitude comparison');

#real/imag comparison
plt.figure(); plt.plot(x,ezlor,'r-',linewidth=4);plt.plot(x,ezloi,'b-',linewidth=4);
plt.plot(x,zhir,'kx:',linewidth=2);plt.plot(x,zhii,'gx:',linewidth=2);plt.title('Re/Im comparison')
