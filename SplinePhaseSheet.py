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
NKhi_nom = 31  # nominal, i.e., not padded number of knots
NKhi = NKhi_nom + 2  # padded number of knots
deltahi = (x.max() - x.min())/NKhi_nom
Xhi_min = x.min() - deltahi/2
Shi = BS.UnivariateCubicSpline(x, NKhi, Xmin=Xhi_min, Delta=deltahi)

#%%
#set up a low-res grid for the unit-amplitude phasor (the DM phase sheet)
NKlo = 11 #NKhi_nom//3
deltalo = (x.max() - x.min())/NKlo
Xlo_min = x.min() + deltalo/2
Slo = BS.UnivariateCubicSpline(x, NKlo, Xmin=Xlo_min)

#clo = np.pi*(np.random.rand(NKlo) - 0.5)  #DM command
clo = np.pi*np.random.randn(NKlo)  #DM command
zlo = Slo.SplInterp(clo)  #DM phase sheet
ezlo = np.exp(1j*zlo)  # DM phasor (unit amplitude)
ezlor = np.real(ezlo); ezloi = np.imag(ezlo)

chi = Shi.GetSplCoefs(ezlo)  # fit hi res splines to DM phase
zhi = Shi.SplInterp(chi)  # estimate of DM phasor
zhir = np.real(zhi);  zhii = np.imag(zhi);

#amplitude comparison
#plt.figure(); plt.plot(x, np.abs(ezlo),'ko-',x,np.abs(zhi),'rx:'); plt.title('amplitude comparison');

#real/imag + amp comparison
plt.figure(); plt.plot(x,ezlor,'r-',linewidth=4);plt.plot(x,ezloi,'b-',linewidth=4);
plt.plot(x,zhir,'kx:',linewidth=2);plt.plot(x,zhii,'gx:',linewidth=2);plt.title('Re/Im comparison')
plt.plot(x,np.abs(zhi),'c-')

#Look at the hi coefs
#plt.figure();plt.plot(np.real(chi),'ro');plt.plot(np.imag(chi),'b*');plt.title('Re/Im Hi Coefs')

#%%   perform a loop to get error stats

NKlo = [7, 11, 15, 17, 23]; ntrials = 11;
phst = np.pi*np.array([1/8,1/6,1/2,3/4,1,5/4,3/2])  # standard deviation of the DM phases
amperr = np.zeros((len(NKlo), len(phst)))  # amplitude errors
rmserr = np.zeros((len(NKlo), len(phst)))
for kk in range(len(NKlo)):
   deltalo = (x.max() - x.min())/NKlo[kk]
   Xlo_min = x.min() + deltalo/2
   Slo = BS.UnivariateCubicSpline(x, NKlo[kk], Xmin=Xlo_min)

   for kp in range(len(phst)):

      ae = 0.  # amplitude errors
      qe = 0.  # re/im errors
      for kt in range(ntrials):

         clo = phst[kp]*np.random.randn(NKlo[kk])  #DM command
         zlo = Slo.SplInterp(clo)  #DM phase sheet
         ezlo = np.exp(1j*zlo)  # DM phasor (unit amplitude)
         ezlor = np.real(ezlo); ezloi = np.imag(ezlo)
         chi = Shi.GetSplCoefs(ezlo)  # fit hi res splines to DM phase
         zhi = Shi.SplInterp(chi)  # estimate of DM phasor
         zhir = np.real(zhi);  zhii = np.imag(zhi);
         qe += np.mean( (zhir - ezlor)**2 + (zhii - ezloi)**2 )
         ae += np.mean( np.abs(ezlo) - np.abs(zhi) );

         if False:  #kt == 3:
            plt.figure(); plt.plot(x,ezlor,'r-',linewidth=4);plt.plot(x,ezloi,'b-',linewidth=4);
            plt.plot(x,zhir,'kx:',linewidth=2);plt.plot(x,zhii,'gx:',linewidth=2);plt.title('Re/Im comparison')
            plt.plot(x,np.abs(zhi),'c-')

      rmserr[kk,kp] = np.sqrt(qe/ntrials)
      amperr[kk,kp] = ae/ntrials

#%%
plt.figure(); plt.plot(phst/np.pi,amperr.T); plt.title('amplitude error');plt.xlabel('(std DM phases)/\pi'); plt.ylabel('mean amplitude error');
plt.figure(); plt.plot(phst/np.pi,rmserr.T); plt.title('RMS fit error');plt.xlabel('(std DM phases)/\pi'); plt.ylabel('RMS fit error');
