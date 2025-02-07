# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 15:28:31 2022
@author: Richard Frazin

This is a module containing functions for sampling from a
power spectral density (PSD).  There are tools for 1D and 2D.
This is intended for modeling surface errors on the optics.
It will be assumed that there are no amplitude effects, so
the height variations are real-valued, making the Fourier
transforms Hermitian.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import cupy as cp

"""
Normal PSD fucntion (1D)
Fmax - the largest spatial frequency - the frequency grid goes from
    0 to +Fmax  (Kmax = 2piFmax)
h - the height of Gaussian
c - center
S - the width (std) of the Gaussian
n - the number of sampling points
"""
def MyGaussian1D(Fmax, h, c, sig, n):
    s = np.linspace(0, Fmax, n)
    f = np.zeros(s.shape)
    for k in range(len(s)):
        f[k] = h*np.exp(- (s[k] - c)**2/(2*sig*sig) )
    return(f)

"""
This surface roughness function is from Jhen Lumbres' thesis.
sigSR has units of nm
The default values for Kmax and Kmin are:
   Kmax = 1/(2.5 um) = 4.e-4 nm^-1 and
   Kmin = 1/(85 um) = 1.1765e-5 nm^-1, resp.
The output units are nm^2m^2
"""
def BetaSR(sigSR, Kmin=1.1765e-5, Kmax=4.0e-4):
    return( (sigSR*sigSR/np.pi)/(Kmax*Kmax - Kmin*Kmin) )

"""
Jhen's surface PSDs are given my sums of von Karman functions
k - spatial wavenumber (1/m)
beta - not sure what it means, but we need it.
inS - inner scale length (l0)
otS - outer scale length (L0)
sigSR - see BetaSR function
Kmin - see BetaSR function
Kmax - see BetaSR function
"""
def VonKarmanPSD(k, beta, inS, otS, alpha, sigSR, Kmin=1.1765e4, Kmax=4.0e5):
    ak = np.abs(k)
    betasr = BetaSR(sigSR, Kmin, Kmax)
    ex = np.exp(- (ak*inS)**2 )
    denom = ( 1/(otS*otS) + ak*ak )**(alpha/2)
    return(betasr + beta*ex/denom)

"""
Jhen's surface PSDs are given by sums of von Karman functions.
This function calls VonKarmanPSD in order to sum them up.
k - spatial wavenumber (units m^-1)
amp - list of a_0 values (usually 1)
beta - list of beta values
inS - list of inner scale values
otS - list of outer scale values
alpha - list of alpha values
sigSr - list of sigSR values
"""
def sumVonKarmanPSD(k, amp, beta, inS, otS, alpha, sigSR, base=1.e-9):
    assert (len(amp) == len(beta) == len(inS) == len(otS) == len(alpha) == len(sigSR))
    psd = base
    for l in range(len(beta)):
        psd += amp[l]*VonKarmanPSD(k, beta[l], inS[l], otS[l], alpha[l], sigSR[l])
    return(psd)
"""
This integrates the 1D or 2D PSD to get the RMS as a function of the maximum spatial
  frequency considered [Kmax - this has nothing to with Kmax in the function
  VonKarmanPSD()].  Kmin sets the lower limit of the integration.

In order to transform the integral to (natural) log space, where q = log(k),
  we first have to make a change of variables.
  For a 1D PSF (not a radial PSF, but a truly 1D object):
    Let f1(k) be the 1D PSD (which we do not have from the MagAO-X team)
    and let g1(q) be the 1D PSD in log space.  Then, the required change of
    variables is given by the condition    f1(k) dk = g1(q) dq -> g1(q) = f1(k)|dk/dq|
  For a 2D radial PSF:
    Let f2(k) be the 2D radial PSF where k^2 = kx^2 + ky^2 and let q = log(k),
      so g2(q) is the 2D PSD in log space
    We can find g2(q) via the condition f2(k) k dk = g2(q) q dq -->
             g2(q) = f2(k) (k/q) |dk/dq| -> g2(q) = f2(exp(q)) (1/q) exp(2q)


Note that in 1D: RMS^2(Kmax) = 2*int_Kmin^Kmax f1(k) dk  (Kmin, Kmax  > 0)
                           = 2*int_{log(Kmin)}^{log(Kmax)} f1(exp(q)) exp(q) dq,
    In 2D:  RMS^2(Kmax) = 2pi*int_Kmin^Kmax f2(k) k dk
                        = 2pi*int_{log(Kmin)}^{log(Kmax)} f2(exp(q)) exp(2q) dq
Since k ranges over orders of magnitude, the log form the integral seems
    more suited to the problem.
Note that the units of k are [1/meters] so 1/(2.5 cm) = (1/0.025 m) = 40 m^-1
fcn  is a function handle that provides access to the function to be integrated.
    It is probably most convenient to make this a Python 'lambda' function
npts - the number of Kmax points
maxKmax - is the Kmax value for the last integration
Kmin - smallest spatial frequency considered in the integral
"""
def integrateLog(fcn, npts=100, maxKmax=2.e4, Kmin=4., ndim=2):
    assert ( (ndim == 1) or (ndim == 2) )
    lowlim = np.log(Kmin)  #  lower limit of integration
    logKmax = np.linspace(np.log(Kmin), np.log(maxKmax), npts)  # integration end points
    if ndim == 1:
        f = lambda t:       2*np.exp(  t)*fcn(np.exp(t))
    else:
        f = lambda t: 2*np.pi*np.exp(2*t)*fcn(np.exp(t))
    RMSsq = np.zeros((npts,))
    natLog2Log10 = np.log10(np.e)  # factor to convert natural logs to 10-based logs
    for nn in (1 + np.arange(npts-1)):
        RMSsq[nn] = quad(f, lowlim, logKmax[nn])[0]
    RMSsq[0] = RMSsq[1]  # don't want the first value to be zero

    return((RMSsq, natLog2Log10*logKmax))


"""
This samples a PSD from an input sequence assuming an
  exponential pdf: p(s) = (1/q)exp(-s/q) where q is the mean
psd - the input PSD -- only positive spatial frequencies!
L - length of the segment (centered on zero)
Fmax - max spatial frequency of psd
n - the number of points in the segment
"""
#def SampleExpPSD1D(psd, Fmax, L, n):
#    lp = len(psd)
#    f = np.linspace(0, Fmax, lp)  # spatial freq grid
#    x = np.linspace(-L/2, L/2, n)
#    surf = np.zeros(x.shape)
#    for k in (1 + np.arange(lp-1)): #don't include 0 spatial freq

#        pwr = np.random.exponential(psd[k])  ^^^ need to include "the dK"

#        camp = np.sqrt(pwr)*np.exp(1j*2*np.pi*np.random.rand())
#        ramp = np.real(camp)
#        iamp = np.imag(camp)
#        kf = 2*np.pi*f[k]  # spatial wavenumber
#        surf += 2*( ramp*np.cos(kf*x) + iamp*np.sin(kf*x) )
#    return(surf)

"""
This is similar to SampleExpPSD1D, except that is it 2D.
In 1D, if f(x) is real and g(k) is its FT, then:
   Re(g(k)) = Re(g(-k)) [even] and Im(g(k)) = - Im(g(-k)) [odd].  When we apply this result to calculate f(x) from g(k),
   we need only integrate over positive values of k to find:
   f(x) = 2*\int_{0}^{\infty} dk [ g_r(k)*cos(kx) - g_i(k)*sin(kx) ], where the standard FT normalization factor has not been included
The same holds
   in 2D when x and k are 2D vectors, but the 2D integral is over the upper 1/2 plane, but to show the integral result,
   you need to apply the above even/odd conditions by integrating over opposite quadrants.

psd - the input psd - assumed to a radial function, meaning that it depends on |k| only.
     -  units assumed to be nm^2 m^2
R - the radius of the surface (units meters), See below for 'square' option
gridspace - spacing of grid points on surface (units meters)
Kmin - minimum spatial frequency (units 1/m)
Kmax - maximum spatial frequency (units 1/m)
     - make sure this is resolved in terms of the gridspace parameter
dK - spatial fequency imcrement
CircOrSqu - must be 'circle' or 'square' for the output shape.  If 'square',
   the square is 2R-by-2R.
"""
def SampleExpPSD2D(psd, R, gridspace, Kmin, Kmax, dK, CircOrSqu='square', useCUPY=False):
    if CircOrSqu not in ['circle','square']:
       raise ValueError("CircOrSqu must be 'circle' or 'square'.")
    if useCUPY:
        pp = cp
    else:
        pp = np
        print("This can be rather slow without GPU acceleration.")
    #create spatial grid
    qq = pp.linspace(-R, R, int(2*R/gridspace))
    qq = pp.meshgrid(qq, qq)
    x = qq[0]; y = qq[1];  # x and y are 2D arrays of the spatial coords
    if CircOrSqu == 'circle':
       circle = pp.ones(x.shape)
       nk = x.shape[0]
       for l in range(nk):
          for m in range(nk):
             if x[m,l]**2 + y[m,l]**2 >= R*R:
                   circle[m,l] = 0.
    surf = pp.zeros(x.shape)  # random error surface
    #create spatial frequency grid
    qq = pp.linspace(-Kmax, Kmax, int(2*Kmax/dK))
    qq = pp.meshgrid(qq,qq)
    u = qq[0]; v = qq[1]  # u and v are 2D arrays of the spatial frequencies
    del(qq)
    nk = u.shape[0]  # length of spatial frequency grid
    tp = 2*pp.pi
    for l in range(nk):
        for m in range(int(nk/2)):  # only consider the lower 1/2 of the freq plane - this is where we implicitly assume the surface error is real-valued
            ak = pp.sqrt(v[m,l]**2 + u[m,l]**2)  # |k|
            if ( (ak < Kmin) or (ak > Kmax)):
                continue
            cf = pp.cos(tp*u[m,l]*x + tp*v[m,l]*y)
            sf = pp.sin(tp*u[m,l]*x + tp*v[m,l]*y)
            pwr = pp.random.exponential(psd(ak))*dK*dK
            camp = pp.sqrt(pwr)*pp.exp(1j*2*pp.pi*pp.random.rand())  # complex amplitude
            ramp = pp.real(camp)
            iamp = pp.imag(camp)
            surf += 2*(ramp*cf -iamp*sf)
    if CircOrSqu == 'circle':  surf *= circle
    if useCUPY:
        surf = cp.asnumpy(surf)
    return(surf)


"""
This loads Jhen's PSD parameters

Note that a precision polished metal mirror from Edmund has an RMS of about 4nm,
  which is consistent with the result from the integrateLog function above when
  using the 'flat' settings below.  Similarly, Edmust offers protected gold OAPs (various offset angle options)
  that have <5 nm and <10 nm RMS surface roughness.  The latter number is consistent
  with the result from integrateLog with 'OAP5' setting.


#OAP PSDs - output units are nm^2m^2
"""
params = {'OAP5':
  {'amp': [1., 1., 1.],
   'alpha': [3.029, -3.103, 0.668],
   'beta': [329.3, 1.6e-12, 3.49e-5],  # units are nm*m
    'otS': [0.019, 16., 0.024],  # units are m
    'inS': [-3.e-6, 4.29e-3, 1.32e-4],  # units are m
   'sigSR': [5.e-6, 5.e-6, 5.5e-1],  # units are nm
   'base': 1.e-9,  # units nm^2m^2
   'Kstart': 1. , # units 1/m  lowest spatial fequency
   'Kend':  1.e5  # units 1/m highest spatial frequency
  },'flat' :  # 50 mm flats, used for fold mirrors
   {'amp': [1., 1., 1.],
    'alpha': [3.284, 1.947, 2.827],
    'beta': [1180., 0.02983 , 44.25],  # units are nm*m
    'otS': [-0.017,-15., -5.7e-4],  # units are m
    'inS': [0.0225, 0.00335, 2.08e-4],  # units are m
    'sigSR': [5.e-5, 5.e-5, 0.08],  # units are nm
    'base': 1.e-10,  # units nm^2m^2
   'Kstart': 1. , # units 1/m  lowest spatial fequency
   'Kend':  1.e5  # units 1/m highest spatial frequency
    },'M3' :  # big telescope tertiary mirror
   {'amp': [1., 1., 0.],
    'alpha': [-27.924, 3.615, 0.],
    'beta': [1.261e-10, 65.27 , 0.],  # units are nm*m
    'otS': [0.42, 0.12, np.inf],  # units are m
    'inS': [-1.31, 8.81e-5, 0.],  # units are m
    'sigSR': [0.05, 0.37, 0.],  # units are nm
    'base': 2.3e-5,  # units nm^2m^2
    'Kstart': 1. , # units 1/m  lowest spatial fequency
    'Kend':  1.e5  # units 1/m highest spatial frequency
    }
}

#make psds
psd_oap = lambda k,  d = params['OAP5'] : sumVonKarmanPSD(k, d['amp'], d['beta'], d['inS'], d['otS'], d['alpha'], d['sigSR'], base=1.e-9)
psd_flat = lambda k, d = params['flat'] : sumVonKarmanPSD(k, d['amp'], d['beta'], d['inS'], d['otS'], d['alpha'], d['sigSR'], base=1.e-9)
#make psd for OAP5

#%%
if True:  # make psd plots

  var_oap , kaxis = integrateLog(psd_oap , npts=200, maxKmax=2.e4, Kmin=4., ndim=2)
  var_flat, kaxis = integrateLog(psd_flat, npts=200, maxKmax=2.e4, Kmin=4., ndim=2)



  Kstart = params['flat']['Kstart']; Kend = params['flat']['Kend']
  kgrid = np.logspace(np.log10(Kstart), np.log10(Kend), 233)
  pflat = 0.*kgrid
  poap =  0.*kgrid
  for m in range(len(kgrid)):
     pflat[m] = psd_flat(kgrid[m])
     poap[m]  = psd_oap(kgrid[m])

  #  plt.rcParams['text.usetex']  this isn't working for me

  plt.figure();
  plt.plot(np.log10(kgrid/2/np.pi), np.log10(pflat),'k-',linewidth=3, label='flat mirror');
  plt.plot(np.log10(kgrid/2/np.pi), np.log10(poap),'r:' ,linewidth=2, label='OAP');
  plt.legend(fontsize=12)
  plt.title('Fits to PSD Measurements')
  plt.xlabel('log10[  spatial frequency) (1/m)  ]',fontsize=12)
  plt.ylabel('log10[  PSD (nm^2/m^2)  ]',fontsize=12)

  plt.figure();
  plt.plot(kaxis,var_flat,'k-',label='flat mirror', linewidth=2);
  plt.plot(kaxis,var_oap ,'r:',label='OAP',linewidth=2);
  plt.title('Variance vs. max Spatial frequency')
  plt.ylabel('Variance [nm^2]',fontsize=12)
  plt.xlabel('log10[ spatial frequency (1/m) ]',fontsize=12)
  plt.legend(fontsize=12)
#%%
