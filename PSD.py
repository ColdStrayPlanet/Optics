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
This is the true Von Karman PSD for 2D, where k = (kx^2 + ky^2)^(1/2)
  k0 -- a break point, beyond which the function drops quickly.
  sigmasq -- total variance, obtained by analytical integration over the plane
  kr - radial spatial frequency - sqrt(kx*kx + ky*ky)
  units:  k0, kx, ky are in  (radians/meter)
          sigmasq is in nm**2
The integral of this funciton over the plane is sigsq
"""
def VonKarman2D(kr, k0, sigsq):
   num = 5*sigsq*k0**(5./3)
   den = 6*np.pi*(kr*kr + k0*k0)**(11./6)
   return(num/den)


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
In 1D, if f(x) is real and g(k) is its FT, then:
   Re(g(k)) = Re(g(-k)) [even] and Im(g(k)) = - Im(g(-k)) [odd].  When we apply this result to calculate f(x) from g(k),
   we need only integrate over positive values of k to find:
   f(x) = 2*\int_{0}^{\infty} dk [ g_r(k)*cos(kx) - g_i(k)*sin(kx) ], where the standard FT normalization factor has not been included
  The same holds in 2D along any line through the origin.  Equivalently, F(u,v) = conj[F(-u,-v)].
"""
"""
This samples a radial Power Spectral Density (PSD) with random amplitudes from the exponential distribution.
The sampling in the frequency plane is logarithmic in the radial direction.

psd - the input psd - assumed to be a radial function, meaning that it depends on |k| only.
     -  units assumed to be nm^2 m^2
R - the radius of the surface (units meters), See below for 'square' option
Kmin - minimum spatial frequency (units 1/m)
Kmax - maximum spatial frequency (units 1/m)
     - make sure this is resolved in terms of the gridspace parameter
nKr  - number or radial bins in the spatial frequency grid
CircOrSqu - must be 'circle' or 'square' for the output shape.  If 'square',
   the square is 2R-by-2R (see above for 'R').
outpix - output image size
"""
def SamplePSD2D_logspace(psd, R, Kmin, Kmax, nKr, CircOrSqu='square', useCUPY=False, outpix=512):
    if CircOrSqu not in ['circle','square']:
       raise ValueError("CircOrSqu must be 'circle' or 'square'.")
    if useCUPY:
        pp = cp
    else:
        pp = np
        print("This can be rather slow without GPU acceleration.  See the useCUPY kwarg.")
    qq = pp.linspace(-R, R, outpix)
    x = pp.meshgrid(qq, qq)[0]  # x and y are 2D arrays of the spatial coords
    y = pp.meshgrid(qq, qq)[1]
    if CircOrSqu == 'circle':
       circle = (x**2 + y**2 <= R**2).astype(float)
    surf = pp.zeros(x.shape)  # output array

    Kr = pp.geomspace(Kmin, Kmax, nKr)  # spatial frequency radii
    for nr in range(nKr):
        kr = Kr[nr]  # magnitude of spatial frequency
        if nr == 0 :
           dR = Kr[1] - Kr[0];
        elif nr == nKr - 1 :
           dR = Kr[nKr-1] - Kr[nKr-2]
        else:
           dR = 0.5*(Kr[nr+1] - Kr[nr-1])
        nphi = 8*kr/Kmin # number of angles
        Phi = pp.linspace(0, 2*pp.pi*(nphi-1)/nphi, nphi) + pp.pi*(np.random.rand() - 0.5)
        dArea =  kr*dR*2*pp.pi/nphi
        for phi in Phi:
           if phi > np.pi: continue # only need the upper 1/2 plane
           u = kr*pp.cos(phi);  v = kr*pp.sin(phi)  #spatial frequencies
           var_k = psd(pp.sqrt(u*u + v*v))  # variance at k from the PSD
           amp_k = pp.random.exponential(pp.sqrt(var_k))  # random amplitude of the wave at (u,v)
           spatialphase = 2*pp.pi*pp.random.rand()  # random spatial phase
           surf += amp_k*pp.sqrt(2*dArea)*pp.cos(spatialphase + u*x + v*y)  # '2'accounts for the half-plane in k-space
    if CircOrSqu == 'circle':  surf *= circle
    if useCUPY:
        surf = cp.asnumpy(surf)
    return(surf)


def _SampleExpPSD2D(psd, R, gridspace, Kmin, Kmax, dK, CircOrSqu='square'):
    if CircOrSqu not in ['circle','square']:
       raise ValueError("CircOrSqu must be 'circle' or 'square'.")
    N = int(2*R/gridspace)  # Grille spatiale
    qq = np.linspace(-R, R, N)
    x, y = np.meshgrid(qq, qq)
    df = 1.0 / (2*R)  # df est la résolution fréquentielle naturelle de la grille FFT
    F_uv = np.zeros((N, N), dtype=complex)
    qq_k = np.linspace(-Kmax, Kmax, int(2*Kmax/dK))  # Grille de fréquences pour la boucle
    u_grid, v_grid = np.meshgrid(qq_k, qq_k)
    nk = u_grid.shape[0]
    twopi = 2 * np.pi

    for l in range(nk):  # spatial frequency loop
        for m in range(int(nk/2)):
            u = u_grid[m,l]
            v = v_grid[m,l]
            ak = np.sqrt(u**2 + v**2)
            if (ak < Kmin) or (ak > Kmax):
                continue

            # Mapping des fréquences sur les indices de la matrice FFT
            # On utilise le modulo N pour gérer les fréquences négatives
            idx_u = int(np.round(u / df)) % N
            idx_v = int(np.round(v / df)) % N

            var_k = psd(ak) # sample from the psd
            ampl_k = np.random.exponential(np.sqrt(var_k))  #  ampl_k est en nm*m (moyenne = sqrt(var_k), variance = var_k)

            # Conversion pour la FFT :
            # On multiplie par N car ifft2 divise par N
            # On multiplie par sqrt(2) car on ne boucle que sur le demi-plan
            mag = ampl_k * dK * N * np.sqrt(2)
            phase = twopi * np.random.rand()

            # On accumule dans la matrice (au cas où dK < df)
            F_uv[idx_u, idx_v] += mag * np.exp(1j * phase)

    surf = np.fft.ifft2(F_uv).real
    if CircOrSqu == 'circle':
       circle = (x**2 + y**2 <= R**2).astype(float)
       surf *= circle
    return surf


def ___SampleExpPSD2D(psd, R, gridspace, Kmin, Kmax, dK, CircOrSqu='square', useCUPY=False):
    if CircOrSqu not in ['circle','square']:
       raise ValueError("CircOrSqu must be 'circle' or 'square'.")
    if useCUPY:
        pp = cp
    else:
        pp = np
        print("This can be rather slow without GPU acceleration.  See the useCUPY kwarg.")
    #create spatial grid
    qq = pp.linspace(-R, R, int(2*R/gridspace))
    x = pp.meshgrid(qq, qq)[0]  # x and y are 2D arrays of the spatial coords
    y = pp.meshgrid(qq, qq)[1]
    if CircOrSqu == 'circle':
       circle = (x**2 + y**2 <= R**2).astype(float)
    surf = pp.zeros(x.shape)  # random error surface
    #create spatial frequency grid
    qq = pp.linspace(-Kmax, Kmax, int(2*Kmax/dK))
    qq = pp.meshgrid(qq,qq)
    u = qq[0]; v = qq[1]  # u and v are 2D arrays of the spatial frequencies
    del(qq)
    nk = u.shape[0]  # length of spatial frequency grid
    twopi = 2*pp.pi
    for l in range(nk):
        for m in range(int(nk/2)):  # only consider the lower 1/2 of the freq plane - this is where we implicitly assume the surface error is real-valued
            ak = pp.sqrt(v[m,l]**2 + u[m,l]**2)  # |k|
            if ( (ak < Kmin) or (ak > Kmax)):
                continue
            spatialphase = twopi*pp.random.rand()  # spatial phase of the cosinusoidal bump on the surface
            wave = pp.cos(twopi*u[m,l]*x + twopi*v[m,l]*y + spatialphase)
            var_k = psd(ak)  # variance at |k| from the PSD
            ampl_k = pp.random.exponential(np.sqrt(var_k))  # the variance of np.random.exponential(beta) is beta**2
            surf += amp_k*np.sqrt(2*dK*dK)*wave # the sqrt(2) factor is because the spatial frequency loop is goes over only the upper 1/2 plane
            #sf = pp.sin(twopi*u[m,l]*x + twopi*v[m,l]*y + spatialphase)
            #camp = pp.sqrt(pwr)*pp.exp(1j*2*pp.pi*pp.random.rand())  # complex amplitude
            #ramp = pp.real(camp)
            #iamp = pp.imag(camp)
            #surf += 2*(ramp*cf -iamp*sf)
    if CircOrSqu == 'circle':  surf *= circle
    if useCUPY:
        surf = cp.asnumpy(surf)
    return(surf)


#############################################################################
###########################################################################
##   MagAO-X   Jhen Lumbres stuff
############################################################################
############################################################################

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
def JhenVonKarmanPSD(k, beta, inS, otS, alpha, sigSR, Kmin=1.1765e4, Kmax=4.0e5):
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
        psd += amp[l]*JhenVonKarmanPSD(k, beta[l], inS[l], otS[l], alpha[l], sigSR[l])
    return(psd)



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
