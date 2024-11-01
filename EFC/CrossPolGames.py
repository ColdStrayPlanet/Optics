#!/usr/bin/env python3
"""
author: Richard Frazin  (rfrazin@umich.edu).  Please email if you have questions.

This is tutorial code designed to teach people how to use the tools I created
in course of writing  the article I sent to the Journal of Astronomical Telescopes,
Instruments and Systems (JATIS) on Oct. 3, 2024, entitled: A Laboratory Method for
Measuring the Cross-Polarizaion in High-Contrast Imaging.
This article is also publicly available at http://arxiv.org/abs/2410.03579

Before doing anything with this file, please see README.txt

"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import EFC  # this module is the main one
# %%
# generate and instance of the EFC class.  This will simulate a coronagraph without any
#   dark hole pixels specifies and no speckle field.  There are no aberrations, so this is the nominal model.
A = EFC.EFC(HolePixels=None, SpeckleFactor=0.)  # if this doesn't work, go back to README.txt

#create a DM command corresponding to a flat surface
C_flat = np.zeros((A.Sx.shape[1],))  # A.Sx is dominant field Jacobian (the variable name has no relationship to the musical note)

#get the dominant ('X') electric field at the detector for the nominal model - this array is complex-valued
fd = A.Field(C_flat, XorY='X', region='Full', DM_mode='phase',return_grad=False,SpeckleFactor=0.)
#        cross    ('Y")
fc = A.Field(C_flat, XorY='Y', region='Full', DM_mode='phase',return_grad=False,SpeckleFactor=0.)
fd = fd.reshape((256,256));  fc = fc.reshape((256,256))
#make images of the imag parts of the dominant and cross fields
plt.figure(); plt.imshow(np.imag(fd),cmap='seismic',origin='lower');plt.colorbar();
plt.figure(); plt.imshow(np.imag(fc),cmap='seismic',origin='lower');plt.colorbar();
# to get the nominal PSF images shown in the top row of Fig. 8, multiply fc and fd by their complex conjugates
# %%

#Make a random DM command (phase values, not heights) and look at the intensities.
#Note .PolIntensity uses the .Field function called above
C_rnd = 0.2*np.random.randn(len(C_flat))
Id_rnd = A.PolIntensity(C_rnd,'X','Full','phase',False,0.).reshape((256,256))
Ic_rnd = A.PolIntensity(C_rnd,'Y','Full','phase',False,0.).reshape((256,256))
plt.figure(); plt.imshow(np.log10(1.e-7  + Id_rnd),cmap='seismic',origin='lower');plt.colorbar();
plt.figure(); plt.imshow(np.log10(1.e-12 + Ic_rnd),cmap='seismic',origin='lower');plt.colorbar();
# %%
#Going back to a flat DM command, let's include the speckle fields from the files SpeckleFieldReducedFrom33x33PhaseScreen_Ex.npy, SpeckleFieldReducedFrom33x33PhaseScreen_Ey.npy
Id_sp = A.PolIntensity(C_flat,'X','Full','phase',False,SpeckleFactor=1.).reshape((256,256))
Ic_sp = A.PolIntensity(C_flat,'Y','Full','phase',False,SpeckleFactor=1.).reshape((256,256))
plt.figure(); plt.imshow(np.log10(1.e-7  + Id_sp),cmap='seismic',origin='lower');plt.colorbar();
plt.figure(); plt.imshow(np.log10(1.e-12 + Ic_sp),cmap='seismic',origin='lower');plt.colorbar();
# %%
#Now let's make up a new speckle field arising from random phases and amplitudes multiplying the CBS basis functions in the input plane
rndphasor = 1 + 0.3*np.random.randn(len(C_flat)) + 1j*np.random.randn(len(C_flat))
spfield_d = (A.Sx@rndphasor).reshape((256,256)) - fd
spfield_c = (A.Sy@rndphasor).reshape((256,256)) - fc
plt.figure(); plt.imshow(np.imag(spfield_d),cmap='seismic',origin='lower');plt.colorbar();
plt.figure(); plt.imshow(np.imag(spfield_c),cmap='seismic',origin='lower');plt.colorbar();
# %%
#Let's recreate the bottom row of Fig.8 in the paper
with open('stuff20240905.pickle','rb') as filep:  stuffB= pickle.load(filep)
#see what's in there!
stuffB.keys()
#this sets up the instance of the EFC class that already contains a dark for the
#  speckle field read in by the .__init__  function
B = EFC.EFC(HolePixels=stuffB['HolePixels'], SpeckleFactor=stuffB['SpeckleFactor'])
et = stuffB['pixel_extent']  # et phone home!
Cdh = stuffB['DHcmd']  # dark hole command for dominant field

IXabh = B.PolIntensity(  Cdh,XorY='X',region='Full',DM_mode='phase',return_grad=False,SpeckleFactor=None).reshape((256,256))
IYabh = B.PolIntensity(  Cdh,XorY='Y',region='Full',DM_mode='phase',return_grad=False,SpeckleFactor=None).reshape((256,256))

plt.figure(); plt.imshow(np.log10(1.e-7 + IXabh[et[2]:et[3],et[0]:et[1]]), extent=et,cmap='seismic',origin='lower');plt.colorbar();
plt.title('Aberrated Dominant PSF (contrast units) with Hole'); plt.ylabel('pixel index'); plt.xlabel('pixel index');

plt.figure(); plt.imshow(np.log10(1.e-13+ IYabh[et[2]:et[3],et[0]:et[1]]), extent=et,cmap='seismic',origin='lower');plt.colorbar();
plt.title('Aberrated Cross PSF (contrast units) with Hole'); plt.ylabel('pixel index'); plt.xlabel('pixel index');
# %%
# Let's make Fig.6 from the paper.
with open('ProbeSolutions20240905.pickle','rb') as pfpp:
    soldict = pickle.load(pfpp);
#see what's in there!
soldict.keys()
sol1 = soldict['Best3'][0]; sol2 = soldict['Best3'][1]; sol2 = soldict['Best3'][2]

pm = 1.   # positive probes
IAhx0 = B.PolIntensity(Cdh,'X','Hole','phase',False,None)
fAhy0 = B.Field(Cdh,'Y','Hole','phase',False,0.);
IAhx1 = B.PolIntensity(Cdh + pm*sol1,'X','Hole','phase',False,None)
phy1p = B.Field(Cdh + pm*sol1,'Y','Hole','phase',False,0.) - fAhy0;


plt.figure(figsize=(10,5)); #result of modulation with solution 1
plt.plot(extfac*np.sqrt(IAhx0),label='Unprobed Dominant $\sqrt{Intensity}$',marker='s',color='black',ls='None');
plt.plot(extfac*np.sqrt(IAhx1),label='Probed Dominant $\sqrt{Intensity}$',marker='s',color='tan',ls='None');
plt.plot(np.real(phy1p),label='real part of cross probe', marker='d',color='crimson',ls='None');
plt.plot(np.imag(phy1p),label='imag part of cross probe',marker='p',color='dodgerblue',ls='None');
plt.title('Probe Fields',fontsize=12)
plt.xlabel('pixel index',fontsize=12);
plt.ylabel('field ($\sqrt{\mathrm{constrast}}$ units)',fontsize=12);
plt.legend();
# %%
#
#To recreate Fig.10 (bearning in mind that the measurements are Poisson random deviates),
#   run the code lines in DarkHoleAnalysis.py under the heading: perform probing estimates

"""
                        Jacobian Testing

All EFC approaches, apart from the model-free one (ref. 10 in the paper) rely
  on the validity of the linearized hybrid equation for the dominant field,
  which is eq.29 in the paper.  This approach relies on the nonlinear version
  in eq.27 and its sibling for the cross field, i.e., eq.28.


Let Dx and Dy be the model (i.e., idealized) Jacobians indicated in the paper:
Dmx = B.Shx   # this only includes the 441 rows corresponding to the dark hole pixels
Dmy = B.Shy   # this only includes the 441 rows corresponding to the dark hole pixels
In this paper optical aberrations are mimicked with a a phase and amplitude screen
  applied to the input field.  A convenient way to do this for (at least for
  aberrations with power at low spatial frequencies) is to take advantage of the
  33x33 spline basis to which the Jacobians correspond.
"""
# %%
with open('stuff20240905.pickle','rb') as filep:  stuffB= pickle.load(filep)
#see what's in there!
stuffB.keys()
#this sets up the instance of the EFC class that already contains a dark for the
#  speckle field read in by the .__init__  function
B = EFC.EFC(HolePixels=stuffB['HolePixels'], SpeckleFactor=stuffB['SpeckleFactor'])
Cdh = stuffB['DHcmd']  # dark hole command for dominant field

#Since we don't have access to the original complex-valued phase screen used to create the
#  the speckle fields (B.spx and B.spy)*SpeckleFactor, let's find a vector of spline coefficients that
#  come close to reproducing B.spx and B.spy.   The basic equation is that the
#  speckle field, f, is given by f = S(a - 1), where a is the complex-valued
#  vector that plays the role of aberrations.
one = np.ones(Cdh.shape)
Bspx = B.spx*B.SpeckleFactor
Bspy = B.spy*B.SpeckleFactor
aber = np.linalg.pinv(B.Sx)@(Bspx + B.Sx@one)
spx = B.Sx@(aber - one)
spy = B.Sy@(aber - one)

plt.figure();plt.plot(np.imag(Bspx),'ko',np.imag(spx),'rx');plt.title('imag part of dominant speckle field');
plt.figure();plt.plot(np.real(Bspx),'ko',np.real(spx),'rx');plt.title('real part of dominant speckle field');
print('For the dominant field, the ratio of the misfit error std to the std of the true value is',
      np.std(spx-Bspx)/np.std(Bspx))
plt.figure();plt.plot(np.imag(Bspy),'ko',np.imag(spy),'rx');plt.title('imag part of cross speckle field');
plt.figure();plt.plot(np.real(Bspy),'ko',np.real(spy),'rx');plt.title('real part of cross speckle field');
print('For the cross field, the ratio of the misfit error std to the std of the true value is',
      np.std(spy-Bspy)/np.std(Bspy))
# %%

print("Still Under Construction")
