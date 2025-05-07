#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:06:35 2024
@author: rfrazin

This performs EFC functions.
In this simple model, the polarized detector fields are given by a spline coefficient vector
  times the propagation matrix corresponding to the desired polarization.

"""
print("Initializing Module.")

usePyTorch = True  # set to False for CPU mode.  True provides GPU acceleration, not neural network stuff

import numpy as np
from os import path as ospath  #needed for isfile(), join(), etc.
from sys import path as syspath
import warnings
from scipy import optimize

if usePyTorch:
   try:
      import torch
   except ImportError:
      print("Unable to import PyTorch.  Using CPU mode.")
      torch = None
else:
   print("CPU mode selected.")
   torch = None
if torch is not None:
   if torch.cuda.is_available():
      print("GPU acceleration via PyTorch enabled.")
   else:
      print("PyTorch imported, but cuda is not available.  Using CPU mode.")
      torch = None

if torch is not None:  # Si PyTorch está disponible y CUDA está habilitado, usamos torch.tensor
    device = 'cuda'
    array =  lambda s: torch.tensor(s, dtype=torch.complex64, device='cuda')
    isreal = lambda s: not s.is_complex()
    exp = torch.exp
    diag = torch.diag
    outer = torch.outer
    ii = torch.tensor(1j, dtype=torch.complex64, device='cuda')
    real = torch.real; imag = torch.imag
    conj = torch.conj
else:
    device = None
    ii = 1j
    exp =  np.exp
    array = np.array
    diag = np.diag
    outer = np.outer
    real = np.real; imag = np.imag
    isreal = lambda s:  all(np.isreal(s))
    conj = np.conj

machine = "homeLinux"
#machine = "officeWindows"
if machine == "homeLinux":
    MySplineToolsLocation = "/home/rfrazin/Py/Optics"
    PropMatLoc = "/home/rfrazin/Py/EFCSimData/"
elif machine == 'officeWindows':
    MySplineToolsLocation = "E:/Python/Optics"
    PropMatLoc = "E:/MyOpticalSetups/EFC Papers/DataArrays"
syspath.insert(0, MySplineToolsLocation)
import Bspline3 as BS  # this module is in MySplineToolsLocation

fpsize = 512  # size of focal plane in pixels
fplength = 20. #length of detector in mm
Reduced = True
if not Reduced: assert False  # the fplength is problematic
if Reduced:  #stuff averaged over 2x2 pixels in the image plane
    fpsize //= 2
    Sxfn = 'ThreeOAP20mmSquareApCgkn33x33_SystemMatrixReducedContrUnits_Ex.npy'  # 'SysMatReduced_LgOAPcg21x21_ContrUnits_Ex.npy'
    Syfn = 'ThreeOAP20mmSquareApCgkn33x33_SystemMatrixReducedContrUnits_Ey.npy'  # 'SysMatReduced_LgOAPcg21x21_ContrUnits_Ey.npy'
    SpecfieldXfn = 'SpeckleFieldX_121224.npy' #'SpeckleFieldReducedFrom33x33PhaseScreen_Ex.npy'  # 'SpeckleFieldReducedFrom24x24screen_Ex.npy'
    SpecfieldYfn = 'SpeckleFieldY_121224.npy' #'SpeckleFieldReducedFrom33x33PhaseScreen_Ey.npy'  # 'SpeckleFieldReducedFrom24x24screen_Ey.npy'



#This avoids the numpy Poisson random generator when the intensity is too big
def RobustPoissonRnd(I):
    assert np.isscalar(I)
    if I <= 1.e4:
        return np.random.poisson(I)
    else:
        return I + np.sqrt(I)*np.random.randn()

#This returns the Cramer-Rao bound matrrix and the Fisher Information (FIM), if desired,
#  under the assumption that the measurements are Poisson distributed.
#If a scalar number, S, is the expected number of counts and the actual number
#  of counts, n, is Poisson distributed denoted as P(n|S), we have <n> = s and <n^2> = S^2 + S.
#  Let the gradient vector of S w.r.t. some parameters be gs, then 1 page of work shows that
#  the Fisher information matrix (FIM) for one experiment is ( 1/(S + rn) )*outer_prod(gs,gs).
#  outer_prod(gs,gs) is a singulat matrix, which can be shown by factorization since: (aa ab; ab bb) = (a 0; 0 b)(a b; a b)
#Note that the FIMs of independent experiments add.
#Then, Cramer-Rao bound is given by inv (FIM)
#S - is a vector (or list) of expected count numbers from independent experiments
#    this excludes readout noise
#Sg - is an array of gradients w.r.t. the parameters to be estimated.
#  Sg.shape[0] must equal len(S)
#dk - dark current noise level (photon units) - (thermal) dark current counts are Poisson - readout noise is not.
#return_FIM - if True, the FIM will be returned, too
def CRB_Poisson(S, Sg, dk=2.5, return_FIM=False):
    if len(S) != Sg.shape[0]:
        raise ValueError("S and Sg must have the same number of items")
    M = len(S); N = Sg.shape[1]
    fim = np.zeros((N,N))
    for k in range(M):
        gg = np.outer( Sg[k,:], Sg[k,:] )
        fim += ( 1./(S[k] + dk) )*gg
    crb = np.linalg.inv(fim)
    if not return_FIM:
        return crb
    else:
        return (crb, fim)

#This calculates the -1*(log likelihood) under a poisson distribution
#   Note that the Ncnt! term is not included
#Ncnt - a list (or array) of count values (measurements)
#I - a list (or array) of intensity values (units are counts, but need not be integer valued.)
#    I is the response variable.
#    note len(I) must equal len(Ncnt)
#Ig  - optional list (not array!) of intensity gradients. If provided the gradient will be output
#Igg - optional list (not array!) of intensity hessians.  If provided, the hessian will be output.
#      requires gradient to be provided.
def NegLLPoisson(Ncnt, I, Ig=None, Igg=None):
    M = len(Ncnt)  # M is the number of "measurements"
    if len(I) != M:
        raise ValueError("Input counts (M) and the number of intensity values do not match.")
    if Ig is not None and len(Ig) != M:
        raise ValueError("Input counts (M) and the number of intensity gradients do not match.")
    if Igg is not None:
        if Ig is None:
            raise ValueError("Hessian requires gradient to be provided.")
        if len(Igg) != M:
            raise ValueError("Input counts (M) and the number of intensity Hessians do not match.")

    negll = 0.  #negative log-likelihood value
    negllg = np.zeros_like(Ig[0])
    negllgg = np.zeros((len(Ig[0]), len(Ig[0])))
    for m in range(M): #fist calculate LL, LLg, LLgg and then multiply by -1 when done
        s = Ncnt[m]*np.log(I[m]) - I[m]
        negll -= s
        if Ig is not None:
            dlnPds = Ncnt[m]/I[m] - 1.
            sg = dlnPds*Ig[m]
            negllg -= sg
            if Igg is not None:
                d2lnPds2 = -1.*Ncnt[m]/(I[m]**2)
                sgg = d2lnPds2*np.outer(Ig[m],Ig[m]) + dlnPds*Igg[m]
                negllgg -= sgg
    if Igg is not None:
        return (negll, negllg, negllgg)
    if Ig is not None:
        return (negll, negllg)
    return negll


#This uses the hybrid equations to calculate the intensity
#f - a vector with 2 complex numbers:
#   [dominant field, cross field] - these are the quantities to be estimated
#   and the gradient is with respect to their real and image parts, as per the 'mode' kwarg
#p - a vector with two complex numbers representing the probe field
# [dominant probe, cross probe]
#mode - this only changes the gradient output.
#  choices are: 'Cross', 'Dom', 'CrossDom', 'CrossSinc','CrossDomSinc'
#   if 'Cross' - grad includes derivs w.r.t. real and imag part of cross fields
#      'Dom'                                                      dominant
#      'CrossDom'                            both    dominant and cross
#IdomOnly - only return dominant intensity.  gradient not available
def ProbeIntensity(f, p, mode='Cross', return_grad=True, return_hess=False,
                   IdomOnly=False):
    if return_hess: assert return_grad
    assert mode in ['Cross', 'Dom', 'CrossDom', 'CrossSinc','CrossDomSinc']
    assert len(f) == 2
    assert len(p) == 2
    CC = np.conj; RE = np.real; IM = np.imag  # makes it easier to read
    Idom = f[0]*CC(f[0]) + p[0]*CC(p[0]) - IM(p[0])*RE(f[0]) + RE(p[0])*IM(f[0])
    Icro = f[1]*CC(f[1]) + p[1]*CC(p[1]) - IM(p[1])*RE(f[1]) + RE(p[1])*IM(f[1])
    Itot = RE(Idom + Icro)
    if IdomOnly:
        assert return_grad is False
        return Idom
    if not return_grad:
        return Itot
    if mode == 'Cross':
        grad = np.zeros((2))
        grad[0] = 2*RE(f[1]) - IM(p[1]) # deriv w.r.t. Re(f[1])
        grad[1] = 2*IM(f[1]) + RE(p[1]) # deriv w.r.t. Im(f[1])
        hess = 2.*np.eye(2,2)
    elif mode == 'Dom':
        grad = np.zeros((2))
        grad[0] = 2*RE(f[0]) - IM(p[0]) # deriv w.r.t Re(f[0])
        grad[1] = 2*IM(f[0]) + RE(p[0]) # deriv w.r.t Im(f[0])
        hess = 2.*np.eye(2,2)
    elif mode == 'CrossDom':
        grad = np.zeros((4))
        grad[0] = 2*RE(f[0]) - IM(p[0]) # deriv w.r.t Re(f[0])
        grad[1] = 2*IM(f[0]) + RE(p[0]) # deriv w.r.t Im(f[0])
        grad[2] = 2*RE(f[1]) - IM(p[1]) # deriv w.r.t. Re(f[1])
        grad[3] = 2*IM(f[1]) + RE(p[1]) # deriv w.r.t. Im(f[1])
        hess = 2.*np.eye(4,4)
    elif mode == 'CrossSinc':
        assert False
        grad = np.zeros((3))
        grad[0] = 2*RE(f[1]) - IM(p[1]) # deriv w.r.t. Re(f[1])
        grad[1] = 2*IM(f[1]) + RE(p[1]) # deriv w.r.t. Im(f[1])
        grad[2] = 2*f[2]                # deriv w.r.t f[2] (which is real)
    elif mode == 'CrossDomSinc':
        assert False
        grad = np.zeros((5))
        grad[0] = 2*RE(f[0]) - IM(p[0]) # deriv w.r.t Re(f[0])
        grad[1] = 2*IM(f[0]) + RE(p[0]) # deriv w.r.t Im(f[0])
        grad[2] = 2*RE(f[1]) - IM(p[1]) # deriv w.r.t. Re(f[1])
        grad[3] = 2*IM(f[1]) + RE(p[1]) # deriv w.r.t. Im(f[1])
        grad[4] = 2*f[2]
    else: assert False
    if not return_hess:
        return (Itot, grad)
    else: return (Itot,grad,hess)


#===============================================================================
#                      EFC Class starts here
#==============================================================================
#This assumes that the input light is linearly polarized in the X-direction
# - HolePixels list (or array) of 1D pixel indices of the pixels in the dark hole.
#    if None, there is no dark hole.  See the non-member MakePixList() function.
#SpeckleFactor is a multiple applied to the additive speckle fields specified by the filenames SpecfieldXfn and SpecfieldYfn
#PupilScreen, if not None, applies a complex-valued transmission screen to the system matrices of both dominant and cross fields.
#  PupilScreen multiplies the system matrices and therefore must be a vector corresponding the ndm-by-ndm spline coefficients.
#  PupilScreen can, for example, model a DM offset if all of the values have unity amplitude.

class EFC():
    def __init__(self, HolePixels=None, SpeckleFactor=0.,PupilScreen=None):
        if HolePixels is not None:
            print("The dark hole has", str(len(HolePixels)) ,"pixels.")

        self.HolePixels = HolePixels
        self.lamb = 1.  # wavelength in microns
        self.ndm = 33  # number of actuators (1D)
        #self.lamdpix = (fpsize/fplength)*5*800*(self.lamb*1.e-3)/(21*0.3367) # "lambda/D" in pixel units, i.e., (pixels per mm)*magnification*focal length*lambda/diameter
        self.lamdpix = 10.
        if Reduced: self.lamdpix = 5.
        self.SpeckleFactor = SpeckleFactor
        if (PupilScreen is not None) and (np.ndim(PupilScreen) != 1) and  (len(PupilScreen) != self.ndm**2) :
              raise ValueError("If not None, PupilScreen must be a vector of the same size as the DM command.")
              assert False
        self.Sx = np.load(ospath.join(PropMatLoc, Sxfn))  # Sytem matrices
        self.Sy = np.load(ospath.join(PropMatLoc, Syfn))
        self.spx = np.load(ospath.join(PropMatLoc, SpecfieldXfn))  # X polarized speckles
        self.spy = np.load(ospath.join(PropMatLoc, SpecfieldYfn))  # y polarized speckles
        assert self.Sx.shape[1] == self.ndm**2
        assert self.Sy.shape == self.Sx.shape
        if PupilScreen is not None:
           self.Sx *= PupilScreen  # same as multiplying by the matrix np.diag(PupilScreen), but much faster
           self.Sy *= PupilScreen
        self.spx = self.spx.flatten()
        self.spy = self.spy.flatten()
        self.CostScale = None
        self.CrossOptimSetUp = False
        self.OptPixels = []

        if HolePixels is not None:  #trim the matrices and the speckle field to correspond to HoleBndy
            self.Shx  = np.zeros((len(HolePixels), self.ndm**2)).astype('complex')  #trimmed system matrices
            self.Shy  = np.zeros((len(HolePixels), self.ndm**2)).astype('complex')
            self.sphx = np.zeros((len(HolePixels),)).astype('complex')
            self.sphy = np.zeros((len(HolePixels),)).astype('complex')
            for k in range(len(HolePixels)):
                self.Shx[k,:] = self.Sx[HolePixels[k],:]
                self.Shy[k,:] = self.Sy[HolePixels[k],:]
                self.sphx[k]  = self.spx[HolePixels[k]]
                self.sphy[k]  = self.spy[HolePixels[k]]

        if torch is not None:  # convert the above quantities into torch.tensors
           self.Sx = array(self.Sx)
           self.Sy = array(self.Sy)
           self.spx = array(self.spx)
           self.spy = array(self.spy)
           if HolePixels is not None:
              self.Shx = array(self.Shx)
              self.Shy = array(self.Shy)
              self.sphx = array(self.sphx)
              self.sphy = array(self.sphy)

        return(None)

    #This returns the x- or y- polarized intensity as a function of the spline coefficient vector
    #This returns numpy arrays even when PyTorch acceleration is enabled.
    #The spline coefficient vector must have a shape (self.ndm**2,)
    #XorY - select the desired polarization 'X' or 'Y'
    #region - if 'Hole' only the intensity inside self.HoleBndy is computed.
    #            'Full' the intensity is calculated over the entire range
    #DM_mode - 'height': coef must be real-valued and the phasor phase is 4pi*coef/self.lambda
    #        - 'phase' : coef must be real-valued and the phasor phase is coef itself
    #return_grad - return the gradient.
    #SpeckleFactor - multiplier for additive speckle field.  Can be 0.  None corresponds to defaul (see __init__)
    def PolIntensity(self, coef, XorY='X', region='Hole', DM_mode='phase', return_grad=True, SpeckleFactor=None):
       if torch is not None:
          otype = 'torch'
       else:
          otype = 'numpy'
       if not return_grad:
            f =       self.Field(coef, XorY=XorY, region=region, DM_mode=DM_mode,
                      return_grad=False, SpeckleFactor=SpeckleFactor, outputType=otype)
       else:
            (f, df) = self.Field(coef, XorY=XorY, region=region, DM_mode=DM_mode,
                      return_grad=True,  SpeckleFactor=SpeckleFactor, outputType=otype)
       I = real(f*conj(f))
       if return_grad:
           dI = 2*real(conj(f)*df.T).T

       if torch is not None:
           I = I.cpu().numpy()
           if return_grad:
              dI = dI.cpu().numpy()

       if not return_grad:
              return I
       else:
          return (I, dI)

    #See self.PolIntensity for notes
    def Field(self, coef, XorY='X', region='Hole', DM_mode='phase', return_grad=True, SpeckleFactor=None, outputType='torch'):
       if SpeckleFactor is None: SpeckleFactor = self.SpeckleFactor
       if XorY       not in ['X', 'Y']:          raise ValueError("Invalid value of 'XorY'.  Self explanatory.")
       if region     not in ['Hole', 'Full']:    raise ValueError("Invalid value for 'region'. Expected 'Hole' or 'Full'.")
       if DM_mode    not in ['height', 'phase']: raise ValueError("Invalid value for 'DM_mode'. Expected 'height' or 'phase'.")
       if outputType not in ['numpy', 'torch']:  raise ValueError("Invalid value for 'outputType'. Expected 'numpy' or 'torch'.")
       if outputType == 'torch' and torch is None: raise ValueError("PyTorch on the GPU must be enabled to use the 'torch' option.")
       nc = self.ndm**2
       if torch is None:
          assert all(np.isreal(coef))
       else:
          coef = torch.tensor(coef, dtype=torch.float32).to(device='cuda')
       if isinstance(coef, np.ndarray):
          if coef.shape != (nc,):  raise ValueError(f"Coeficientes deben tener la forma ({nc},). Forma actual: {coef.shape}.")
       elif isinstance(coef, torch.Tensor):
          if coef.shape != torch.Size([nc]):  raise ValueError(f"Coeficientes deben tener la forma ({nc},). Forma actual: {coef.shape}.")
       else:  raise TypeError(f"Tipo de coef no soportado: {type(coef)}")

       debug = False

       if debug: print("update! coef device", coef.device)

       if XorY == 'X':
           if region == 'Hole':
               Sys = self.Shx
               sp = self.sphx
           else:  # 'Full'
               Sys = self.Sx
               sp  = self.spx
       else:  # 'Y'
           if region == 'Hole':
               Sys = self.Shy
               sp  = self.sphy
           else:  #'Full'
               Sys = self.Sy
               sp  = self.spy

       if debug: print("Sys device", Sys.device)

       if DM_mode == 'height':
           c = exp(ii * 4. * np.pi * coef / self.lamb)
           if return_grad:
               dc = ii * 4. * np.pi * c / self.lamb
       elif DM_mode == 'phase':
           c = exp(ii * coef)  # Replaced np.exp with exp and 1j with ii
           if return_grad:
               dc = ii * c
       else:  # Not an option
           assert False

       if debug: print("c device", c.device)
       if debug: assert False

       f = Sys @ c
       f += SpeckleFactor * sp

       if return_grad:
         df = Sys*dc  # This is the same as Sys @ diag(dc), assuming 'Speckles' don't depend on c
       if outputType == 'torch':
           if return_grad:
               return (f.to(device='cuda'), df.to(device='cuda')) if return_grad else f.to(device='cuda')
           else:
               return f.to(device='cuda')
       elif outputType == 'numpy':
          if torch is not None:
            if return_grad:
               return (f.cpu().numpy(), df.cpu().numpy())
            else:
               return f.cpu().numpy()
          else:  # torch is None
             if return_grad:
                return (f, df)
             else: return f
       else:
           raise ValueError("outputType must be 'numpy' or 'torch'.")

    def SetupCrossOpt(self, c0, OptPixels):  # see self.CostCrossFieldWithDomPenalty
        self.CrossOptimSetUp = True
        self.OptPixels = OptPixels
        crfield0 = self.Field(c0, XorY='Y',region='Hole',DM_mode='phase',return_grad=False,SpeckleFactor=0.)
        if OptPixels is None:
            self.crfield0 = crfield0
            return None
        self.OptPixHoleDex = []
        self.crfield0 = np.zeros((len(OptPixels),)).astype('complex')
        counter = -1
        for opix in OptPixels:
            assert opix in self.HolePixels
            opdex = self.HolePixels.index(opix)
            self.OptPixHoleDex.append( opdex )
            counter += 1
            self.crfield0[counter] = crfield0[opdex]
        return None
    #This is a cost function for minimizing the ReOrIm (see below) corresponding
    #  to real or imag component of the cross field
    #a - command for departure from c0
    #c0 - dark hole command for the dominant field
    #return_grad (with respect to a) - bool
    #optPixels - a list of pixels whose fields are to be optimized (all must in self.HolePixels)
    #     if None, then self.HolePixels is used
    #ReOrIm - quantity to be minimized.  options are 'Re', 'Im' #
    #intthr - intensity threshold for cost penalty of dominant intensity
    #      - if None, not applied
    #pScale - multiplier applied to dominant intensity penalty above 'intthr'
    def CostCrossFieldWithDomPenalty(self, a, c0, return_grad=False, OptPixels=None, mode='Int', intthr=1.e-6, pScale=3.e-3):
        scale = 1.e9  # this helps some of the optimizers
        cmdthr = np.pi/4; # command amplitude limit (radians) - when small it corresponds to the linear approx to exp(1j*x);  too large it may strain the validity of the hybrid equations
        cmdpenamp = 1.e9  # command amplitude penalty scale
        assert mode in ['Re','Im','Int']
        if (self.CrossOptimSetUp is False) or (OptPixels != self.OptPixels):
            self.SetupCrossOpt(c0, OptPixels)

        crfield0 = self.crfield0
        fh  = self.Field(c0 + a, XorY='Y',region='Hole',DM_mode='phase',return_grad=False,SpeckleFactor=0.)
        if OptPixels is None:
            f = fh
        else:  # this is needed because the constraintPixels and OptPixels indices correspond to the full image
            f = np.zeros((len(OptPixels),)).astype('complex')
            for k in range(len(OptPixels)):
                f[k] = fh[self.OptPixDex[k]]

        if mode in ['Re','Im']:
          re = np.real(f);
          im = np.imag(f);
          re0 = np.real(crfield0)
          im0 = np.imag(crfield0)
          assert re0.shape == re.shape
        if mode == 'Re':
             cost = - 0.5*np.sum((re-re0)**2)
        elif mode == 'Im':
             cost = - 0.5*np.sum((im-im0)**2)
        else:   # mode is 'Int'
            cost = - 0.5*np.sum(np.real( (f - crfield0)*np.conj(f - crfield0) ))

        # command amplitude penalty term
        wathp = np.where(a >  cmdthr)[0]
        wathm = np.where(a < -cmdthr)[0]
        cost += cmdpenamp*np.sum(  a[wathp] - cmdthr)
        cost += cmdpenamp*np.sum(- a[wathm] - cmdthr)

        if intthr is not None:  # penalty fcn is Ix - threshold  if Ix > threshold
            q = self.PolIntensity(a + c0,'X','Hole','phase',return_grad=False, SpeckleFactor=None)
            wqth = np.where(q > intthr)[0]
            q = q[wqth] - intthr
            cost += pScale*q.sum()

        if not return_grad:
            return cost*scale

        else:  # return_grad
           if OptPixels is None:
                f, df = self.Field(c0+a, XorY='Y',region='Hole',DM_mode='phase',
                                   return_grad=True,SpeckleFactor=0.)
           else:
                f_, df_ = self.Field(c0+a, XorY='Y',region='Hole',DM_mode='phase',
                                   return_grad=True,SpeckleFactor=0.)
                df = np.zeros((len(OptPixels),df_.shape[1])).astype('complex')
                for k in range(len(OptPixels)):
                    df[k,:] = df_[self.OptPixDex[k],:]

           re = np.real(f);  dre = np.real(df)
           im = np.imag(f);  dim = np.imag(df)
           if mode == 'Re':
                dcost = - np.sum(dre.T*(re-re0), axis=1)
           elif mode == 'Im':
                dcost = - np.sum(dim.T*(im-im0), axis=1)
           else:  # mode is 'Int'
                dcost = - np.sum(np.real(df.T*(f-crfield0)), axis=1)
           dcost += cmdpenamp*len(wathp)
           dcost -= cmdpenamp*len(wathm)

           if intthr is not None:
                Ix, gIx = self.PolIntensity(a + c0,'X','Hole','phase',return_grad=True, SpeckleFactor=None)
                gIx = gIx[wqth,:]; #Ix = Ix[wqth]
                dcost += pScale*np.sum(gIx,axis=0)
           return cost*scale, dcost*scale


    #This sets up optimizations for minimizing the fuction self.CostCrossField
    #c0 - see initial dark hole comment
    #a0 - initial guess to start the optimizer
    #ReOrIm - see self.CostCrossField
    #method - choices are 'CG', 'NCG'
    def OptCrossField(self, c0, a0=None, ReOrIm='Re', maxiter=20):
        options = {'disp': True, 'maxiter': maxiter}

        cfcn = lambda a: self.CostCrossFieldWithDomPenalty(a, ReOrIm, c0, intthr=1.e-8,pScale=1.e-4, return_grad=True)  # set costfunction
        init_cost = cfcn(a0)
        print('Starting Cost', init_cost[0])

        if False:  #nonlinear constraint for SLSQP
            f_bound = 1.e-4
            ub = f_bound*np.ones(2*len(self.HolePixels))
            def gradReImField(c):
              s,ds = self.Field(c0+c,'X','Hole','phase',return_grad=True, SpeckleFactor=None)
              #ss = np.concatenate( (np.real(s),np.imag(s)), axis=0)
              dsds = np.concatenate( (np.real(ds),np.imag(ds)), axis=0)
              return dsds
            def ReImField(c):
              s    = self.Field(c0+c,'X','Hole','phase',return_grad=False,SpeckleFactor=None)
              ss   = np.concatenate((np.real(s),np.imag(s)), axis=0)
              return ss
            con = optimize.NonlinearConstraint(ReImField,ub,-ub,jac=gradReImField)
        return None

    #This is a cost function for a dark hole in the dominant polarization ('X')
    #c - DM command vector
    #scale - setting this to 10^6 seems to help the Newton-CG minimizer
    def CostHoleDominant(self, c, return_grad=True, scale=1.e6,SpeckleFactor=None):
        self.CostScale = scale
        if np.iscomplexobj(c):
            print("CostHoleDominant: Warning: Input parameter c should not be complex-valued.")
            assert sum(np.imag(np.abs(c))) == 0.
        assert c.shape == (self.Sx.shape[1],)
        if return_grad:
            I, dI = self.PolIntensity(c,XorY='X',region='Hole',DM_mode='phase',return_grad=True,SpeckleFactor=SpeckleFactor)
            cost = np.sum(I)
            dcost = np.sum(dI, axis=0)
            return (scale*cost, scale*dcost)
        else:
            I     = self.PolIntensity(c,XorY='X',region='Hole',DM_mode='phase',return_grad=False,SpeckleFactor=SpeckleFactor)
            cost = np.sum(I)
            return scale*cost


    #This calculates the Hessian corresponding to CostHoleDominant.   See Frazin (2018) JOSAA v35, p594
    #Note that the intensity of each detector pixel has its own Hessian, which would be a large object.
    #  This Hessian assumes that the cost is the sum of the pixel intensities in the dark hole
    def HessianCostHoleDominant(self, coef, SpeckleFactor=None):
        if self.CostScale is None:
            print("You must run CostHoleDominant to set self.CostScale first.")
            assert False
        if SpeckleFactor is None: SpeckleFactor = self.SpeckleFactor
        coef = array(coef)
        Sys = self.Shx  #  hole system matrix - see self.PolIntensity
        f = Sys@exp(ii*coef)
        f += SpeckleFactor*self.sphx  # hole speckle field
        df = ii*Sys*exp(ii*coef)  # same as Sys@np.diag(np.exp(1j*coef))
        H = array(np.zeros((len(coef),len(coef))))
        for m in range(Sys.shape[0]):  #loop over pixels
            H += 2*real( outer(df[m,:], conj(df[m,:]) ) )
            H += 2*real(diag( ii*df[m,:]*conj(f[m])) )   # additional term for diagonal elements
        if torch is not None:
           H = H.cpu().numpy()
        return H*self.CostScale

    #This does the optimization over the dominant intensity to dig the dark hole
    #c0 - initial guess
    #method - options are 'NCG' (Newton-ConjGrad), 'CG' (ConjGrad), 'SLSQP'
    #NCG with the analytical Hessian is great, but applying CG afterwards is effective, too.
    def DigHoleDominant(self, c0, method='NCG',maxiter=20):
        if method == 'NCG':  #Newton-CG set up.  leads to large command values without a penalty
           options={'disp':True, 'maxiter':maxiter}
           out = optimize.minimize(self.CostHoleDominant,c0,args=(),method='Newton-CG',options=options,
                                   jac=True,hess=self.HessianCostHoleDominant)
        elif method == 'CG':
           options={'disp':True, 'maxiter':maxiter}
           out = optimize.minimize(self.CostHoleDominant,c0,args=(),method='CG',options=options,jac=True)

        elif method == 'SLSQP':   #SLSQP set up
            options = {'disp': True, 'maxiter':10 }
            maxabs = np.pi/12
            conmat = np.eye(len(c0))
            ub =   maxabs*np.ones(c0.shape)
            lb = - maxabs*np.ones(c0.shape)
            constr = optimize.LinearConstraint(conmat, lb=lb, ub=ub)
            out = optimize.minimize(self.CostHoleDominant, c0,options=options,
                                    method='SLSQP',jac=True,constraints=(constr,))
            ffvalue = self.CostHoleDominant(out['x'], return_grad=False)
            print("Final Dark Hole Cost = ", ffvalue)
        return out

    #This makes the 'M matrix', which is the linearized system matrix (in terms of the DM phase),
    #  but broken into real and imag parts.
    #c0 - the linearization point
    #XorY - 'X' or 'Y'
    #pixlist - the list of 1D pixel indices within the dark hole that are
    #  to be kept dark while the cross field is probed.
    #With U,S,V = np.linalg.svd(M)  the last rows of V, e.g., V[k,:], are basis vectors corresponding to the small SVs of M
    def MakeMmat(self, c0, XorY='X', pixlist=None):
        if pixlist is None: pixlist = self.HolePixels
        assert XorY in ['X', 'Y']
        lpl = len(pixlist);
        spl = pixlist
        if XorY == 'X':
            S = self.Sx
        else:
            S = self.Sy
        cc0 = np.cos(c0); sc0 = np.sin(c0)
        M = np.zeros((2*lpl, self.Sx.shape[1]))
        for k in range(lpl):
            M[k      ] = np.real(S[spl[k],:])*cc0 - np.imag(S[spl[k],:])*sc0
            M[k + lpl] = np.real(S[spl[k],:])*sc0 + np.imag(S[spl[k],:])*cc0
        return M

    #This returns a matrix, V, whose columns for a basis of the M matrix (see self.MakeMmat()  )
    #c0 - reference DM command, probably corresponding to a dark hole
    #pixlist - list of pixels that define the M matrix, if None then self.HolePixels is used
    #sv_thresh the singular value ration used to define the effective null space of M
    #   the 10^-3 value is chosen by comparing the SVs of self.Shx to the norms of
    #   vectors obtained by self.Shy@VM[k,:] and examining the slopes of the two curves.
    #   If you do this you will see that the cross norms decay more slowly at first
    def GetNull(self, c0, pixlist=None, sv_thresh=1.e-3):
        if pixlist is None: pixlist = self.HolePixels
        M = self.MakeMmat(c0, pixlist)
        UM, SM, VM = np.linalg.svd(M)  # the ROWS of VM, e.g., VM[0,:] are the singular vectors
        svals = np.zeros(len(c0))
        svals[:len(SM)] = SM
        isv = np.where(svals < SM[0]*sv_thresh)[0]
        VM = VM.T # now the columns of VM are the singular vectors
        V = np.zeros((VM.shape[0],len(isv)))
        for k in range(len(isv)):
            V[:,k] = VM[:,isv[k]]
        return V


    #==========================#
    #  scrapyard for EFC class #
    #==========================#


    def _CostCrossFieldWithDomPenalty(self, a, ReOrIm, c0, ampthr=1.e-4, return_grad=False):
        assert False
        scale = 1.  # this helps some of the optimizers
        penaltyScale = 1.
        assert ReOrIm in ['Re','-Re','Im','-Im']

        cost = 0.0;
        f = self.Field(c0+a, XorY='Y',region='Hole',DM_mode='phase',return_grad=False,SpeckleFactor=0.)
        re = np.real(f);
        im = np.imag(f);
        if ReOrIm == 'Re':
             cost += -re.sum()
        elif ReOrIm == '-Re':
             cost += re.sum()
        elif ReOrIm == 'Im':
             cost += -im.sum()
        else:  # '-Im'
             cost += im.sum()

        if ampthr is not None:  # penalty fcn is \sqrt(Ix) - threshold  if \sqrt(Ix) > threshold
            q = np.sqrt( self.PolIntensity(a + c0,'X','Hole','phase',return_grad=False, SpeckleFactor=None)  )
            wqth = np.where(q > ampthr)[0]
            q = q[wqth] - ampthr
            cost += penaltyScale*q.sum()

        if not return_grad:
            return cost*scale

        else:  # return_grad
           f, df = self.Field(c0+a, XorY='Y',region='Hole',DM_mode='phase',
                                    return_grad=True,SpeckleFactor=0.)
           re = np.real(f);  dre = np.real(df)
           im = np.imag(f);  dim = np.imag(df)
           if ReOrIm == 'Re':
                dcost = -dre.sum(axis=0)
           elif ReOrIm == '-Re':
                dcost = dre.sum(axis=0)
           elif ReOrIm == 'Im':
                dcost = -dim.sum(axis=0)
           else:  # '-Im'
                dcost = dim.sum(axis=0)

           if ampthr is not None:
                Ix, gIx = self.PolIntensity(a + c0,'X','Hole','phase',return_grad=True, SpeckleFactor=None)
                Ix = Ix[wqth]
                gIx = gIx[wqth,:]
                gq = 0.5*(gIx.T*(1./np.sqrt(q))).T
                dcost += penaltyScale*np.sum(gq,axis=0)
           return cost*scale, dcost*scale

    #This calculates the Hessian corresponding to CostHillCross.   See Frazin (2018) JOSAA v35, p594
    #Note that the intensity of each detector pixel has its own Hessian, which would be a large object.
    #  This Hessian assumes that the cost is the sum of the pixel intensities in the dark hole
    def _HessianCostHillCross(self, coef, SpeckleFactor=None):
        if self.CostScale is None:
            print("You must run CostHillCross to set self.CostScale first.")
            assert False
        if SpeckleFactor is None: SpeckleFactor = self.SpeckleFactor
        Sys = self.Shy  #  hole system matrix - see self.PolIntensity
        f = Sys.dot(np.exp(1j*coef))
        f += SpeckleFactor*self.sphy  # hole speckle field
        df = 1j*Sys*np.exp(1j*coef)  # same as Sys@np.diag(np.exp(1j*coef))
        H = np.zeros((len(coef),len(coef)))
        for m in range(Sys.shape[0]):  #loop over pixels
            H += -2*np.real( np.outer(df[m,:], df[m,:].conj()) )
            H += -2*np.real(np.diag( 1j*df[m,:]*f[m].conj() ))  # additional term for diagonal elements
        return H*self.CostScale


    #This calculates Ey on self.HolePixels when the DM command is defined in
    #  terms of basis vectors
    #This does not include the speckle field.  In terms of the simulation, it is the model field
    #The idea is that the basis vectors in a useful effective null space for
    #  dominant field
    #a  - vector of null space coefficients. must have len(a) == Vn.shape[1]
    #   - if Vn=None is it simply a command vector in the original command space
    #c0 - initial (dark hole) DM command
    #Vn - the columns of Vn form the new basis.  See self.GetNull
    #     if None, no new basis is used.
    #return_grad - if True it returns the derivatives with respect to the coefficient vector a
    def _CrossFieldNewBasis(self, a, c0, Vn=None, return_grad=True):
        if Vn is not None:
            assert len(a) == Vn.shape[1]
            Vna = Vn@a  # phasor = np.exp(1j*Vn@a)
        else:
            assert len(a) == self.Sx.shape[1]
            Vna = a
            Vn = np.eye(len(a))

        S = self.Shy*np.exp(1j*c0)  # equiv to self.Shy@np.diag(np.exp(1j*c0))
        #ey = S@np.exp(1j*Vna); ey_re = np.real(ey); ey_im = np.imag(ey)  # the code below is slightly faster
        ey_re = np.real(S)@np.cos(Vna) - np.imag(S)@np.sin(Vna)  # real part of field
        ey_im = np.real(S)@np.sin(Vna) + np.imag(S)@np.cos(Vna)  # imag part of field
        if not return_grad:
            return (ey_re, ey_im)
        else:
            Vnc = (Vn.T*np.cos(Vna)).T
            Vns = (Vn.T*np.sin(Vna)).T
            dr = - np.real(S)@Vns - np.imag(S)@Vnc
            di =   np.real(S)@Vnc - np.imag(S)@Vns
            return (ey_re, ey_im, dr, di)

    #Ithresh - the intensity level at which the penalty for X intensity turns on
    def _CostCrossDiffIntensityRatio(self, a, ReOrIm, c0, Vn, return_grad=False, Ithresh = 1.e-9, scale=1.):
        def QuadThreshPenalty(s, thr, s_grad=None, return_grad=False, offset=1.e-15):  #  assumes s >=0.  offset is small intensity value
            wh = np.where(s > thr)[0]
            swt = s[wh] - thr
            pen = 0.5*np.sum( swt**2 ) + offset
            if not return_grad:
                return  pen
            else:
                assert (s_grad is not None)
                sgw = s_grad[wh,:]
                gpen = np.sum( sgw.T*swt, axis=1 )
                return pen, gpen
        assert ReOrIm in ['Re','Im']
        re0, im0 = self.CrossFieldNewBasis(np.zeros(a.shape),c0,Vn,return_grad=False)
        if not return_grad:
            Ix = self.PolIntensity(c0 + Vn@a,'X','Hole','phase',False,None)
            re, im = self.CrossFieldNewBasis(a,c0,Vn,return_grad=False)
            #cIx = self.CostHoleDominant(c0 + Vn@a, return_grad=False, scale=1.0)
            den = QuadThreshPenalty(Ix,Ithresh,None,False)
        else:
            Ix, gIx = self.PolIntensity(c0 + Vn@a,'X','Hole','phase',True,None)
            re, im, dre, dim = self.CrossFieldNewBasis(a,c0,Vn,return_grad=True)
            den, dden = QuadThreshPenalty(Ix,Ithresh,gIx,True)
            dden = Vn.T@dden
        if ReOrIm == 'Re':
            num = np.sum((re-re0)**2)
        else:
            num = np.sum((im-im0)**2)
        cost = - num/den  # we want to minimize this ratio
        if not return_grad: return cost*scale
        if ReOrIm == 'Re':
            dnum = 2*dre.T@(re - re0)
        elif ReOrIm == 'Im':
            dnum = 2*dim.T@(im - im0)
        dcost = - dnum/den + (num/den**2)*dden
        return cost*scale, dcost*scale
    def _CostCrossIntensityRatio(self, a, ReOrIm, c0, Vn, return_grad=False):
        assert False
        scale = 1.   # playing with this can help optimization sometimes
        assert ReOrIm in ['Re','Im']
        if not return_grad:
            re, im = self.CrossFieldNewBasis(a,c0,Vn,return_grad=False)
            cIx = self.CostHoleDominant(c0 + Vn@a, return_grad=False, scale=1.0)
        else:
            re, im, dre, dim = self.CrossFieldNewBasis(a,c0,Vn,return_grad=True)
            cIx, dcIx = self.CostHoleDominant(c0 + Vn@a, return_grad=True, scale=1.0)
            dcIx = Vn.T@dcIx
        if ReOrIm == 'Re':
            num = np.sum(re*re)
        else:
            num = np.sum(im*im)
        cost = - num/cIx  # we want to maximize this ratio
        if not return_grad: return cost*scale
        if ReOrIm == 'Re':
            dnum = 2*dre.T@re
        elif ReOrIm == 'Im':
            dnum = 2*dim.T@im
        dcost = - dnum/cIx + (num/cIx**2)*dcIx
        return cost*scale, dcost*scale

    #This cost fcn does modulate the cross field much because most of the gradient comes
    #  from the dominant field
    def _CostCrossDiffIntensityRatio(self, a, ReOrIm, c0, Vn, return_grad=False, scale=1.e8):
        assert False
        assert ReOrIm in ['Re','Im']
        re0, im0 = self.CrossFieldNewBasis(np.zeros(a.shape),c0,Vn,return_grad=False)
        if not return_grad:
            re, im = self.CrossFieldNewBasis(a,c0,Vn,return_grad=False)
            cIx = self.CostHoleDominant(c0 + Vn@a, return_grad=False, scale=1.0)
        else:
            re, im, dre, dim = self.CrossFieldNewBasis(a,c0,Vn,return_grad=True)
            cIx, dcIx = self.CostHoleDominant(c0 + Vn@a, return_grad=True, scale=1.0)
            dcIx = Vn.T@dcIx
        if ReOrIm == 'Re':
            num = np.sum((re-re0)**2)
        else:
            num = np.sum((im-im0)**2)
        cost = - num/cIx  # we want to minimize this ratio
        if not return_grad: return cost*scale
        if ReOrIm == 'Re':
            dnum = 2*dre.T@(re - re0)
        elif ReOrIm == 'Im':
            dnum = 2*dim.T@(im - im0)
        dcost = - dnum/cIx + (num/cIx**2)*dcIx
        return cost*scale, dcost*scale

    #This is a cost function for a bright hill in the cross polarization ('Y')
    #c - DM command vector
    #scale - setting this to 10^? seems to help the Newton-CG minimizer
    def _CostHillCross(self, c, return_grad=True, scale=1.e11):
        assert False  # this approach doesn't work
        self.CostScale = scale
        assert np.iscomplexobj(c) is False
        assert c.shape == (self.Sx.shape[1],)
        if return_grad:
            I, dI = self.PolIntensity(c,XorY='Y',region='Hole',DM_mode='phase',return_grad=True)
            cost = - np.sum(I)
            dcost = - np.sum(dI, axis=0)
            return (scale*cost, scale*dcost)
        else:
            I     = self.PolIntensity(c,XorY='Y',region='Hole',DM_mode='phase',return_grad=False)
            cost = - np.sum(I)
            return scale*cost

    #This calculates something like signal-to-noise ratio for the cross polarization
    #metric - the various options correspond to:
    #  'A' - Iy/(const + Ix)
    def _CostCrossSNR(self,c, metric='A'):  # passed gradient test 2/5/24
        assert False # this approach doesn't seem to work
        const = 1.e-9
        Ix, gIx = self.PolIntensity(c,'X')
        Iy, gIy = self.PolIntensity(c,'Y',SpeckleFactor=0.)
        if metric == 'A':
            s = np.sum(Iy/(const + Ix))
            gs = gIy.T/(const + Ix) - gIx.T*Iy/( (const + Ix)**2 )
            gs = np.sum(gs,axis=1)
        else: assert False
        return (-1*s, -1*gs)  #make it loss function instead of a profit function

    #This uses the Newton Conjugate Grad method to find minima of a function
    #   from a number of random starting points.
    def _FindCrossMinima(self):
        assert False  # this approach doesn't seem to work
        funB = self._CostCrossSNR
        options={'disp':False, 'maxiter':15}
        rnd = lambda scale : scale*(2*np.pi*np.random.rand(self.ndm**2) - np.pi)
        nholes = 9
        command_dh = []
        cost_dh = []
        command_c = []
        cost_c = []
        for k in range(nholes):
          out = optimize.minimize(self.CostHoleDominant,rnd(1.),args=(),method='Newton-CG',options=options,jac=True)
          command_dh.append(out['x'])
          cost_dh.append(out['fun'])
          out = optimize.minimize(self.CostCrossSNR,out['x'],args=(),method='Newton-CG',options=options,jac=True)
          command_c.append(out['x'])
          cost_c.append(out['fun'])

        return {'dh_commands': command_dh, 'cross_commands': command_c, 'dh_cost': cost_dh, 'cross_cost': cost_c}
#===============================================================================
#                      EFC Class ends here
#==============================================================================

#this creates the b-spline basis coefficients corresponding to a linear phase,
#  which behaves as an off-axis source.
#For now, the off axis beam is displacesd on the x-axis
#   - Warning!  this code and BS.BivariateCubicSPline disagree on the
#   x and y directions
#nc - the number of cycles per pupil
#ndm - the DM is ndm x ndm
def SplCoefs4LinearPhase(nc, ndm):
   s = np.linspace(-np.pi, np.pi, 13*ndm)
   sx, sy = np.meshgrid(s,s)
   lp = np.exp(1j*nc*sy)  # linear phase function
   sx = sx.flatten()
   sy = sy.flatten()
   lp = lp.flatten()

   #now set up cubic b-spline
   cbs = BS.BivariateCubicSpline(sx, sy, ndm)
   spco = cbs.GetSplCoefs(lp)
   return spco


#This makes a list of the 1D pixel indices corresponding to a rectangular
# region (specified by its corners) within a flattened 2D array.
#corners - list (or array) corresponding to the 2D coords of the corners of the
# desired region in the following order [Xmin, Xmax, Ymin, Ymax].  The boundaries
# are inclusive.
#BigArrayShape a tuple (or whatever) corresponding the shape of the larger array
def MakePixList(corners, BigArrayShape):
    assert len(BigArrayShape) == 2
    assert corners[2] <= BigArrayShape[0] and corners[3] <= BigArrayShape[0]
    assert corners[0] <= BigArrayShape[1] and corners[2] <= BigArrayShape[1]
    pixlist = []
    rows = np.arange(corners[2], corners[3]+1)  # 'y'
    cols = np.arange(corners[0], corners[1]+1)  # 'x'
    ff = lambda r,c : np.ravel_multi_index((r,c), (BigArrayShape[0],BigArrayShape[1]))
    for r in rows:
        for c in cols:
            pixlist.append(ff(r,c))
    return pixlist
