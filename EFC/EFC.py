#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 11:37:56 2025

@author: rfrazin
"""

import numpy as np
from os import path as ospath  #needed for isfile(), join(), etc.
from sys import path as syspath
import warnings
from scipy import optimize

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


#set things up for a 15x15 DM with its phasor represented by a 33x33 B-spline grid.

s = np.linspace(-0.5, 0.5, 165); xx, yy = np.meshgrid(s,s);
Delta11 = 1./11; Delta15 = 1./15; Delta21=1./21; Delta33 = 1./31  # setting delta33 to 1/31 is for padding
#B15 = BS.BivariateCubicSpline(xx.flatten(),yy.flatten(),15,Xmin=-0.5+Delta15,Delta=Delta15)
#B11 = BS.BivariateCubicSpline(xx.flatten(),yy.flatten(),11,Xmin=-0.5+Delta11,Delta=Delta11)
#B15 = BS.BivariateCubicSpline(xx.flatten(),yy.flatten(),15,Xmin=-0.5+Delta15,Delta=Delta15)
B21 = BS.BivariateCubicSpline(xx.flatten(),yy.flatten(),21,Xmin=-0.5+Delta21,Delta=Delta21)
B33 = BS.BivariateCubicSpline(xx.flatten(),yy.flatten(),33,Xmin=-0.5-Delta33,Delta=Delta33)


class EFC():
   def __init__(self, SpLo, SpHi, DomPropMat, CrossPropMat, Cross2PropMat=None):
      self.SpLo = SpLo
      self.SpHi = SpHi
      self.SysD = DomPropMat
      self.SysC = CrossPropMat
      self.SysC2 = Cross2PropMat
      if DomPropMat.shape != CrossPropMat.shape: raise ValueError("DomPropMat and CrossPropMat must have the same shape.")
      if (Cross2PropMat is not None) and (): raise ValueError("The two CrossProp matrices must have the same shape.")
      if SpHi.Nsp != DomPropMat.shape[1]: raise ValueError("The hi-res spline and the propagation matrix must be consistent.")
      return None


   #this takes a DM command on the low-res spline grid and calculates the the spline coefs
   #  on the hi-res grid.
   #dmc - the DM command (flattened), phase units are assumed.  Must have len == SpLo.Nsp
   #SpLo - Bspline3.BivariateCubicSpline object corresonding to the DM height interpolation
   #SpHi - Bspline3.BivariateCubicSpline object used to represent the continuous DM phasor on the Bspline basis representing the pupil field
   def DMcmd2PupilCoefs(self, dmc, return_grad=False):
      if dmc.ndim !=1: raise ValueError(f"input param dmc must have 1 dimension. It has shape {dmc.shape}.")
      if np.iscomplexobj(dmc): raise ValueError("input param dmc must be real-valued")
      if len(dmc) != self.SpLo.Nsp: raise ValueError(f"input param dmc must have length {self.SpLo.Nsp}. It has length {len(dmc)}.")
      phase = self.SpLo.SplInterp(dmc)
      if return_grad:
         dphase = self.SpLo.mat
      phasor = np.exp(1j*phase)
      if return_grad:
         dphasor = 1j*dphase*phasor
      HiCo = self.SpHi.GetSplCoefs(phasor)
      if return_grad:
         dHiCo = self.SpHi.GetSplCoefs(dphasor)
         return( (HiCo, dHiCo) )
      return(None)


   def DMcmd2DetField(self, dmc, pmat='dom', return_grad=False):
      if pmat not in ['dom','cross','cross2']:
         raise ValueError("the string pmat must be one of the allowable choices.")
      if pmat == 'cross2' and self.C2 is None: raise Exception("Cross2PropMat not initialized.")
      if pmat == 'dom': Sys = self.SysD
      if pmat == 'cross': Sys = self.SysC
      if pmat == 'cross2': Sys = self.SysC2
      if not return_grad:
         co = self.DMcmd2PupilCoefs(dmc)
         field = Sys@co
         return(field)
      else: # return the Jacobian, too
         co, dco = self.DMcmd2PupilCoefs(dmc, return_grad=True)
         field = Sys@co
         jac = Sys@dco
         return ( (field, jac))
