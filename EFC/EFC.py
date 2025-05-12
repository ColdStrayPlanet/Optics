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
from scipy.optimize import minimize, LinearConstraint
from scipy.sparse import lil_matrix

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
    Sxfn = 'ThreeOAP20mmSquareApCgkn33x33_SystemMatrixReducedContrUnits_Ex.npy'
    Syfn = 'ThreeOAP20mmSquareApCgkn33x33_SystemMatrixReducedContrUnits_Ey.npy'
DomMat = np.load(ospath.join(PropMatLoc,Sxfn));
CroMat = np.load(ospath.join(PropMatLoc,Syfn));

#set things up for a 15x15 DM with its phasor represented by a 33x33 B-spline
s = np.linspace(-0.5, 0.5, 165); xx, yy = np.meshgrid(s,s);
Delta11 = 1./11; Delta15 = 1./15; Delta21=1./21; Delta33 = 1./31  # setting delta33 to 1/31 is for padding
#B15 = BS.BivariateCubicSpline(xx.flatten(),yy.flatten(),15,Xmin=-0.5+Delta15,Delta=Delta15)
#B11 = BS.BivariateCubicSpline(xx.flatten(),yy.flatten(),11,Xmin=-0.5+Delta11,Delta=Delta11)
#B15 = BS.BivariateCubicSpline(xx.flatten(),yy.flatten(),15,Xmin=-0.5+Delta15,Delta=Delta15)
B21 = BS.BivariateCubicSpline(xx.flatten(),yy.flatten(),21,Xmin=-0.5+Delta21,Delta=Delta21)
B33 = BS.BivariateCubicSpline(xx.flatten(),yy.flatten(),33,Xmin=-0.5-Delta33,Delta=Delta33)

#%%

class EFC():
   def __init__(self, SpLo, SpHi, DomPropMat, CrossPropMat, Cross2PropMat=None):
      self.SpLo = SpLo
      self.SpHi = SpHi
      self.SysD = DomPropMat
      self.SysC = CrossPropMat
      self.SysC2 = Cross2PropMat
      self.detshape = (int(np.sqrt(DomPropMat.shape[0])), int(np.sqrt(DomPropMat.shape[0])))   # shape of detector
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
      phasor = np.exp(1j*phase)
      HiCo = self.SpHi.imat@phasor
      if not return_grad:
         return(HiCo)
      dphase = self.SpLo.mat
      dphasor = 1j*(dphase.T*phasor).T
      dHiCo = self.SpHi.imat@dphasor
      return( (HiCo, dHiCo) )


   def DMcmd2Field(self, dmc, pmat='dom', pixlist=None, return_grad=False):
      if pmat not in ['dom','cross','cross2']:
         raise ValueError("the string pmat must be one of the allowable choices.")
      if pmat == 'cross2' and self.C2 is None: raise Exception("Cross2PropMat not initialized.")
      if pmat == 'dom': Sys = self.SysD
      if pmat == 'cross': Sys = self.SysC
      if pmat == 'cross2': Sys = self.SysC2
      if pixlist is not None:
         Sys = Sys[pixlist,:]
      if not return_grad:
         co = self.DMcmd2PupilCoefs(dmc)
         field = Sys@co
         return(field)
      else: # return the Jacobian, too
         co, dco = self.DMcmd2PupilCoefs(dmc, return_grad=True)
         field = Sys@co
         jac = Sys@dco
         return ( (field, jac) )

   def DMcmd2Intensity(self, dmc, pmat='dom', pixlist=None, return_grad=False):
      if not return_grad:
         field = self.DMcmd2Field(dmc, pmat, pixlist=pixlist, return_grad=False)
         I = np.real(field*np.conj(field))
         return(I)
      else:
         field, dfield = self.DMcmd2Field(dmc, pmat, pixlist=pixlist, return_grad=True)
         I = np.real(field*np.conj(field))
         dI = 2*np.real( (dfield.T)*np.conj(field)  ).T
         return( (I, dI) )

   #This makes a list of the 1D pixel indices corresponding to a rectangular
   # region (specified by its corners) within a flattened 2D array.
   #corners - list (or array) corresponding to the 2D coords of the corners of the
   # desired region in the following order [Xmin, Xmax, Ymin, Ymax].  The boundaries
   # are inclusive.
   #BigArrayShape a tuple (or whatever) corresponding the shape of the larger array
   def MakePixList(self, corners):
       BigArrayShape=self.detshape
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

   def DigDominantHole(self, dmc, pixlist, DMconstr=np.pi/4):
      N = int(np.sqrt(self.SpLo.Nsp))
      def Cost(dmc, pmat='dom'):
         I, gI = self.DMcmd2Intensity(dmc, pmat=pmat, pixlist=pixlist, return_grad=True)
         cost = np.sum(I)
         gcost = np.sum(gI,axis=0)
         return((cost, gcost))
      def make_gradient_matrix(N):  # this for the inequality constraints on DM command
          rows = []
          cols = []
          data = []
          idx = lambda i, j: i * N + j  # 1D index of actuator (i,j)
          row_id = 0
          for i in range(N):  # ∂q/∂x (horizontal)
              for j in range(N - 1):
                  rows += [row_id, row_id]
                  cols += [idx(i, j), idx(i, j + 1)]
                  data += [-1, 1]
                  row_id += 1
          for i in range(N - 1):  # ∂q/∂y (vertical)
              for j in range(N):
                  rows += [row_id, row_id]
                  cols += [idx(i, j), idx(i + 1, j)]
                  data += [-1, 1]
                  row_id += 1

          lilmatrix = lil_matrix((row_id, N * N))
          for r, c, d in zip(rows, cols, data):
              lilmatrix[r, c] = d
          return lilmatrix.tocsr()
      conmat = make_gradient_matrix(N)
      constraint = LinearConstraint(conmat, -DMconstr, DMconstr)
      ops = {'maxiter':50, 'xtol':1.e-6, 'verbose':2}
      out = minimize(Cost, dmc, jac=True, constraints=[constraint], method='trust-constr',options=ops)

      return out
