#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 10:13:48 2025
@author: Richard Frazin


This tests some my ideas for iterative Jacobian improvement for EFC

"""

import numpy as np
from os import path as ospath  #needed for isfile(), join(), etc.
from sys import path as syspath
import random
import time
from scipy.optimize import minimize as mize
import torch
import matplotlib.pyplot as plt



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
LittleJac = True  # only consider a certain set of pixels for the phase/amp screen
if LittleJac:
    pl = MakePixList([97,160,97,160],(256,256))  # take a central square
    sqnpl = int(np.sqrt(len(pl)))  #

machine = "homeLinux"
#machine = "officeWindows"
if machine == "homeLinux":
    MySplineToolsLocation = "/home/rfrazin/Py/Optics"
    PropMatLoc = "/home/rfrazin/Py/EFCSimData/"

elif machine == 'officeWindows':
    MySplineToolsLocation = "E:/Python/Optics"
    PropMatLoc = "E:/MyOpticalSetups/EFC Papers/DataArrays"
syspath.insert(0, MySplineToolsLocation)

#Jacobian files
#fnjac = ospath.join(PropMatLoc,"Jacobian_TorchThreeOAPmodel256x256_23x23_complex.npy") #"DataArrays/Jacobian_TorchThreeOAPmodel256x256_65x65_complex.npy")
#Nominal Jacobian
fnjacn = ospath.join(PropMatLoc, 'JacobianTorch3OAP_64x64pix_23x23spl_Nominal_complex.npy')
jacn = np.load(fnjacn)  #
#perturbed Jocobian
fnjacp = ospath.join(PropMatLoc, 'JacobianTorch3OAP_64x64pix_23x23spl_Perturbed_complex.npy')
jacp = np.load(fnjacp)  #
# filename with intensity values
fnIntensity = ospath.join(PropMatLoc,"Intensity_NoNoise_JacobianTorch3OAP_64x64pix_23x23spl_Perturbed_complex.npy")
#jacn = jacn[pl,:]  # extract the central pixels

#%% make a perturbed jacobian
if False:
   pertsc = 2.e-2  # pertubation scale
   jacp = pertsc*np.abs(jacn).max()*(np.random.randn(jacn.shape[0],jacn.shape[1]) +
                                  1j*np.random.randn(jacn.shape[0],jacn.shape[1]))
   jacp += jacn
   f = np.sum(jacp,axis=1).reshape((sqnpl,sqnpl))
   plt.figure();plt.imshow(np.abs(f),cmap='seismic',origin='lower');plt.colorbar();
#%%
#This class performs various tasks for probe based measurement of the Jacobian for EFC
#JacInit - complex valued, initial guess of the Jacobian to start the optimization
#JacTrue - complex valued, the true Jacobian used to simulate the measurements
#nModAngles - the number of angles used modulated each DM actuator
#defaultDMC - the default command (phase angle in this case) for the DMs that are not being modulated in a given measurement
class EmpiricalJacobian():
   def __init__(self, JacInit, JacTrue, nModAngles=8, defaultDMC=0., Torch=False):
      self.Torch = False
      if JacInit.shape != JacTrue.shape:
         raise Exception(f"JacInit [shape: {JacInit.shape}] and JacTrue [shape: {JacTrue.shape}] are not compatible.")
      if not ((np.iscomplexobj(JacInit)) and (np.iscomplexobj(JacTrue))):
         raise ValueError("Input args JacInit and JacTrue must be complex valued.")
      if Torch:
         if not torch.cuda.is_available():
            raise Exception("cuda not available.  Rerun with the Torch=False option.")
         else:
            self.Torch = True
            self.device = "cuda"
      self.dataset = None
      self.JacInit = JacInit
      self.JacTrue = JacTrue
      self.nModAngles = nModAngles
      self.angles = np.linspace(0, 2*np.pi*(nModAngles-1)/nModAngles, nModAngles)
      self.defaultDMC = defaultDMC
      na = JacInit.shape[1]
      self.na1D = int(np.round(np.sqrt(na)))
      if not np.isclose(np.sqrt(na), na/round(np.sqrt(na))):
         raise Exception("Expecting square array of actuators.  See self.Cost.")
      self.maxjac = np.max(np.abs(JacTrue))
      return(None)

   #This gets the intensity data used to estimate the Jacobian
   #fileWpath  filename, including the path specifying the .npy file
   #mode - either 'load' or 'create' the array containing the intensity data
   #noiseModel - not yet implemented
   # returns -  The array containg the data is placed in self.dataset
   def GetOrMakeIntensityData(self, fileWpath, mode='load',noiseModel='PoissonPlusRead'):
      if mode not in ['load','create']:
         raise ValueError("kwarg mode must be 'load' or 'create'.")
      if fileWpath[-4:] != ".npy":
         raise ValueError("fileWpath must be an .npy file.")
      if mode == 'create':
         if not ospath.isfile(fileWpath):
            yn = input(f"The file {fileWpath} does not exist.  Create it [y/n]? ")
            if yn not in ['y','Y','yes','Yes','Si','si','oui','Oui']:
               print("Exiting.  If you want to load a data file, set the mode kwarg to 'load'.")
               return(None)
         else:
            yn = input(f"The file {fileWpath} already exists.  Overwrite it [y/n]? ")
            if yn not in ['y','Y','yes','Yes','Si','si','oui','Oui']:
               print("No new file made.  Exiting.")
               return(None)
      else:  # mode = 'load'
         if not ospath.isfile(fileWpath):
            print(f"file {fileWpath} not found.  Exiting.")
            return(None)
         else:
            self.dataset = np.load(fileWpath)
            return(None)

      #  create the intensity data set
      jact = self.JacTrue
      self.dataset = np.zeros((jact.shape[0],jact.shape[1],self.nModAngles))
      for kc in range(jact.shape[1]):  # loop over actuators
         dmp = np.exp(1j*np.ones((jact.shape[1],))*self.defaultDMC) # phases for non-modulated actuators
         for kg in range(len(self.angles)):  # phasor for modulated actuator
            dmp[kc] = np.exp(1j*self.angles[kg])
            for kp in range(jact.shape[0]): # pixel loop
               u = jact[kp,:]@dmp
               self.dataset[kp, kc, kg] = np.real(u*np.conj(u))  # put in the intensity
      np.save(fileWpath,self.dataset)

   #This is a cost function to enable gradient-based optimization of a given row of the Jacobian
   # (see self.IntensityModel).  This assumes the data are stored in the class variable self.dataset
   #row - the vector over whose values the cost function will be optimized
   #    it is a row of the Jacobian matrix.  It is real valued: [real part, imag part]
   #pixind - the detector pixel index corresponding to the row number of the Jacobian being optimized
   #regparam - scalar regularization parameter on the square difference between row and nominal value of row
   #centercon - force the imag component of the element corresp. to the central actuator to be zero.
   def Cost(self, row, pixind, regparam=0., centercon=False, return_grad=False):
      if self.dataset is None:
         raise Exception("The dataset needs to be loaded as a member variable.  See self.GetOrMakeIntensityData.")
      if regparam < 0.:
               raise ValueError("input 'regparam' must be >= 0.")
      nact = self.JacTrue.shape[1]  # number of DM actuators
      if not self.Torch:
         cost = 0.
         if return_grad:
            dcost = np.zeros(len(row))
         for ka in range(nact): # loop over actuators
            for kt in range(len(self.angles)):
                ykakt = self.dataset[pixind, ka, kt]
                if return_grad:
                  I, dI = self.IntensityModel(row, ka, self.angles[kt], return_grad=True)
                else:
                  I     = self.IntensityModel(row, ka, self.angles[kt], return_grad=False)
                cost += 0.5*(I - ykakt)**2
                if return_grad:
                  dcost += (I - ykakt)*dI

         if centercon:
             kc = np.ravel_multi_index((self.na1D//2,self.na1D//2) ,(self.na1D, self.na1D))
             cost += 0.5*self.maxjac*row[self.na1D + kc]**2  # imag comp of element corresp. the central actuator
             if return_grad:
                dcost[self.na1D + kc] += self.maxjac*row[self.na1D + kc]
         if regparam > 0.:
             rowinit = np.concatenate((np.real(self.JacInit[pixind,:]), np.imag(self.JacInit[pixind,:])))
             cost  += 0.5*regparam*np.sum((row - rowinit)**2)
             if return_grad > 0:
                dcost += regparam*(row - rowinit)
         if return_grad:
            return (cost, dcost)
         else:
            return(cost)
      else: # self.Torch is True  this in PyTorch, leveraging the autograd setup in self.IntensityMdoel
         device = self.device
         row_torch = torch.tensor(row, dtype=torch.float32, device=device, requires_grad=True)
         cost_torch = torch.tensor(0.0, dtype=torch.float32, device=device)

         for ka in range(nact):
            for kt in range(len(self.angles)):
               ykakt = torch.tensor(self.dataset[pixind, ka, kt], dtype=torch.float32, device=device)
               I = self.IntensityModel(row_torch, ka, self.angles[kt], return_grad=False)
               cost_torch += 0.5 * (I - ykakt)**2

         if centercon:
            kc = np.ravel_multi_index((self.na1D//2, self.na1D//2), (self.na1D, self.na1D))
            center_val = row_torch[self.na1D + kc]
            cost_torch += 0.5 * self.maxjac * center_val**2

         if regparam > 0.:
            rowinit = np.concatenate((np.real(self.JacInit[pixind,:]), np.imag(self.JacInit[pixind,:])))
            rowinit_torch = torch.tensor(rowinit, dtype=torch.float32, device=device)
            diff = row_torch - rowinit_torch
            cost_torch += 0.5 * regparam * torch.sum(diff**2)

         if return_grad:  #return numpy array because the output is already differentiable
            cost_torch.backward()
            grad = row_torch.grad.detach().cpu().numpy()
            return cost_torch.item(), grad
         else:
            return cost_torch


   #This calls optimization routines to optimize a row of the Jacobian
   #pixind - the detector pixel index (see self.CostIntensity)
   #method - optimization method.  Must be one of: ['CG' ]
   #startpoint - the starting guess for the optimization.  If None, self.JacInit is used
   def RowOptimize(self, pixind, method='CG', startpoint=None, Torchlearnrate=1.e-3,TorchAdamIters=10,use_lbfgs=True):
      if pixind < 0 or pixind >= self.JacInit.shape[0]:
         raise IndexError(f"pixind {pixind} is out of bounds for Jacobian with {self.JacInit.shape[0]} rows.")
      if startpoint is None:
            startpoint = np.concatenate((np.real(self.JacInit[pixind,:]),np.imag(self.JacInit[pixind,:])))
      if not self.Torch:  # use numpy
         ops = {'disp':False, 'maxiter':50}
         fun = lambda row: self.Cost(row, pixind, return_grad=True)
         out = mize(fun, startpoint, args=(), method=method, jac=True, options=ops)
         return((out['x'], out['fun']))
      else:   # use PyTorch
         rowtorch = torch.tensor(startpoint, dtype=torch.float32, device=self.device, requires_grad=True)
         optimizer = torch.optim.Adam([rowtorch], lr=Torchlearnrate)
         for i in range(TorchAdamIters):
            optimizer.zero_grad()  # Réinitialiser les gradients à chaque itération
            cost = self.Cost(rowtorch, pixind, return_grad=False)
            cost.backward()  # Rétropropagation pour calculer les gradients
            optimizer.step()  # parameter update
         #mid_result = rowtorch.detach().cpu().numpy()
         if use_lbfgs:
              def closure():
                 optimizer_lbfgs.zero_grad()
                 cost, _ = self.Cost(rowtorch, pixind, return_grad=True)
                 cost.backward()
                 return cost
              optimizer_lbfgs = torch.optim.LBFGS([rowtorch], max_iter=20, tolerance_grad=1e-5, line_search_fn='strong_wolfe')
              optimizer_lbfgs.step(closure)
         final_row = rowtorch.detach().cpu().numpy()
         final_cost = self.Cost(rowtorch.detach(), pixind, return_grad=False).item()
         return (final_row, final_cost)


         return rowtorch.detach().cpu().numpy()  #numpy output

   def LoopOverPixels(self):
      self.GetOrMakeIntensityData(fnIntensity,'load')
      nact = self.JacInit.shape[1]
      npix = self.JacInit.shape[0]
      jacnew = np.zeros(self.JacInit.shape).astype('complex')
      cost = []
      timestart = time.time()

      for kp in range(npix):
         out = self.RowOptimize(kp)
         if not self.Torch:
            row = out[0]; cost.append(out[1])
         else:  # self.Torch is True
            row = out
            optinfo = None
         jacnew[kp,:] = row[:nact] + 1j*row[nact:]
         if kp == 0:
            print(f"First pixel of {npix} done.  Estimated total time is {npix*(time.time()-timestart)/3600} hours.")
         if np.remainder(kp,100) == 0:
            print(f"Pixel {kp} of {npix} done.  Time so far is {(time.time()-timestart)/3600} hours.")
      return( (jacnew, optinfo) )

   #This returns the intensity for a specified pixel index (1D(
   #row - the row vector containing the estimated jacobian for the pixel in question
   #   it is a real-valued vector of length of twice the number of actuators (real and imag parts)
   #actnum - the index of the actuator being modulated
   #actphase - the phase of modulated actuator
   #return_grad - also return the gradient w.r.t to 'row'
   #   the gradient is a real-valued vector of length of twice the number of actuators
   def IntensityModel(self,row,actnum,actphase,return_grad=False):
      if self.defaultDMC != 0.:
         raise Exception("The default phase (self.DefaultDMC) must be zero for this formulation.")
      if len(row) != 2*self.JacTrue.shape[1] or row.ndim != 1:
         raise ValueError(f"the argument row must be 1D and have length {2*self.JacTrue.shape[1]}.  It has shape {row.shape}.")
      na = self.JacTrue.shape[1]  # number of actuators
      if not self.Torch:
         dmp = np.exp(1j*np.ones((na,))*self.defaultDMC) # phases for non-modulated actuators
         dmp[actnum] = np.exp(1j*actphase)  # treat the modulated actuator
         u = (row[:na] + 1j*row[na:])@dmp
         uu = np.real( u*np.conj(u) )
         if not return_grad:
            return(uu)
         else: # determine and return the gradient w.r.t. row
            grad = np.zeros((len(row),))

            #quadratic term
            grad[actnum]      =  2.*row[actnum] #quadratic term
            grad[actnum + na] = 2.*row[actnum + na]  #quadratic term

            #the R_m and Zm terms
            stm = np.sin(actphase); ctm = np.cos(actphase);
            Smr = np.sum(row[:na]) - row[actnum]  # real part
            Smi = np.sum(row[na:]) - row[na + actnum]  #imag part
            grad[actnum]      +=  2.*Smr*ctm + 2.*Smi*stm
            grad[actnum + na] += -2.*Smr*stm + 2.*Smi*ctm
            unir = np.ones((na,))*2.*(row[actnum]*ctm + row[actnum + na]*stm + Smr)  # the last term corresponds to Zm
            unii = np.ones((na,))*2.*(row[actnum]*stm - row[actnum + na]*ctm + Smi)  # the last term corresponds to Zm
            unii[actnum] = 0.; unir[actnum]= 0.
            grad[:na] += unir
            grad[na:] += unii

            return((uu, grad))
      else:  #  self.Torch is True, implement the intensity in PyTorch using autograd
            device = self.device
            rowtorch = torch.tensor(row, dtype=torch.float32, device=device, requires_grad=True)
            real = rowtorch[:na]; imag = rowtorch[na:]
            complex_row = torch.complex(real, imag)
            dmp = torch.ones(na, dtype=torch.complex64, device=device)
            dmp *= torch.exp(1j * torch.tensor(self.defaultDMC, device=device))
            dmp[actnum] = torch.exp(1j * torch.tensor(actphase, device=device))
            u = torch.dot(complex_row, dmp)
            uu = torch.real(u * torch.conj(u))
            if not return_grad:  # this is needs to be kept in the form of a differentiable torch tensor
               return uu
            else:  # return a numpy array containing the gradient
               uu.backward()
               grad = rowtorch.grad.detach().cpu().numpy()
               return (uu.item(), grad)



#%%

if False: # prepare dataset for the iteration scheme below
   acts = np.arange(jacn.shape[1])  # actuator indices
   angles = np.linspace(0, 2*np.pi*15/16, 16)
   n_iter = 3  # number of (more-or-less) gradient steps

   obs = np.zeros((jacn.shape[0], jacn.shape[1], len(angles)))  #intensity data - modulate each actuator independently!

   for kt in range(len(acts)):  # actuator loop
      actphases  = np.zeros(jacn.shape[1])  # initial phases of actuators
      actphasors = np.exp(1j*actphases)
      for ka in range(len(angles)): # data collection loop
          actphases[ kt] = angles[ka]
          actphasors[kt] = np.exp(1j*angles[ka])
          obs[:,kt,ka] = np.abs(jacp@actphasors)**2
   ObsAngleMean = np.mean(obs, axis=2)
   for kp in range(obs.shape[0]): # loop over pixels
     for kt in range(len(acts)):
        obs[kp,kt,:] -= ObsAngleMean[kp,kt]


#%% this simiple iteration scheme works surprisingly well
   tstart = time.time()
   for ki in range(n_iter):
      acts = np.arange(jacn.shape[1])  # actuator indices
      random.shuffle(acts)  # shuffle the order
      for kt in range(len(acts)):
         act = acts[kt]  # index of actuator
         for kp in range(obs.shape[0]):
            s =  jacp[kp, :act  ]@actphasors[:act]
            s += jacp[kp, act+1:]@actphasors[act+1:]
            sr = np.real(s); si = np.imag(s)
            #assert False  # replace jacp with jaco

            # regression block.  the pinv will be applied to mat.
            mat = np.zeros((len(angles),2))
            y = np.zeros((len(angles),))  # vector of measurements

            for ka in range(len(angles)):
               y[ka] = obs[kp,act,ka]
               mat[ka, 0] =  sr*np.cos(angles[ka]) + si*np.sin(angles[ka])
               mat[ka, 1] = -sr*np.sin(angles[ka]) + si*np.cos(angles[ka])
            mat *= 2
            jachat = np.linalg.pinv(mat)@y
            jacn[kp,act] = jachat[0] + 1j*jachat[1]  # jacobian update
      print(f"iterion {ki} complete.  Total time is {(time.time()-tstart)/60} minutes.")
      plt.figure(); plt.imshow(np.abs(np.sum(jacn,axis=1)).reshape((62,62)),cmap='seismic',origin='lower');plt.colorbar();

if False:  # check algebra
#%%
   from sympy import symbols, cos, sin, I, expand, diff, simplify, re, im, conjugate

   # Define symbols
   dpr, dpi, spr, spi, theta_l = symbols('dpr dpi spr spi theta_l', real=True)
   dlr, dli = symbols('dlr dli', real=True)  # For l ≠ n

   # Define complex values
   dpn = dpr + I * dpi
   Sn = spr + I * spi
   exp_jtheta = cos(theta_l) + I * sin(theta_l)

   # Define R_n = Re[dpn * exp(j*theta_l) * conjugate(Sn)]
   Rn_complex = dpn * exp_jtheta * conjugate(Sn)
   Rn = re(Rn_complex)

   # Compute partial derivatives of Rn
   dR_d_dpr = simplify(diff(Rn, dpr))
   dR_d_dpi = simplify(diff(Rn, dpi))
   dR_d_dlr = simplify(diff(Rn, dlr).subs({dlr: dpr, dli: dpi}))  # l ≠ n placeholder
   dR_d_dli = simplify(diff(Rn, dli).subs({dlr: dpr, dli: dpi}))
