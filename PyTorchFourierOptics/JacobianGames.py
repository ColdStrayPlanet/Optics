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

#Jacobian file
fnjac = ospath.join(PropMatLoc,
         "Jacobian_TorchThreeOAPmodel256x256_23x23_complex.npy") #"DataArrays/Jacobian_TorchThreeOAPmodel256x256_65x65_complex.npy")
jaco = np.load(fnjac)  # original jacobian (ideal jacobian)
jaco = jaco[pl,:]  # extract the central 128x128 pixels

#%% make the perturbed jacobian
pertsc = 1.e-2  # pertubation scale
jacp = pertsc*np.abs(jaco).max()*(np.random.randn(jaco.shape[0],jaco.shape[1]) +
                               1j*np.random.randn(jaco.shape[0],jaco.shape[1]))
jacp += jaco
f = np.sum(jacp,axis=1).reshape((sqnpl,sqnpl))
plt.figure();plt.imshow(np.abs(f),cmap='seismic',origin='lower');plt.colorbar();
#%%
#This class performs various tasks for probe based measurement of the Jacobian for EFC
#JacInit - initial guess of the Jacobian to start the optimization
#JacTrue - the true Jacobian used to simulate the measurements
#nModAngles - the number of angles used modulated each DM actuator
#defaultDMC - the default command (phase angle in this case) for the DMs that are not being modulated in a given measurement
class EmpiricalJacobian():
   def __init__(self, JacInit, JacTrue, nModAngles=8, defaultDMC=0.):
      if JacInit.shape != JacTrue.shape:
         raise Exception(f"JacInit [shape: {JacInit.shape}] and JacTrue [shape: {JacTrue.shape}] are not compatible.")
      self.JacInit = JacInit
      self.JacTrue = JacTrue
      self.nModAngles = nModAngles
      self.angles = np.linspace(0, 2*np.pi*(nModAngles-1)/nModAngles, nModAngles)
      self.defaultDMC = defaultDMC
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
      dmp = np.exp(1j*np.ones((na,))*self.defaultDMC) # phases for non-modulated actuators
      dmp[actnum] = np.exp(1j*actphase)
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
         grad[actnum]      += 2.*Smr*ctm + 2.*Smi*stm
         grad[actnum + na] += 2.*Smr*stm - 2.*Smi*ctm
         unir = np.ones((na,))*2.*(row[actnum]*ctm + row[actnum + na]*stm + Smr)  # the last term corresponds to Zm
         unii = np.ones((na,))*2.*(row[actnum]*stm - row[actnum + na]*ctm + Smi)  # the last term corresponds to Zm
         unii[actnum] = 0.; unir[actnum]= 0.
         grad[:na] += unir
         grad[na:] += unii

         return((uu, grad))


#%%

if False: # prepare dataset for the iteration scheme below
   acts = np.arange(jaco.shape[1])  # actuator indices
   angles = np.linspace(0, 2*np.pi*15/16, 16)
   n_iter = 3  # number of (more-or-less) gradient steps

   obs = np.zeros((jaco.shape[0], jaco.shape[1], len(angles)))  #intensity data - modulate each actuator independently!

   for kt in range(len(acts)):  # actuator loop
      actphases  = np.zeros(jaco.shape[1])  # initial phases of actuators
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
      acts = np.arange(jaco.shape[1])  # actuator indices
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
            jaco[kp,act] = jachat[0] + 1j*jachat[1]  # jacobian update
      print(f"iterion {ki} complete.  Total time is {(time.time()-tstart)/60} minutes.")
      plt.figure(); plt.imshow(np.abs(np.sum(jaco,axis=1)).reshape((62,62)),cmap='seismic',origin='lower');plt.colorbar();
