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


#%%

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
