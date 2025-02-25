#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:59:53 2025
@author: Richard Frazin


This creates a model of square stellar coronagraph for my JATIS article for
  the purposes of phase retrieval.


"""
import numpy as np
from sys import path
from os.path import isfile
import torch
import TorchFourierOptics as TFO
splinetoolsloc = "../"
path.append(splinetoolsloc)
import Bspline3 as BS

GPU = False  #run on the GPU?
screentype = 'spline'
sz = 65  # phase/amp screen will be sz-by-sz
if GPU and torch.cuda.is_available():
   device = 'cuda'
else:
   device = 'cpu'
   GPU = False
   if not torch.cuda.is_available(): print("cuda is not available.  CPU implementation.")
theseparameters = {'wavelength': 0.9, 'max_chirp_step_deg':30.0}
F = TFO.TorchFourierOptics(params=theseparameters, GPU=GPU)

#set up initial coefficients
spco = torch.stack([  #two channel vector representing the spline coefficients
         torch.ones((sz*sz,), device=device), torch.zeros((sz*sz,), device=device)])
crap = spco.requires_grad_(); del crap  # make spco differentiable


def OAP_Model(spco, screentype=screentype):  # spco is a set of 1D coefficients that specify the complex screen
   if screentype not in ['spline','pixels']:
      raise ValueError("screentype must be 'pixels' or 'spline'.")
   L0 = 20.515e3; dx0 = L0/sz;   #set up initial 1D coordinate
   x0 = torch.linspace( - L0/2 + dx0/2, L0/2 - dx0/2, sz).to(device)  # units are microns
   x0np =  np.linspace( - L0/2 , L0/2 , 3*sz)

   #set up initial screen
   if screentype == 'pixels':
      pts = torch.stack([spco[0].reshape((sz,sz)),
                         spco[1].reshape((sz, sz)) ], dim=0)
   elif screentype == 'spline':
      x0np, y0np = np.meshgrid(x0np,x0np,indexing='ij')
      filename = "SplineInterpMatrix"+str(sz)+"x"+str(sz)+"_L"+str(L0)+".npy"
      if isfile(filename):
         spmat = np.load(filename)
      else:
         Spl = BS.BivariateCubicSpline(x0np.flatten(),y0np.flatten(),sz)
         spmat = Spl.mat
         np.save(filename,spmat)
      spmat = torch.tensor(spmat).to(device)  # Spl.mat (the interpolation matrix) is purely real-valued
      pts = torch.stack([
         (spmat@spco[0]).reshape((len(x0np),len(x0np))),
         (spmat@spco[1]).reshape((len(x0np),len(x0np))) ])
      x0 = torch.tensor(x0np).to(device)
   # resample pts to high res
   sz1 = 512
   dx01 = L0/sz1; x01 = torch.linspace( - L0/2 + dx01/2, L0/2 - dx01/2, sz1).to(device)
   g = F.ResampleField2D(pts,x0,x01)
   #create the soft edge on the plane wave source in VLF
   g = F.ApplyStopAndOrAperture(g, x01 ,-1. , d_ap=L0, shape='square', smoothing_distance=1210.)


   #now propagate to the occulter plane
   L1 = 2250.; dx1 = 4; x1 = torch.linspace(-L1/2 + dx1/2, L1/2 - dx1/2 , int(L1//dx1)).to(device)
   g = F.FFT_prop(g,x01,x1, 0.8e6, dist_in_front=8.5e4)

   #apply stop and aperture in occulter plane
   g = F.ApplyStopAndOrAperture(g, x1 , 142., d_ap=2250., shape='square', smoothing_distance=71.)

   # Take an optical FT to arrive at the Lyot stop
   L2 = 8.192e3; dx2 = 16.; x2 =  torch.linspace(-L2/2 + dx2/2, L2/2 - dx2/2 , int(L2//dx2)).to(device)
   g = F.FFT_prop(g,x1,x2, 0.4e6, None)
   #apply Lyot Stop
   g = F.ApplyStopAndOrAperture(g,x2,-1., d_ap=7.9e3   , shape='square', smoothing_distance=150.)
   #F.MakeAmplitudeImage(g,x2,'linear');plt.title("just after Lyot stop");
   #gnp_Lyot = F.torch2numpy_complex(g)

   # Take an optical FT to arrive at the detector
   NN3 = 512
   L3 = 2200.; dx3 = L3/NN3; x3 = torch.linspace(-L3/2 + dx3/2, L3/2 - dx3/2, NN3)
   g = F.FFT_prop(g,x2,x3,4.e5, dist_in_front=1.e5)
   #F.MakeAmplitudeImage(g,x3,'linear'); plt.title('detector plane');
   #gnp_det = F.torch2numpy_complex(g)

   return g[0,256:266,256:266]
#%%
#Jacobian
Jg = torch.autograd.functional.jacobian(OAP_Model, spco)
