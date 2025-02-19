#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:59:53 2025
@author: Richard Frazin


This creates a model of square stellar coronagraph for my JATIS article for
  the purposes of phase retrieval.


"""
import numpy as np
import torch
import TorchFourierOptics as TFO

GPU = True
if GPU and torch.cuda.is_available():
   device = 'cuda'
else:
   device = 'cpu'
   GPU = False
   if not torch.cuda.is_available(): print("cuda is not available.  CPU implementation.")
theseparameters = {'wavelength': 0.9, 'max_chirp_step_deg':30.0}
F = TFO.TorchFourierOptics(params=theseparameters, GPU=GPU)

#%%
#set up initial screen
sz = 65  # phase/amp screen will be sz-by-sz
pts = torch.ones((2,sz,sz), requires_grad=True).to(device)  # channel 0 is for the real part, channel 1 is for the imaginary part
pts[0] *= 0.0
#set up initial 1D coordinate corresponding to pts
L0 = 20.515e3; dx0 = L0/sz;  x0 = torch.linspace( - L0/2 + dx0/2, L0/2 - dx0/2, sz).to(device)  # units are microns
# resample to high res
sz1 = 512
dx01 = L0/sz1; x01 = torch.linspace( - L0/2 + dx01/2, L0/2 - dx01/2, sz1).to(device)
g = F.ResampleField2D(pts,x0,x01)
#create the soft edge on the plane wave source in VLF
g = F.ApplyStopAndOrAperture(g, x01 ,-1. , d_ap=L0, shape='square', smoothing_distance=1210.)
#%%

#now propagate to the occulter plane
L1 = 2250.; dx1 = 3; x1 = torch.linspace(-L1/2 + dx1/2, L1/2 - dx1/2 , int(L1//dx1)).to(device)
g = F.FFT_prop(g,x01,x1, 0.8e6, dist_in_front=8.5e4)

#apply stop and aperture in occulter plane
g = F.ApplyStopAndOrAperture(g, x1 , 142., d_ap=2250., shape='square', smoothing_distance=71.)
