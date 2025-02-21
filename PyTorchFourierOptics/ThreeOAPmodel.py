#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:59:53 2025
@author: Richard Frazin


This creates a model of square stellar coronagraph for my JATIS article for
  the purposes of phase retrieval.


"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import TorchFourierOptics as TFO

GPU = False
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
pts = torch.stack([
    torch.ones((sz, sz), device=device),  # Parte real inicializada en 1
    torch.zeros((sz, sz), device=device)  # Parte imaginaria inicializada en 0
], dim=0)  #  we will ultimately be taking the gradient with respect to this
crap = pts.requires_grad_(); del(crap)
#%%

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
L1 = 2250.; dx1 = 4; x1 = torch.linspace(-L1/2 + dx1/2, L1/2 - dx1/2 , int(L1//dx1)).to(device)
g = F.FFT_prop(g,x01,x1, 0.8e6, dist_in_front=8.5e4)

#apply stop and aperture in occulter plane
g = F.ApplyStopAndOrAperture(g, x1 , 142., d_ap=2250., shape='square', smoothing_distance=71.)

# Take an optical FT to arrive at the Lyot stop
L2 = 8.192e3; dx2 = 16.; x2 =  torch.linspace(-L2/2 + dx2/2, L2/2 - dx2/2 , int(L2//dx2)).to(device)
g = F.FFT_prop(g,x1,x2, 0.4e6, None)
#apply Lyot Stop
g = F.ApplyStopAndOrAperture(g,x2,-1., d_ap=7.9e3   , shape='square', smoothing_distance=150.)
F.MakeAmplitudeImage(g,x2,'linear');plt.title("just after Lyot stop");
#gnp_Lyot = F.torch2numpy_complex(g)

# Take an optical FT to arrive at the detector
NN3 = 512
L3 = 2200.; dx3 = L3/NN3; x3 = torch.linspace(-L3/2 + dx3/2, L3/2 - dx3/2, NN3)
g = F.FFT_prop(g,x2,x3,4.e5, dist_in_front=1.e5)
F.MakeAmplitudeImage(g,x3,'linear'); plt.title('detector plane');
#gnp_det = F.torch2numpy_complex(g)
