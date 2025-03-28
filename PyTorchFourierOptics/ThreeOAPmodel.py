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
from sys import path
from os.path import isfile
import time
import torch
import TorchFourierOptics as TFO
splinetoolsloc = "../"
path.append(splinetoolsloc)
import Bspline3 as BS

GPU = True  #run on the GPU?
differentiable = False  # if True, the model will be differentiable with respect to the input tensor
screentype = 'spline'  # 'pixels' or 'spline' a 41x41 'pixel' grid gives very uniform entrance field (with unity coefs)
sz =  23  # phase/amp screen will be sz-by-sz
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
# if the model is linear, the Jacobian can be found by propagating the inputs one pixel (or spline coef) at a time
if differentiable:
   crap = spco.requires_grad_(); del crap  # make spco differentiable
#%%

#This is the differentiable OAP model function
#spc - 1D vector (i.e., flattened) of coefficients specifying either the square pixels or spline coefficients in the screen
#       can either be a complex-valued np.array or a 2-channel torch.array
#slice_indices - specifies the spatial indices of the output tensor (corresponding to the image plane) to be returned.
#   This may be desirable when calculating the gradient with respect to the input tenstor (spc)
#    It is a list, tuple or array with 4 elements.  The spatial indices will be used as: output[:, si[0]:si[1], si[2]:si[3]]  (see code)
#  if None - no slicing is applied and full spatial extent is returnd.
def OAP_Model(spc, slice_indices=None, screentype=screentype):  # spc is a set of 1D coefficients that specify the complex screen
   if slice_indices is not None:
       if len(slice_indices) != 4:
           raise ValueError("The parameter slice_indices must consist of 4 numbers.")
   if screentype not in ['spline','pixels']:
      raise ValueError("screentype must be 'pixels' or 'spline'.")
   if not isinstance(spc, torch.Tensor):  # handle numpy.array input
        spct = torch.stack([
        torch.tensor(np.real(spc), dtype=torch.float32),
        torch.tensor(np.imag(spc), dtype=torch.float32)], dim=0).to(device)
   else:
       spct = spc

   L0 = 20.515e3; dx0 = L0/sz;   #set up initial 1D coordinate  - to match the 33x33 spline model in VLF, should be 20.60638e3 microns
   x0 = torch.linspace( - L0/2 + dx0/2, L0/2 - dx0/2, sz).to(device)  # units are microns
   x0np =  np.linspace( - L0/2 , L0/2 , 3*sz)
   #set up initial screen
   if screentype == 'pixels':
      pts = torch.stack([spct[0].reshape((sz,sz)),
                         spct[1].reshape((sz, sz)) ], dim=0)
   elif screentype == 'spline':
      x0np, y0np = np.meshgrid(x0np,x0np,indexing='ij')
      filename = "SplineInterpMatrix"+str(sz)+"x"+str(sz)+"_L"+str(L0)+".npy"
      if isfile(filename):
         spmat = np.load(filename)
      else:
         #the numpy operation below can cause an OpenMP conflict on some installations
         Spl = BS.BivariateCubicSpline(x0np.flatten(),y0np.flatten(),sz)
         spmat = Spl.mat
         np.save(filename,spmat)
      spmat = torch.tensor(spmat).to(device)  # Spl.mat (the interpolation matrix) is purely real-valued
      pts = torch.stack([
         (spmat@spct[0]).reshape((len(x0np),len(x0np))),
         (spmat@spct[1]).reshape((len(x0np),len(x0np))) ])
      del spmat; torch.cuda.empty_cache()  # free this memory!
      x0 = torch.tensor(x0np).to(device)
   # resample pts to high res
   sz1 = 512
   dx01 = L0/sz1; x01 = torch.linspace( - L0/2 + dx01/2, L0/2 - dx01/2, sz1).to(device)
   g = F.ResampleField2D(pts,x0,x01)
   #create the soft edge on the plane wave source in VLF
   g = F.ApplyStopAndOrAperture(g, x01 ,-1. , d_ap=L0, shape='square', smoothing_distance=1210.)
   #now propagate to the occulter plane
   L1 = 2250.; dx1 = 8; x1 = torch.linspace(-L1/2 + dx1/2, L1/2 - dx1/2 , int(L1//dx1)).to(device)
   g = F.FFT_prop(g,x01,x1, 0.8e6, dist_in_front=8.5e4)
   #apply stop and aperture in occulter plane
   g = F.ApplyStopAndOrAperture(g, x1 , 142., d_ap=2250., shape='square', smoothing_distance=71.)

   # Take an optical FT to arrive at the Lyot stop
   L2 = 8.192e3; dx2 = 32.; x2 =  torch.linspace(-L2/2 + dx2/2, L2/2 - dx2/2 , int(L2//dx2)).to(device)
   g = F.FFT_prop(g,x1,x2, 0.4e6, None)
   #apply Lyot Stop
   g = F.ApplyStopAndOrAperture(g,x2,-1., d_ap=7.9e3   , shape='square', smoothing_distance=150.)
   #the numpy operation below can cause an OpenMP conflict on some installations
   #F.MakeAmplitudeImage(g,x2,'linear');plt.title("just after Lyot stop");
   #gnp_Lyot = F.torch2numpy_complex(g)

   # Take an optical FT to arrive at the detector
   #    Evaluating the FT in non-centered tile of the image plane would require some more work
   NN3 = 256
   L3 = 2200.; dx3 = L3/NN3; x3 = torch.linspace(-L3/2 + dx3/2, L3/2 - dx3/2, NN3)
   g = F.FFT_prop(g,x2,x3,4.e5, dist_in_front=1.e5)
   #the numpy operation below can cause an OpenMP conflict on some installations
   #F.MakeAmplitudeImage(g,x3,'linear'); plt.title('detector plane');
   #gnp_det = F.torch2numpy_complex(g)

   #scale g to Virtual Lab Fusion amplitudes for the same optical system
   g *= 1.442e-9  # this gives the output for flat input the same total energy as the realistic image
   if slice_indices is None:
       return g
   else:
       si = slice_indices
       return g[:,si[0]:si[1], si[2]:si[3]]
#END OAP_Model function

##%%   this could be too big.  See the BuildJacobianViaTiles function below
#mmodel = lambda spco: OAP_Model(spco, slice_indices=[110,130,60,80])
#jac = torch.autograd.functional.jacobian(mmodel, spco)

#%%
def BuildJacobianLinear():
    g = OAP_Model(spco,None)
    #jacobian tensor
    jac = np.zeros((g.shape[1]*g.shape[2],sz*sz)).astype('complex')
    b = torch.zeros((sz*sz,),device=device)
    time_start = time.time()
    for k in range(sz*sz):
        a = torch.zeros((sz*sz,),device=device)
        a[k] = 1.0
        c = torch.stack([a,b], dim=0).to(device)
        if not  c.shape == spco.shape:
            raise Exception(f"The input tensor shape of c, {c.shape}, is wrong.  It should be {spco.shape}.")
        g = OAP_Model(c,None)
        jac_k = g[0,:,:].cpu().numpy() + 1j*g[1,:,:].cpu().numpy()  # the Jacobian is simply the unit response
        jac[:,k] = jac_k.reshape((g.shape[1]*g.shape[2],))

        if np.mod(k,10) == 0:
            print(f"Finished step {k} of {sz*sz}.  Total time elapsed is {(time.time()-time_start)/60} minutes.")
    return jac


#%%
#This routine pieces together the Jacobian, one tile in the output plane at a time.
#It uses the command jac = torch.autograd.functional.jacobian(OAP_Model, spco)
# nT - The number of 1D spatial segments for the tiling.  The number of tiles will be nT*nT
# spco - input tensor specifying the point at which the gradient will be evaluated.
#    if the system is linear in spco, the value of its elements should not matter
def _BuildJacobianViaTiles(nT, spc=spco):
    raise Exception("For linear optical systems it is better to use BuildJacobianLinear")
    #first get the size of the output field and time it
    tt = time.time()
    g = OAP_Model(spc, slice_indices=None, screentype=screentype)
    print("Time to run model without gradient:", (time.time() - tt), "seconds")
    print(f"field tensor shape: {g.shape}.  The spatial dimension is {g[0].shape}.")
    if g.shape[1] != g.shape[2]:
        raise Exception("The spatial dimensions must be square.")
    npix = g.shape[1]
    rem = np.remainder(npix,nT)
    if rem != 0:
        raise Exception(f"The spatial dimension {g.shape[1]} needs to be divisible by input parameter nT {nT}.")
    nnp = int(npix/nT)
    sd = [0, 0, 0, 0]  # slice indices

    #build up the jacobian.
    jacReRe = torch.zeros((g.shape[1],g.shape[2],spco.shape[1])).to('cpu') # Jacobian of the real of g w.r.t the real part of spco
    jacReIm = torch.zeros((g.shape[1],g.shape[2],spco.shape[1])).to('cpu') # Jacobian of the real of g w.r.t the imag part of spco
    jacImRe = torch.zeros((g.shape[1],g.shape[2],spco.shape[1])).to('cpu') # Jacobian of the imag of g w.r.t the real part of spco
    jacImIm = torch.zeros((g.shape[1],g.shape[2],spco.shape[1])).to('cpu') # Jacobian of the imag of g w.r.t the imag part of spco

    del g;  torch.cuda.empty_cache()

    #in this loop, liljac is the jacobian for one tile.  It is 5D:
    #   dim 0 = real [0] or image [1]  part of gradient
    #   dim 1 = y spatial dimension
    #   dim 2 = x spatial dimension
    #   dim 3 = real [0] or imag [1] part of spco
    #   dim 4 = 1D spatial index of spco (spco is spatially flattened)
    for kx in range(nT):
        tt = time.time()
        sd[2] = nnp*kx ; sd[3] = sd[2] + nnp
        for ky in range(nT):
          sd[0] = nnp*ky ; sd[1] = sd[0] + nnp
          mmodel = lambda spco: OAP_Model(spco, slice_indices=sd)

          liljac = torch.autograd.functional.jacobian(mmodel, spco)


          #The Cauchy-Riemann conditions imply jacReRe = JacImIm and jacReIm = - jacImRe
          #   so jac_complex = jacReRe + j*JacImRe  consider the linear function u = a*v (all three are complex nums) and then du_r/dv_r, etc.
          jacReRe[sd[0]:sd[1], sd[2]:sd[3]] =  liljac[0,:,:,0,:]
          jacReIm[sd[0]:sd[1], sd[2]:sd[3]] =  liljac[0,:,:,1,:]
          jacImRe[sd[0]:sd[1], sd[2]:sd[3]] =  liljac[1,:,:,0,:]
          jacImIm[sd[0]:sd[1], sd[2]:sd[3]] =  liljac[1,:,:,1,:]

          if kx == 0 and ky == 3: print(f"time per tile is {(time.time()-tt)/3} seconds")
          if ky == nT-1: print(f"column {kx} time = {time.time()-tt} s. There are {nT} columns.")
    jacReRe_np = jacReRe.numpy()
    jacReIm_np = jacReIm.numpy()
    jacImIm_np = jacImIm.numpy()
    jacImRe_np = jacImRe.numpy()

    return  ( jacReRe_np, jacReIm_np, jacImRe_np, jacImIm_np  )


# Before closing the kernel to avoid the OpenMP confitct save the output.
if False:
   jacReRe, jacReIm, jacImRe, jacImIm   =  _BuildJacobianViaTiles(16, spco=spco)
   #The neede Jacobian is given by:
   jac = jacReRe + 1j*jacImRe  # there are several equivalent expressions.


##################  Do Not Import Torch! (due to OpenMP conflict)
from os import path as ospath
#Test the Chauchy-Riemann conditions: jacReRe = JacImIm and jacReIm = - jacImRe
if False:
    loc = "E:/MyOpticalSetups/EFC Papers/DataArrays"
    fn = ospath.join(loc, "Jacobian_TorchThreeOAPmodel256x256_64x64_ReRe.npy")
    assert ospath.isfile(fn)
    ReRe = np.load(fn)
    fn = ospath.join(loc, "Jacobian_TorchThreeOAPmodel256x256_64x64_ReIm.npy")
    assert ospath.isfile(fn)
    ReIm = np.load(fn)
    fn = ospath.join(loc, "Jacobian_TorchThreeOAPmodel256x256_64x64_ImRe.npy")
    assert ospath.isfile(fn)
    ImRe = np.load(fn)
    fn = ospath.join(loc, "Jacobian_TorchThreeOAPmodel256x256_64x64_ImIm.npy")
    assert ospath.isfile(fn)
    ImIm = np.load(fn)


    lReRe = lambda ky, kx : ReRe[ky,kx,:]
    lReIm = lambda ky, kx : ReIm[ky,kx,:]
    lImRe = lambda ky, kx : ImRe[ky,kx,:]
    lImIm = lambda ky, kx : ImIm[ky,kx,:]

    kx = 11; ky= 150;
    plt.figure(); plt.plot(lReRe(ky,kx),'ko',lImIm(ky,kx),'rx'); plt.title(f'ReRe,ImIm condition. ky={ky}, kx={kx}')
    plt.figure(); plt.plot(lReIm(ky,kx),'ko',-lImRe(ky,kx),'rx'); plt.title(f'ReIm,ImRe condition. ky={ky}, kx={kx}')
