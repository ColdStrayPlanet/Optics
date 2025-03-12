# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 16:48:49 2025

@author: Richard Frazin

See notes from Feb/Mar 2025 in Google Docs

This uses the Jacobian from ThreeOAPmodel.py and the "measured" VLF intensity with 
the OAP phase errors in attempt to find a reasonable phase/amplitude screen to
account for OAP phase errors within an otherwise ideal model.



"""

import numpy as np
import os
import pickle
import time

#spline interpolation matrix file
fnspl = os.path.join("E:\Python\Optics\PyTorchFourierOptics", "SplineInterpMatrix65x65_L20515.0.npy")
if not os.path.isfile(fnspl):
    print("Can't find spline interpolation matrix file", fnspl)
else:
    splm = np.load(fnspl)

#Jacobian file
fnjac = os.path.join("E:/MyOpticalSetups/EFC Papers/" , 
         "DataArrays/Jacobian_TorchThreeOAPmodel256x256_65x65_complex.npy")
#   its SVD
fnjacSVD = os.path.join("E:/MyOpticalSetups/EFC Papers/",
                     "DataArrays/JacobianSVD_TorchThreeOAPmodel256x256_65x65_complex.pickle")
#image file
fnimg = os.path.join("E:/MyOpticalSetups/EFC Papers/DataArrays", 
    "Three_NonPlanarOAPcoronagraphWithScreens_RealisticModel_PWinput90deg_DomField.npy")
jac = np.load(fnjac)
img = np.load(fnimg)
if jac.shape[1] != splm.shape[1]:
    raise Exception("Jacobian and spline interpolation matrix are inconsistant.")


#multiply jac by the scaling factor for consistency (see 3/10/25 notes)\
jac *= 18.154
img = img.reshape((256**2,))
assert img.shape[0]  == jac.shape[0]

if not os.path.isfile(fnjacSVD):
   print("Can't find the file with the SVD of the Jacobian:", fnjacSVD, "This will take time (< 1hr on Ozzy) and memory 35 GB.")
   U, s, V = np.linalg.svd(jac,full_matrices=True,compute_uv=True)  # jac = U@s@Vh
   V = np.conj(V.T)   # this is because the 3rd item return by the SVD algo is Hermitian conj of V
else:
    print("Loading SVD.  Could take a couple of minutes.")
    with open(fnjacSVD,'rb') as filep: SVD = pickle.load(filep) 
    U = SVD['U']; s = SVD['s']; V = SVD['V']; note = SVD['note']

#now compare the truncated pseudo-inverse solutions 
nnz = np.count_nonzero(s)
errm =  np.zeros((nnz,)) # error metric array
norm1 = np.zeros((nnz,)) # solution norm array
norm2 = np.zeros((nnz,))
pj = np.zeros((jac.shape[1],jac.shape[0])).astype('complex')  # pseudoinverse 

timestart = time.time()
for k in range(nnz):
    # if p and q are vectors of length M and N, resp., np.outer(p,q) has shape (M,N)
    pj +=   np.outer(V[:,k],np.conj(U[:,k]))/s[k]
    sol = pj@img  # pinv solution (spline coefficients)
    imsol = splm@sol  # spline interpolation onto flattened amplitude/phase screen
    norm1[k] = (np.abs(imsol) - 1.).max()
    norm2[k] = np.median(np.abs(imsol) - 1.)
    errm[k] = np.mean(np.abs(jac@sol - img)) 
    print(f"k = {k} of {nnz} done. Time = {(time.time()-timestart)/3600} hours.")
