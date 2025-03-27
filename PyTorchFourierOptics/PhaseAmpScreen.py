# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 16:48:49 2025

@author: Richard Frazin

See notes from Feb/Mar 2025 in Google Docs

This uses the Jacobian from ThreeOAPmodel.py and the "measured" VLF intensity with 
the OAP phase errors in attempt to find a reasonable phase/amplitude screen to
account for OAP phase errors within an otherwise ideal model.

Regularized Psuedo-inverse notes (see Eq. 29 in Frazin & Rodack JOSAA, 38, 10, 1557, 2021):
    We seek the regularized pinv of the M-by-N (M > N) matrix A with:
        A  - M-by-N matrix we seek to invert 
        R  - N-by-N regularization matrix (positive semi-def)
        y  - M vector of observations (measurements, data, whatever)
        x  - N vector representing the regressand, i.e., the quantity to be estimated
        x_0 - N vector for non-centered regularization
        b -  positive regularization parameter
        
 \hat{x} = [A^T A + b R ]^{-1} \times [ A^T y + b R x0  ] 


"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time


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
    pl = MakePixList([64,192,64,192],(256,256))


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
img = np.load(fnimg)
jac = np.load(fnjac)
if jac.shape[1] != splm.shape[1]:
    raise Exception("Jacobian and spline interpolation matrix are inconsistant.")

if LittleJac:  # cut Jac (and img) down to size
    img = img.reshape((256**2,))[pl]
    jac = jac[pl,:]

#%%

#multiply jac by the scaling factor for consistency (see 3/10/25 notes)\
jac *= 18.154
assert img.shape[0]  == jac.shape[0]

if not LittleJac:  # Get (or do) the SVD of the full Jacobian
    if not os.path.isfile(fnjacSVD):
       print("Can't find the file with the SVD of the Jacobian:", fnjacSVD, "This will take time (< 1hr on Ozzy) and memory 35 GB.")
       U, s, V = np.linalg.svd(jac,full_matrices=True,compute_uv=True)  # jac = U@s@Vh
       V = np.conj(V.T)   # this is because the 3rd item returned by the SVD algo is Hermitian conj of V
    else:
        print("Loading SVD.  Could take a couple of minutes.")
        with open(fnjacSVD,'rb') as filep: SVD = pickle.load(filep) 
        U = SVD['U']; s = SVD['s']; V = SVD['V']; note = SVD['note']
else: # reduce the Jacobian and then do (or get) its SVD
    U, s, V = np.linalg.svd(jac, True,True)
    V = np.conj(V.T)

#%%  i 
if True: # regularized pinv, see notes in the doc string.  R is taken to be the identity matrix
    sscale =s.max() # largest s.v.
    JhJ = np.conj(jac.T)@jac
    Jhy = np.conj(jac.T)@img
    rp = np.logspace(-1,-7,13) # regularization parameters
    solnom = np.ones((JhJ.shape[0],)).astype('complex')  # point for non-centered regularization
    
    erry = np.zeros((len(rp),))  # data misfit error
    norm = np.zeros((len(rp),))  # norm of regularization value 
    for k in range(len(rp)):
        sol = np.linalg.inv( JhJ + (rp[k]*sscale**2)*np.eye(JhJ.shape[0])  )@( Jhy +  rp[k]*sscale**2*np.ones((JhJ.shape[0])) )
        erry[k] = np.mean(np.abs(jac@sol - img)**2)
        norm[k] = np.sqrt(np.mean(np.abs(sol - solnom)**2))

    plt.figure();plt.plot(np.log10(rp), erry,'ko-'); plt.title('data misfit error');
    plt.figure();plt.plot(np.log10(rp), norm,'ko-'); plt.title('RMS deviation from unity')

#%% 
    k = 1;  # try this reg param
    sol = np.linalg.inv( JhJ + (rp[k]*sscale**2)*np.eye(JhJ.shape[0])  )@( Jhy +  rp[k]*sscale**2*np.ones((JhJ.shape[0])) )
    imsol = jac@sol
    splim = splm@sol  # this is the phase/amplitude screen


#%%
if False:  #now compare the truncated pseudo-inverse solutions 
    nnz = np.count_nonzero(s)
    errm1 =  np.zeros((nnz,)) # error metric array
    errm2 =  np.zeros((nnz,))
    norm1 = np.zeros((nnz,)) # solution norm array
    norm2 = np.zeros((nnz,))
    pj = np.zeros((jac.shape[1],jac.shape[0])).astype('complex')  # pseudoinverse 
    
    favorite = None
    timestart = time.time()
    for k in range(nnz):
        # if p and q are vectors of length M and N, resp., np.outer(p,q) has shape (M,N)
        pj +=   np.outer(V[:,k],np.conj(U[:,k]))/s[k]
        if k == favorite:
            pjfav = pj.copy()
            print(f"favorite index {k}")
            if id(pj) == id(pjfav):
                raise Exception("There is no hope for this world.")
        sol = pj@img  # pinv solution (spline coefficients)
        imsol = splm@sol  # spline interpolation onto flattened amplitude/phase screen
        norm1[k] = (np.abs(imsol) - 1.).max()
        norm2[k] = np.median(np.abs(imsol) - 1.)
        errm1[k] = np.mean(np.abs(jac@sol - img)  ) 
        errm2[k] = np.mean(np.abs(jac@sol - img)**2)
        if k == favorite:  break
        if np.remainder(k,50) == 0: 
           print(f"k = {k} of {nnz} done. Time = {(time.time()-timestart)/60} minutes.")
           
    favorite = 1123; fa = favorite 
    pjfav = (V[:,:fa]*(1./s[:fa]))@np.conj(U[:,:fa].T)
    sol = pjfav@img;  imsol = splm@sol ; hatimg = jac@sol



