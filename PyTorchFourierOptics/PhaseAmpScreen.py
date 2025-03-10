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
from os import path as ospath

#Jacobian file
fnjac = ospath.join("E:/MyOpticalSetups/EFC Papers/" , 
         "DataArrays/Jacobian_TorchThreeOAPmodel256x256_65x65_complex.npy")
#image file
fnimg = ospath.join("E:/MyOpticalSetups/EFC Papers/DataArrays", 
    "Three_NonPlanarOAPcoronagraphWithScreens_RealisticModel_PWinput90deg_DomField.npy")
jac = np.load(fnjac)
img = np.load(fnimg)

#multiply jac by the scaling factor for consistency (see 3/10/25 notes)\
jac *= 18.154
img = img.reshape((256**2,))
assert img.shape[0]  == jac.shape[0]

U, s, Vh = np.linalg.svd(jac,full_matrices=True,compute_uv=True)  # jac = U@s@Vh


