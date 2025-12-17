#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 10:44:40 2025
@author: Richard Frazin
This makes the figures for the "Gratis" paper
"""
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import pickle
import EFC as EFCm


#%%  Make dark holes.  Store iteration info
lamD = 5.89 # "lambda/D" in pixel units
A = EFCm.EFC(EFCm.B21, EFCm.B33, EFCm.DomMat, EFCm.CroMat)

#%% make the the iteration figures
loadDHinfo = True
if not loadDHinfo :  # do the optimizations
   pl45 = A.MakePixList( np.round(128 + lamD*np.array([3,7,3,7]) ).astype('int') ) #pixel lists for dark holes
   plxa = A.MakePixList( np.round(128 + lamD*np.array([3,7,-2,2])).astype('int') )
   print(f"The dark holes have {len(pl45)} and {len(plxa)} pixels.")
   (out45, sols45, cost45) = A.DigDominantHole(np.zeros((441,)), pl45, DMconstr=np.pi/2)
   (outxa, solsxa, costxa) = A.DigDominantHole(np.zeros((441,)), plxa, DMconstr=np.pi/2)
else:
   with open("DHinfo12172025.pickle","rb") as fp: DHinfo = pickle.load(fp)
   cost45 = np.array(DHinfo['cost45'])/len(DHinfo['pl45']) # mean intensity in DH
   costxa = np.array(DHinfo['costxa'])/len(DHinfo['plxa'])
   ccro45 = []; ccroxa = [] # secondary mean intensities
   for sol in DHinfo['sols45']:
      ccro45.append(np.mean(A.DMcmd2Intensity(sol,'cross',pixlist=DHinfo['pl45'])))
   for sol in DHinfo['solsxa']:
      ccroxa.append(np.mean(A.DMcmd2Intensity(sol,'cross',pixlist=DHinfo['plxa'])))





#%%



#%% images of the polarized field on the entrance pupil
dir1 = "../../EFCSimData"
PupXx = np.load(join(dir1, 'BigBeamReduceCG_Pupil_InputX_Ex.npy'))
PupXy = np.load(join(dir1, 'BigBeamReduceCG_Pupil_InputX_Ey.npy'))
PupYx = np.load(join(dir1, 'BigBeamReduceCG_Pupil_InputY_Ex.npy'))
PupYy = np.load(join(dir1, 'BigBeamReduceCG_Pupil_InputY_Ey.npy'))
ext = [-10,10,-10,10] #pupil extension in mm

plt.figure();plt.imshow(np.abs(PupXx)-1, extent=ext, cmap='coolwarm',origin='lower');plt.colorbar();
plt.xticks(fontsize=8);plt.yticks(fontsize=8);plt.title(r"$|E_{xx}|-1$");plt.xlabel("x (mm)",fontsize=10);plt.ylabel("y (mm)");

plt.figure();plt.imshow(np.abs(PupYy)-1, extent=ext, cmap='coolwarm',origin='lower');plt.colorbar();
plt.xticks(fontsize=8);plt.yticks(fontsize=8);plt.title(r"$|E_{yy}|-1$");plt.xlabel("x (mm)",fontsize=10);plt.ylabel("y (mm)");

plt.figure();plt.imshow(np.abs(PupXx) - np.abs(PupYy), extent=ext, cmap='coolwarm',origin='lower');plt.colorbar();
plt.xticks(fontsize=8);plt.yticks(fontsize=8);plt.title(r"$|E_{yy}|- |E_{xx}|$");plt.xlabel("x (mm)",fontsize=10);plt.ylabel("y (mm)");

plt.figure();plt.imshow(np.abs(PupXy), extent=ext, cmap='coolwarm',origin='lower');plt.colorbar();
plt.xticks(fontsize=8);plt.yticks(fontsize=8);plt.title(r"$|E_{xy}|$");plt.xlabel("x (mm)",fontsize=10);plt.ylabel("y (mm)");

plt.figure();plt.imshow(np.abs(PupYx), extent=ext, cmap='coolwarm',origin='lower');plt.colorbar();
plt.xticks(fontsize=8);plt.yticks(fontsize=8);plt.title(r"$|E_{yx}|$");plt.xlabel("x (mm)",fontsize=10);plt.ylabel("y (mm)");
