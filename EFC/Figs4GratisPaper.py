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
with open("DHinfo12172025.pickle","rb") as fp: DHinfo = pickle.load(fp)
loadDHinfo = True
if not loadDHinfo :  # do the optimizations
   pl45 = A.MakePixList( np.round(128 + lamD*np.array([3,7,3,7]) ).astype('int') ) #pixel lists for dark holes
   plxa = A.MakePixList( np.round(128 + lamD*np.array([3,7,-2,2])).astype('int') )
   print(f"The dark holes have {len(pl45)} and {len(plxa)} pixels.")
   (out45, sols45, cost45) = A.DigDominantHole(np.zeros((441,)), pl45, DMconstr=np.pi/2)
   (outxa, solsxa, costxa) = A.DigDominantHole(np.zeros((441,)), plxa, DMconstr=np.pi/2)
   cost45 = np.array(DHinfo['cost45'])/len(DHinfo['pl45']) # mean intensity in DH
   costxa = np.array(DHinfo['costxa'])/len(DHinfo['plxa'])
   ccro45 = []; ccroxa = [] # secondary mean intensities
   for sol in DHinfo['sols45']:
      ccro45.append(np.mean(A.DMcmd2Intensity(sol,'cross',pixlist=DHinfo['pl45'])))
   for sol in DHinfo['solsxa']:
      ccroxa.append(np.mean(A.DMcmd2Intensity(sol,'cross',pixlist=DHinfo['plxa'])))

#%%  SVD cross spectrum figures
ADr = A.SysD[DHinfo['pl45'],:]
ACr = A.SysC[DHinfo['pl45'],:]
Ud, sd, Vd = np.linalg.svd(ADr, full_matrices=True); Vd = np.conj(Vd.T);
Uc, sc, Vc = np.linalg.svd(ACr, full_matrices=True); Vc = np.conj(Vc.T);

sdmax = np.max(sd)
scc = []
for k in range(len(sd)):
   scc.append(np.linalg.norm(ADr@Vc[:,k]))
scc = np.array(scc)

plt.figure(figsize=(6,4))
plt.plot(sd[:200] / sdmax, 'ko-', linewidth=2, label='Primary Spectrum')
plt.plot(scc[:200] / sdmax, 'rx-', linewidth=1.5, label='Cross spectrum')
plt.xlabel('Mode index k')
plt.ylabel('Normalized magnitude')
#plt.title('')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

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

#%%
Phasorax = A.LinearPhaseForOffAxisSource(6.1, 0. , output='Phasor')
Phasor45  = A.LinearPhaseForOffAxisSource(7.2, 45., output='Phasor')
#functions for modulated intensities
Cd45 = lambda cmd: np.mean(A.DMcmd2Intensity(DHinfo['sols45'][-1] +cmd, 'dom', pixlist=DHinfo['pl45'] ))
Cc45 = lambda cmd: np.mean(A.DMcmd2Intensity(DHinfo['sols45'][-1] +cmd, 'cross', pixlist=DHinfo['pl45'] ))
Cdax = lambda cmd: np.mean(A.DMcmd2Intensity(DHinfo['solsxa'][-1] + cmd, 'dom', pixlist=DHinfo['plax'] ))
Ccax = lambda cmd: np.mean(A.DMcmd2Intensity(DHinfo['solsxa'][-1] +cmd, 'cross', pixlist=DHinfo['plax'] ))
Ioa45  = lambda cmd: np.max(A.DMcmd2Intensity(DHinfo['sols45'][-1] + cmd, pmat='dom' ,pixlist=DHinfo['pl45'], OffAxPhasor=Phasor45))
Ioaax = lambda cmd: np.max(A.DMcmd2Intensity(DHinfo['solsxa'][-1] + cmd ,pmat='dom',pixlist=DHinfo['plax'], OffAxPhasor=Phasorax))
cmddom = lambda  :  0.002*np.pi*(np.random.rand(441) - 0.5)
cmdcro = lambda  :   0.02*np.pi*(np.random.rand(441) - 0.5)

I0    = A.DMcmd2Intensity(0*DHinfo['sols45'][-1],'dom'  ).reshape((256,256));
I0c   = A.DMcmd2Intensity(0*DHinfo['sols45'][-1],'cross').reshape((256,256));

Idh45    = A.DMcmd2Intensity(DHinfo['sols45'][-1],'dom'  ).reshape((256,256));
Idh45c   = A.DMcmd2Intensity(DHinfo['sols45'][-1],'cross').reshape((256,256));
Ioa45dh  = A.DMcmd2Intensity(DHinfo['sols45'][-1],pmat='dom',OffAxPhasor=Phasor45).reshape((256,256));
IdhXax   = A.DMcmd2Intensity(DHinfo['solsxa'][-1],'dom'  ).reshape((256,256));
IdhXaxc  = A.DMcmd2Intensity(DHinfo['solsxa'][-1],'cross').reshape((256,256));
IoaXaxdh = A.DMcmd2Intensity(DHinfo['solsxa'][-1],pmat='dom',OffAxPhasor=Phasorax).reshape((256,256));

#lambda functions for random modulation
cmddom = lambda  :  0.002*np.pi*(np.random.rand(len(sol45)) - 0.5)
cmdcro = lambda  :   0.02*np.pi*(np.random.rand(len(sol45)) - 0.5)
Nmodulations = 50
Ccromod45  = []
Cdommod45  = []
CcromodXax = []
CdommodXax = []
Imodoa45   = []
ImodoaXax  = []
for k in range(Nmodulations):
   cd = cmddom();
   if k == 0:
      cd *= 0.
   CdommodXax.append(CdXax(cd))
   CcromodXax.append(CcXax(cd))
   ImodoaXax.append(IoaXax(cd))
CdommodXax = np.array(CdommodXax)
CcromodXax = np.array(CcromodXax)
ImodoaXax  = np.array(ImodoaXax)
print([CdommodXax[0], CcromodXax[0], ImodoaXax[0] ])
#%%
plt.figure();
plt.plot(CdommodXax,'ko', label='Primary');
plt.plot(CcromodXax*1.e4,'rx', label='1.e4*Secondary');
plt.plot(ImodoaXax*5.e-9, 'g*',label='5.e-9 Planet' );
plt.legend(loc='upper left'); plt.title('Random Modulation about DH Command');
plt.xlabel('modulation trial');plt.ylabel('Intensity (contrast)');


#%%
#plt.figure(); plt.imshow(np.log10(I0[100:180,100:180] + 1.e-6) ,origin='lower',cmap='coolwarm');plt.colorbar();
plt.figure(); plt.imshow(np.log10(IdhXax[100:180,100:180] + 1.e-10) ,origin='lower',cmap='coolwarm');plt.colorbar();
plt.figure(); plt.imshow(np.log10(Idh45[100:180,100:180] + 1.e-10) ,origin='lower',cmap='coolwarm');plt.colorbar();

#plt.figure(); plt.imshow(np.log10(I0c[100:180,100:180] + 1.e-12) ,origin='lower',cmap='coolwarm');plt.colorbar();
plt.figure(); plt.imshow(np.log10(IdhXaxc[100:180,100:180] + 1.e-14) ,origin='lower',cmap='coolwarm');plt.colorbar();
#plt.figure(); plt.imshow(np.log10(Idh45c[100:180,100:180] + 1.e-14) ,origin='lower',cmap='coolwarm');plt.colorbar();

plt.figure(); plt.imshow(np.log10(Ioa45dh[100:180,100:180] + 1.e-4),origin='lower',cmap='coolwarm'); plt.colorbar();
plt.title('off-axis source, dark hole DM command')
plt.figure(); plt.imshow(np.log10(IoaXaxdh[100:180,100:180] + 1.e-4),origin='lower',cmap='coolwarm'); plt.colorbar();
plt.title('off-axis source, dark hole DM command')
