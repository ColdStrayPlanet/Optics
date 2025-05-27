#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 18:42:03 2025

@author: rfrazin
"""

import numpy as np
import matplotlib.pyplot as plt
from os import path as ospath  #needed for isfile(), join(), etc.
from sys import path as syspath
import pickle
import EFC

with open('Stuff05152025.pickle','rb') as fp: stuff = pickle.load(fp); del(fp)
A = EFC.EFC(EFC.B21, EFC.B33, EFC.DomMat, EFC.CroMat)
sols45  = stuff['interm45']; #intermediate solutions
solsXax = stuff['intermXax']
cost45  = stuff['cost45']  # objective fcn values
costAx  = stuff['costXax']
pl45 =  stuff['pixlist45d']  # list of dark hole pixels
plXax = stuff['pixlistXax']
IoffAx45 = stuff['IoffAx45']  #intensity of off-axis beam
IoffAxXax = stuff['IoffAxXax']
CostCross45 = stuff['CostCross45'] # sum of cross intensity in dark hole
CostCrossXax = stuff['CostCrossXax']
sol45 = sols45[-1] # final dark hole solution
solXax = solsXax[-1]

PhasorXax = A.LinearPhaseForOffAxisSource(5.2, 0. , output='Phasor')
Phasor45  = A.LinearPhaseForOffAxisSource(7.0, 45., output='Phasor')

#functions for modulated intensities
Cd45 = lambda cmd: np.sum(A.DMcmd2Intensity(sol45+cmd, 'dom', pixlist=pl45 ))
Cc45 = lambda cmd: np.sum(A.DMcmd2Intensity(sol45+cmd, 'cross', pixlist=pl45 ))
CdXax = lambda cmd: np.sum(A.DMcmd2Intensity(solXax+cmd, 'dom', pixlist=plXax ))
CcXax = lambda cmd: np.sum(A.DMcmd2Intensity(solXax+cmd, 'cross', pixlist=plXax ))
Ioa45  = lambda cmd: np.max(A.DMcmd2Intensity(sol45  + cmd, pmat='dom' ,pixlist=pl45, OffAxPhasor=Phasor45))
IoaXax = lambda cmd: np.max(A.DMcmd2Intensity(solXax + cmd ,pmat='dom',pixlist=plXax,OffAxPhasor=PhasorXax))

#%%
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
   cc = cmdcro();
   if k == 0:
      cd *= 0.
      cc *= 0.
   Cdommod45.append( Cd45( cd))
   CdommodXax.append(CdXax(cd))
   Ccromod45.append( Cc45( cc))
   CcromodXax.append(CcXax(cc))
   Imodoa45.append( Ioa45(cc))
   ImodoaXax.append(IoaXax(cc))


#%%
#Full Field Intensities
Idh45   = A.DMcmd2Intensity(sols45[-1],'dom').reshape((256,256));
Ioa45dh  = A.DMcmd2Intensity(sols45[-1],pmat='dom',OffAxPhasor=Phasor45).reshape((256,256));
IdhXax  = A.DMcmd2Intensity(solsXax[-1],'dom').reshape((256,256));
IoaXaxdh = A.DMcmd2Intensity(solsXax[-1],pmat='dom',OffAxPhasor=PhasorXax).reshape((256,256));


#%%
#CG iterations
CostCross45 = [];  CostCrossXax = []  # total cross intensities in dark hole
IoffAxXax = []; IoffAx45  = [] # peak off-axis intensities
for k in range(len(sols45)):
   CostCross45.append( np.sum(A.DMcmd2Intensity(sols45[k ], 'cross', pixlist=pl45 )))
   IoffAx45.append(np.max(A.DMcmd2Intensity(sols45[k], pmat='dom' ,pixlist=pl45, OffAxPhasor=Phasor45)))
for k in range(len(solsXax)):
   CostCrossXax.append(np.sum(A.DMcmd2Intensity(solsXax[k], 'cross', pixlist=plXax)))
   IoffAxXax.append(np.max(A.DMcmd2Intensity(solsXax[k],pmat='dom',pixlist=plXax,OffAxPhasor=PhasorXax)))
#%%  plots of intensity vs. CG iteration
plt.figure();plt.plot(np.log10(cost45)-2.35,'ko-', np.log10(CostCross45 )-2.35,'bo-');plt.title('45deg box');plt.xlabel('CG iteration');plt.ylabel('mean intensity (contrast units)');
plt.figure();plt.plot(np.log10(costAx)-2.35,'ko-', np.log10(CostCrossXax)-2.35,'bo-');plt.title('X-axis box');plt.xlabel('CG iteration');plt.ylabel('mean intensity (contrast units)');


#%%
# plot showing that the cross intensity is poorly modulated by commands that modulate the dominant intensity
if ospath.isfile("randomDMcommand_n441.npy"):
   cmdrnd = np.load("randomDMcommand_n441.npy")
else:
   cmdrnd = np.pi*(np.random.rand(441) - 0.5)
cmdampl = np.logspace(-4,-2,33)

Cd45rnd = []
Cc45rnd = []
for k in range(len(cmdampl)):
   Cd45rnd.append(Cd45(cmdrnd*cmdampl[k]))
   Cc45rnd.append(Cc45(cmdrnd*cmdampl[k]))
Cd45rnd = np.array(Cd45rnd); Cc45rnd = np.array(Cc45rnd)

#%%
cmdnorm = np.sqrt(np.sum(cmdrnd**2))
plt.figure();plt.plot((cmdampl*cmdnorm),np.log10(Cd45rnd/len(pl45)),'ko-')
plt.plot((cmdampl*cmdnorm),np.log10(Cc45rnd/len(pl45)),'r*-.')
plt.xlabel("2-norm of DM command"); plt.ylabel("Mean Hole Intensity (contrast units)")
