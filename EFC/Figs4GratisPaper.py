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
trunc = lambda x: float(np.format_float_scientific(x, precision=5))

#%% load or calculate the dark hole solutions
lamD = 5.89 # "lambda/D" in pixel units
PropMatLoc = "/home/rfrazin/Py/EFCSimData/"
Domfn =  'SysMatNorm_Xx_BigBeam2Cg256x256_Lam0.9_33x33.npy'
Crofn =  'SysMatNorm_Xx_BigBeam2Cg256x256_Lam0.9_33x33.npy'
Dom2fn = 'SysMatNormPiShift_Yy_BigBeam2Cg256x256_Lam0.9_33x33.npy'
Cro2fn = 'SysMatNormPiShift_Yx_BigBeam2Cg256x256_Lam0.9_33x33.npy'
DomMat = np.load(join(PropMatLoc,Domfn));
CroMat = np.load(join(PropMatLoc,Crofn));
DomMat2 = np.load(join(PropMatLoc,Dom2fn));
CroMat2 = np.load(join(PropMatLoc,Cro2fn));

A = EFCm.EFC(EFCm.B21, EFCm.B33, DomMat, CroMat, Dom2PropMat=DomMat2, Cross2PropMat=CroMat2)
#%%
loadDHinfo = False
if loadDHinfo:
   DHinfofilename = "DHinfo02062026.pickle"
   with open(DHinfofilename, "rb") as fp: DHinfo = pickle.load(fp)
else:  # dont load DHinfo -- do the optimizations
   pl45 = A.MakePixList( np.round(128 + lamD*np.array([4,7,4,7]) ).astype('int') ) #pixel lists for dark holes
   plxa = A.MakePixList( np.round(128 + lamD*np.array([4,7,-1.5,1.5])).astype('int') )
   print(f"The dark holes have {len(pl45)} and {len(plxa)} pixels.")
   (out45, sols45, cost45) = A.DigDominantHole(np.zeros(441,), pl45, TwoDoms=True, DMconstr=np.pi/2)
   (outxa, solsxa, costxa) = A.DigDominantHole(np.zeros(441,), plxa, TwoDoms=True, DMconstr=np.pi/2)

#%%  evaluate all of the mean intensities in the dark holes
   DHinfo['cdom45']  = []; DHinfo['cdomxa']  = [] # primary mean intensities  XX
   DHinfo['ccro45']  = []; DHinfo['ccroxa']  = [] # secondary mean intensities XY
   DHinfo['cdom452'] = []; DHinfo['cdomxa2'] = [] # primary mean intensities YY
   DHinfo['ccro452'] = []; DHinfo['ccroxa2'] = [] # secondary mean intensities YX


   for sol in DHinfo['sols45']:
     DHinfo['ccro45'].append(np.mean(A.DMcmd2Intensity(sol,'cross',pixlist=DHinfo['pl45'])))
     DHinfo['cdom45'].append(np.mean(A.DMcmd2Intensity(sol,'dom',pixlist=DHinfo['pl45'])))
     DHinfo['ccro452'].append(np.mean(A.DMcmd2Intensity(sol,'cross2',pixlist=DHinfo['pl45'])))
     DHinfo['cdom452'].append(np.mean(A.DMcmd2Intensity(sol,'dom2',pixlist=DHinfo['pl45'])))
   for sol in DHinfo['solsxa']:
     DHinfo['cdomxa'].append(np.mean(A.DMcmd2Intensity(sol,'dom',pixlist=DHinfo['plxa'])))
     DHinfo['ccroxa'].append(np.mean(A.DMcmd2Intensity(sol,'cross',pixlist=DHinfo['plxa'])))
     DHinfo['cdomxa2'].append(np.mean(A.DMcmd2Intensity(sol,'dom2',pixlist=DHinfo['plxa'])))
     DHinfo['ccroxa2'].append(np.mean(A.DMcmd2Intensity(sol,'cross2',pixlist=DHinfo['plxa'])))


#%% show spline columumn 976 ~ (29,19) of the matrices.  Filenames such as Kn29dand19d_Exx.jpg
kn = 976
fnyx = join("../..", "EFCSimData/SysMatNorm_Yx_BigBeam2Cg256x256_Lam0.9_33x33.npy")
fnyy = join("../..", "EFCSimData/SysMatNorm_Yy_BigBeam2Cg256x256_Lam0.9_33x33.npy")
Bx = np.load(fnyx, mmap_mode='r');  By = np.load(fnyy, mmap_mode='r');
Exxkn = A.SysD[:,kn].reshape((256,256))
Exykn = A.SysC[:,kn].reshape((256,256))
Eyxkn = Bx[:,kn].reshape((256,256))
Eyykn = By[:,kn].reshape((256,256))

norm = np.abs(Exxkn).max()
bD = r'\mathbf{D}' # raw string
ext = [-1.,1.,-1.,1.] #image plane in mm
plt.figure(); plt.imshow(np.abs(Exxkn)/norm,cmap='coolwarm',extent=ext); plt.colorbar();
plt.xlabel('mm');plt.ylabel('mm');plt.title(rf'$|{bD}_{{xx}}[:,976]|$'); #raw f-string
plt.figure(); plt.imshow(np.abs(Exykn)/norm,cmap='coolwarm',extent=ext); plt.colorbar();
plt.xlabel('mm');plt.ylabel('mm');plt.title(rf'$|{bD}_{{xy}}[:,976]|$'); #raw f-string
plt.figure(); plt.imshow(np.abs(Eyxkn)/norm,cmap='coolwarm',extent=ext); plt.colorbar();
plt.xlabel('mm');plt.ylabel('mm');plt.title(rf'$|{bD}_{{yx}}[:,976]|$'); #raw f-string
plt.figure(); plt.imshow(np.abs(Eyykn)/norm,cmap='coolwarm',extent=ext); plt.colorbar();
plt.xlabel('mm');plt.ylabel('mm');plt.title(rf'$|{bD}_{{yy}}[:,976]|$'); #raw f-string



#%%  SVD cross spectrum figures
ADr = A.SysD[DHinfo['pl45'],:]
ACr = A.SysC[DHinfo['pl45'],:]
Ud, sd, Vd = np.linalg.svd(ADr, full_matrices=True); Vd = np.conj(Vd.T);
Uc, sc, Vc = np.linalg.svd(ACr, full_matrices=True); Vc = np.conj(Vc.T);

sdmax = np.max(sd); scmax = np.max(sc)
sdc = []; scd = [];
for k in range(len(sd)):
   sdc.append(np.linalg.norm(ADr@Vc[:,k]))
   scd.append(np.linalg.norm(ACr@Vd[:,k]))
sdc = np.array(sdc); scd = np.array(scd);

plt.figure(figsize=(6,4))
plt.plot(sd[:200] / sdmax, 'ko-', linewidth=2, label='Primary Spectrum')
plt.plot(sdc[:200] / sdmax, 'rx-', linewidth=1.5, label='1-2 Cross Spectrum')
plt.xlabel('Mode index k')
plt.ylabel('Normalized magnitude')
#plt.title('')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(6,4))
plt.plot(sc[:200] / scmax, 'ko-', linewidth=2, label='Secondary Spectrum')
plt.plot(scd[:200] / scmax, 'rx-', linewidth=1.5, label='2-1 Cross Spectrum')
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

#%%  full field intensities
Phasorax = A.LinearPhaseForOffAxisSource(6.1, 0. , output='Phasor')
Phasor45  = A.LinearPhaseForOffAxisSource(7.2, 45., output='Phasor')

I0    = A.DMcmd2Intensity(0*DHinfo['sols45'][-1],'dom'  ).reshape((256,256));
I0c   = A.DMcmd2Intensity(0*DHinfo['sols45'][-1],'cross').reshape((256,256));
Idh45    = A.DMcmd2Intensity(DHinfo['sols45'][-1],'dom'  ).reshape((256,256));
Idh45c   = A.DMcmd2Intensity(DHinfo['sols45'][-1],'cross').reshape((256,256));
Ioa45dh  = A.DMcmd2Intensity(DHinfo['sols45'][-1],pmat='dom',OffAxPhasor=Phasor45).reshape((256,256));
IdhXax   = A.DMcmd2Intensity(DHinfo['solsxa'][-1],'dom'  ).reshape((256,256));
IdhXaxc  = A.DMcmd2Intensity(DHinfo['solsxa'][-1],'cross').reshape((256,256));
IoaXaxdh = A.DMcmd2Intensity(DHinfo['solsxa'][-1],pmat='dom',OffAxPhasor=Phasorax).reshape((256,256));

#%%functions for modulated intensities
# soldex = -1 for the DH command itself. soldex is set to previous values for earlier iterations
# cmd is an additive command, used for modulations about a given iteration
Phasorax = A.LinearPhaseForOffAxisSource(6.1, 0. , output='Phasor')
Phasor45  = A.LinearPhaseForOffAxisSource(7.2, 45., output='Phasor')
Cd45 = lambda soldex, cmd  : np.mean(A.DMcmd2Intensity(DHinfo['sols45'][soldex] + cmd, 'dom', pixlist=DHinfo['pl45'] ))
Cc45 = lambda soldex, cmd  : np.mean(A.DMcmd2Intensity(DHinfo['sols45'][soldex] + cmd, 'cross', pixlist=DHinfo['pl45'] ))
Cdax = lambda soldex, cmd  : np.mean(A.DMcmd2Intensity(DHinfo['solsxa'][soldex] + cmd, 'dom', pixlist=DHinfo['plxa'] ))
Ccax = lambda soldex, cmd  : np.mean(A.DMcmd2Intensity(DHinfo['solsxa'][soldex] + cmd, 'cross', pixlist=DHinfo['plxa'] ))
Ioa45  = lambda soldex, cmd: np.max(A.DMcmd2Intensity(DHinfo['sols45' ][soldex] + cmd, pmat='dom' ,pixlist=DHinfo['pl45'], OffAxPhasor=Phasor45))
Ioaax = lambda soldex, cmd : np.max(A.DMcmd2Intensity(DHinfo['solsxa' ][soldex] + cmd ,pmat='dom',pixlist=DHinfo['plxz'], OffAxPhasor=Phasorax))

#%%

#make iteration figures
lpl45 = len(DHinfo['pl45']);   lplxa = len(DHinfo['plxa'])
Cdom45  = DHinfo['cdom45'] ; Ccro45  = DHinfo['ccro45']
Cdom45b = DHinfo['cdom452'] ; Ccro45b  = DHinfo['ccro452']
Cdomxa  = DHinfo['cdomxa'] ; Ccroxa   = DHinfo['ccroxa']
Cdomxab = DHinfo['cdomxa2'] ; Ccroxab  = DHinfo['ccroxa2']

plt.figure(); plt.title("Dark Hole Intensity Vs. Iteration");
plt.plot(np.log10(Cdom45),'k-',linewidth=3, label="Primary (XX)");
plt.plot(np.log10(Cdom45b),'ko',linewidth=3, label="Primary (YY)");
plt.plot(np.log10(Ccro45),'g-',linewidth=2, label="Secondary (XY)");
plt.plot(np.log10(Ccro45b),'g^',linewidth=2, label="Secondary (YX)");
#plt.plot(np.log10(Cdom45+Ccro45),'r-', linewidth=1, label="Total");
plt.legend(loc='upper right');
plt.xlabel('Iteration',fontsize=12); plt.ylabel('Intensity',fontsize=12);

plt.figure()
plt.plot(np.log10(Cdomxa),'ko-',linewidth=3,label="Primary");
plt.plot(np.log10(Ccroxa),'gx:',linewidth=2,label="Secondary");
plt.plot(np.log10(Cdomxa+Ccroxa),'r-',linewidth=1, label="Total");
plt.legend(loc='upper right');
plt.xlabel('Iteration',fontsize=12); plt.ylabel('Intensity',fontsize=12);


#%%

#lambda functions for random modulation
cmddom = lambda  :  0.001*np.pi*(np.random.rand(len(DHinfo['sols45'][-1])) - 0.5)
cmdcro = lambda  :   0.01*np.pi*(np.random.rand(len(DHinfo['sols45'][-1])) - 0.5)
# calculate intensities with random modulation
Nmodulations = 50
Ccromod45  = []
Cdommod45  = []
Ccromodax = []
Cdommodax = []
Imodoa45   = []
Imodoaax  = []
for k in range(Nmodulations):
   cd = cmddom();
   if k == 0:
      cd *= 0.
   Cdommodax.append(Cdax(-1, cd))
   Ccromodax.append(Ccax(-1, cd))
   Imodoaax.append(Ioaax(-1, cd))
Cdommodax = np.array(Cdommodax)
Ccromodax = np.array(Ccromodax)
Imodoaax  = np.array(Imodoaax)
print(f"DH primary intensity:{Cdommodax[0]}, DH secondary intensity:{Ccromodax[0]}, Off-axis source intensity: {Imodoaax[0]}" )

plt.figure();  # figure showing modulation about DH command
plt.plot(Cdommodax,'ko', label='Primary');
plt.plot(Ccromodax*25.,'rx', label='25*Secondary');
plt.plot(Imodoaax*2.e-9, 'g*',label='5.e-9 Planet' );
plt.legend(loc='lower right'); plt.title('Random Modulation about DH Command');
plt.xlabel('modulation trial');plt.ylabel('Intensity (contrast)');


#%%
plt.figure(); plt.imshow(np.log10(Ioa45dh[100:180,100:180] + 1.e-4),origin='lower',cmap='coolwarm'); plt.colorbar();
plt.title('off-axis source, dark hole DM command')

fig, axs = plt.subplots(
    3, 2,
    figsize=(6, 7.5),     # Plus compact verticalement
    sharex='col',
    sharey='row'
)
# Réduire les marges entre subplots
plt.subplots_adjust(wspace=0.05, hspace=0.02)
images = [
    np.log10(I0[100:180, 100:180] + 1e-7),
    np.log10(I0c[100:180, 100:180] + 1e-10),
    np.log10(Idh45[100:180, 100:180] + 1e-10),
    np.log10(Idh45c[100:180, 100:180] + 1e-12),
    np.log10(IdhXax[100:180, 100:180] + 1e-9),
    np.log10(IdhXaxc[100:180, 100:180] + 1e-10),
]
titles = [
    "Nominal Primary", "Nominal Secondary",
    "Primary Hole #1", "Secondary Hole #1",
    "Primary Hole #2", "Secondary Hole #2"
]
for idx, ax in enumerate(axs.flat):
    im = ax.imshow(images[idx], origin='lower', cmap='coolwarm')
    ax.set_title(titles[idx], fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
# Optimisation finale de l'agencement
plt.tight_layout(pad=0.5)  # Réduit encore l’espace autour
plt.show()







#%%
if True:
   cutpix = 30; isz = (128 - cutpix)/128; ext = [-isz,isz,-isz,isz];ii1=cutpix;ii2=256-cutpix;
   plt.figure();plt.imshow(np.log10(I0[ii1:ii2,ii1:ii2]+1.e-9) ,extent=ext,origin='lower',cmap='coolwarm'),plt.colorbar();
   plt.title("Primary PSF",fontsize=10);plt.xlabel("mm",fontsize=10);plt.ylabel("mm",fontsize=10);
   plt.figure();plt.imshow(np.log10(I0c[ii1:ii2,ii1:ii2]+1.e-12) ,extent=ext,origin='lower',cmap='coolwarm'),plt.colorbar();
   plt.title("Secondary PSF",fontsize=10);plt.xlabel("mm",fontsize=10);plt.ylabel("mm",fontsize=10);


if False:
   #plt.figure(); plt.imshow(np.log10(I0[100:180,100:180] + 1.e-6) ,origin='lower',cmap='coolwarm');plt.colorbar();
   plt.figure(); plt.imshow(np.log10(IdhXax[100:180,100:180] + 1.e-10),origin='lower',cmap='coolwarm');plt.colorbar();
   plt.figure(); plt.imshow(np.log10(Idh45[ 100:180,100:180] + 1.e-10),origin='lower',cmap='coolwarm');plt.colorbar();

   #plt.figure(); plt.imshow(np.log10(I0c[100:180,100:180] + 1.e-12) ,origin='lower',cmap='coolwarm');plt.colorbar();
   plt.figure(); plt.imshow(np.log10(IdhXaxc[100:180,100:180] + 1.e-14) ,origin='lower',cmap='coolwarm');plt.colorbar();
   #plt.figure(); plt.imshow(np.log10(Idh45c[100:180,100:180] + 1.e-14) ,origin='lower',cmap='coolwarm');plt.colorbar();

   plt.figure(); plt.imshow(np.log10(Ioa45dh[100:180,100:180] + 1.e-4),origin='lower',cmap='coolwarm'); plt.colorbar();
   plt.title('off-axis source, dark hole DM command')
   plt.figure(); plt.imshow(np.log10(IoaXaxdh[100:180,100:180] + 1.e-4),origin='lower',cmap='coolwarm'); plt.colorbar();
   plt.title('off-axis source, dark hole DM command')
