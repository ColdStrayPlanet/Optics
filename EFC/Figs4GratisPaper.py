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

#%%
lamD = 5.89 # "lambda/D" in pixel units
PropMatLoc = "/home/rfrazin/Py/EFCSimData/"
Domfn =  'SysMatNorm_Xx_BigBeam2Cg256x256_Lam0.9_33x33.npy'
Crofn =  'SysMatNorm_Xy_BigBeam2Cg256x256_Lam0.9_33x33.npy'
Dom2fn = 'SysMatNormPiShift_Yy_BigBeam2Cg256x256_Lam0.9_33x33.npy'
Cro2fn = 'SysMatNormPiShift_Yx_BigBeam2Cg256x256_Lam0.9_33x33.npy'
DomMat = np.load(join(PropMatLoc,Domfn));
CroMat = np.load(join(PropMatLoc,Crofn));
DomMat2 = np.load(join(PropMatLoc,Dom2fn));
CroMat2 = np.load(join(PropMatLoc,Cro2fn));
DHinfo = None

A = EFCm.EFC(EFCm.B21, EFCm.B33, DomMat, CroMat, Dom2PropMat=DomMat2, Cross2PropMat=CroMat2)
#%%
loadDHinfo = True
if loadDHinfo:
   DHinfofilename = "DHinfo02092026_XXYYopt.pickle"
   with open(DHinfofilename, "rb") as fp: DHinfo = pickle.load(fp)
else:  # dont load DHinfo -- do the optimizations
   pl45 = A.MakePixList( np.round(128 + lamD*np.array([2,5,2,5]) ).astype('int') ) #pixel lists for dark holes
   plxa = A.MakePixList( np.round(128 + lamD*np.array([2,5,-1.5,1.5])).astype('int') )
   print(f"The dark holes have {len(pl45)} and {len(plxa)} pixels.")
   (out45, sols45, cost45) = A.DigDominantHole(np.zeros(441,), pl45, TwoDoms=True, DMconstr=np.pi/2)
   (outxa, solsxa, costxa) = A.DigDominantHole(np.zeros(441,), plxa, TwoDoms=True, DMconstr=np.pi/2)

# evaluate all of the mean intensities in the dark holes
   if DHinfo is None:
      DHinfo = dict()
   DHinfo['sols45'] = sols45;  DHinfo['solsxa'] = solsxa;
   DHinfo['plxa'] = plxa; DHinfo['pl45'] = pl45;
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


#%%

AD1 = A.SysD[ DHinfo['pl45'],:]
AD2 = A.SysD2[DHinfo['pl45'],:]
AC1 = A.SysC[ DHinfo['pl45'],:]
AC2 = A.SysC2[DHinfo['pl45'],:]

Ud1, sd1, Vd1 = np.linalg.svd(AD1, full_matrices=True); Vd1 = np.conj(Vd1.T);
Ud2, sd2, Vd2 = np.linalg.svd(AD2, full_matrices=True); Vd2 = np.conj(Vd2.T);
Uc1, sc1, Vc1 = np.linalg.svd(AC1, full_matrices=True); Vc1 = np.conj(Vc1.T);
Uc2, sc2, Vc2 = np.linalg.svd(AC2, full_matrices=True); Vc2 = np.conj(Vc2.T);
normd1 = sd1.max(); normd2 = sd2.max(); normc1 = sc1.max(); normc2 = sc2.max()


#ADr = A.SysD[DHinfo['pl45'],:]
#ACr = A.SysC[DHinfo['pl45'],:]
#Ud, sd, Vd = np.linalg.svd(ADr, full_matrices=True); Vd = np.conj(Vd.T);
#Uc, sc, Vc = np.linalg.svd(ACr, full_matrices=True); Vc = np.conj(Vc.T);


#%% Linear System Analysis

#It is easy to show that for two complex numbers a and b that the value of the
#  real number s that minimizes |(a - exp(j s)*b)|^2 is s = arg(a conj(b) ).
#  The same thing works for matrices with the dot product of two matrices A and B
#  defined as \sum_i \sum_j A_{ij} B_{ij}^*
#
def matrixdot(A,B):
      return( np.sum( A*np.conj(B) ) )
def MatrixDiffMeasure(A,B):
   if A.shape != B.shape:
      raise ValueError("Input matrices A and B must have the same shape.")
   if np.allclose(A, np.zeros(A.shape)) or np.allclose(B, np.zeros(B.shape)):
      raise ValueError("Input matrices A and B both must be nonzero.")
   s = np.exp(1j*np.angle( matrixdot(A,B) ))
   AA = np.real(matrixdot(A,A));  BB = np.real(matrixdot(B,B))
   h = np.real( matrixdot(A -s*B, A - s*B) )
   h /= np.sqrt(AA*BB)
   return(h)

#The columns of the matrices A and B are singular vectors and svA and svB
#  are the corresponding singular values
def SingularVectorCompare(A,B,svA, svB,weighting='quadratic'):
    weighttypes =  {'quadratic': None, 'linear': None}
    if A.shape != B.shape:
       raise ValueError("Input matrices A and B must have the same shape.")
    if svA.shape != svB.shape:
       raise ValueError("Input vectors svA and svB must have the same shape.")
    if np.ndim(A) != 2 or np.ndim(svA) != 1 or np.ndim(svB) != 1 or svA.shape[0] != svB.shape[0] :
       raise ValueError("At least one of the inputs is misdimensioned.")
    if weighting not in weighttypes:
       raise ValueError(f"kwarg weighting must be one of the allowed options: {weighttypes.keys()}")
    if weighting == 'linear':
       pass
    if weighting == 'quadratic':
       svA = svA**2; svB = svB**2

    if len(svA) < A.shape[1]:
       A = A[:,:len(svA)];
       B = B[:,:len(svB)]
    sumA = np.sum(svA); sumB = np.sum(svB)
    A = (A*svA)/sumA; B = (B*svB)/sumB
    numerator = np.linalg.norm( np.conj(B.T).dot(A),'fro')**2
    denominator = np.sqrt( (np.linalg.norm( np.conj(B.T).dot(B),'fro')**2) *
                           (np.linalg.norm( np.conj(A.T).dot(A),'fro')**2))
    return ( numerator/denominator )


#%%

sd2d1 = []; sc1d1 = []; sc1c2 = []; sd1c1 = [];
for k in range(len(sd1)):
   sc1d1.append(np.linalg.norm(AC1@Vd1[:,k])/normc1)
   sd2d1.append(np.linalg.norm(AD2@Vd1[:,k])/normd2)
   sc1c2.append(np.linalg.norm(AC1@Vc2[:,k])/normc1)
   sd1c1.append(np.linalg.norm(AD1@Vc1[:,k])/normd1)
sd2d1 = np.array(sd2d1); sc1d1 = np.array(sc1d1); sc1c2 = np.array(sc1c2); sd1c1 = np.array(sd1c1)
#%%
plt.figure()#figsize=(6,4))
plt.plot(np.log10(sc1[:200]/normc1), 'ko:', linewidth=4, markersize=9, label='XY Spectrum')
plt.plot(np.log10(sc1d1[:200]), 'rx-', linewidth=2, label='XY-XX Cross Spectrum')
plt.xlabel('Mode index')
plt.ylabel('Normalized magnitude')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure()#figsize=(6,4))
plt.plot(np.log10(sd1[:200]/normd1), 'ko:', linewidth=4, markersize=9, label='XX Spectrum')
plt.plot(np.log10(sd1c1[:200]), 'rx-', linewidth=2, label='XX-XY Cross Spectrum')
plt.xlabel('Mode index')
plt.ylabel('Normalized magnitude')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure()#figsize=(6,4))
plt.plot(np.log10(sc1[:200]/normc1), 'ko:', linewidth=4, markersize=9, label='XY Spectrum')
plt.plot(np.log10(sc1c2[:200]), 'rx-', linewidth=2, label='XY-YX Cross Spectrum')
plt.xlabel('Mode index')
plt.ylabel('Normalized magnitude')
plt.legend()
plt.grid(True)
plt.tight_layout()



plt.figure()#figsize=(6,4))
plt.plot(np.log10(sd2[:200] / normd2), 'ko:', linewidth=4, markersize=9, label='YY Spectrum')
plt.plot(np.log10(sd2d1[:200] ), 'rx-', linewidth=1.5, label='YY-XX Cross Spectrum')
plt.xlabel('Mode index')
plt.ylabel('Normalized magnitude')
plt.legend()
plt.grid(True)
plt.tight_layout()


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
Cdom45  = np.array(DHinfo['cdom45'])  ; Ccro45  = np.array(DHinfo['ccro45'])
Cdom45b = np.array(DHinfo['cdom452']) ; Ccro45b = np.array(DHinfo['ccro452'])
Cdomxa  = np.array(DHinfo['cdomxa'])  ; Ccroxa  = np.array(DHinfo['ccroxa'])
Cdomxab = np.array(DHinfo['cdomxa2']) ; Ccroxab = np.array( DHinfo['ccroxa2'])

plt.figure(); plt.title("Dark Hole Intensity Vs. Iteration");
plt.plot(np.log10(Cdom45/2),'k',linewidth=6, label="Primary (XX)");
plt.plot(np.log10(Cdom45b/2),'r',linewidth=2, label="Primary (YY)");
plt.plot(np.log10(Ccro45/2),'purple',linewidth=6, label="Secondary (XY)");
plt.plot(np.log10(Ccro45b/2),'cyan',linewidth=2, label="Secondary (YX)");
plt.legend(loc='upper right');
plt.xlabel('Iteration',fontsize=12); plt.ylabel('Intensity',fontsize=12);

plt.figure(); plt.title("Dark Hole Intensity Vs. Iteration");
plt.plot(np.log10(Cdomxa/2),'k',linewidth=6, label="Primary (XX)");
plt.plot(np.log10(Cdomxab/2),'r',linewidth=2, label="Primary (YY)");
plt.plot(np.log10(Ccroxa/2),'purple',linewidth=6, label="Secondary (XY)");
plt.plot(np.log10(Ccroxab/2),'cyan',linewidth=2, label="Secondary (YX)");
plt.legend(loc='upper right');
plt.xlabel('Iteration',fontsize=12); plt.ylabel('Intensity',fontsize=12);



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


#%% Make the full page figures showing the nominal and dark hole results.
DHinfofilename = "DHinfo02092026_XXYYopt.pickle"
with open(DHinfofilename, "rb") as fp: dxy = pickle.load(fp)
DHinfofilename = "DHinfo02092026_XXopt.pickle"
with open(DHinfofilename, "rb") as fp: dx = pickle.load(fp)
cmd0 = np.zeros((441,))


#%%  This is for the full page figures with 24 images
def make_5x4_figure(images, page_size=(8.5, 11),
                left_margin=1, right_margin=1,
                top_margin=1, bottom_margin=1,
                caption_space=1.5,dotcolor='green',centerdot=None):
    """
    Crée une figure Letter portrait avec grille 5x4 d´images et espace pour caption.
    images : list d'images (length = 20)
    axes[0,0] is images[0], axes[0, 1] is images[1], axes[1, 0] is images[4], etc.
    axes[0,0] is the upper left.
    page_size : (largeur, hauteur) en pouces
    margins : marges en pouces
    caption_space : hauteur de la zone pour caption en pouces
    """
    fig_width, fig_height = page_size
    fig = plt.figure(figsize=page_size)

    # coordonnées normalisées
    left = left_margin / fig_width
    right = 1 - right_margin / fig_width
    bottom = (bottom_margin + caption_space) / fig_height
    top = 1 - top_margin / fig_height
    gs = fig.add_gridspec(
        5, 4, # grille 6x4
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=0.05,
        hspace=0.05)
    axes = gs.subplots()
    for ax, img in zip(axes.flat, images):
       im = ax.imshow(img, aspect='equal', cmap='coolwarm',origin='lower',vmax=img.max(),vmin=img.min())
       if centerdot is not None:
          ax.plot(centerdot[0],centerdot[1],marker='o',color=dotcolor,markersize=12)
       ax.axis('off')
       cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
       ticks = np.linspace(np.floor(img.min()), np.floor(img.max()),4)
       cbar.set_ticks(ticks)
       cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
       cbar.ax.tick_params(labelsize=12)
       for label in cbar.ax.get_yticklabels():
           label.set_fontweight('bold')

    return(None)

#%%
#the make list of images for 5x4 grid
DHinfofilename = "DHinfo02092026_XXYYopt.pickle";
with open(DHinfofilename, "rb") as fp: dxy = pickle.load(fp)
DHinfofilename = "DHinfo02092026_XXopt.pickle";
with open(DHinfofilename, "rb") as fp: dx = pickle.load(fp)

sol45a = dx['sols45'][-1]; sol45b = dxy['sols45'][-1]
solxaa = dx['solsxa'][-1]; solxab = dxy['solsxa'][-1]
i1=100; i2=180; centerdot=(28,28)

cmd0 = np.zeros(dx['sols45'][-1].shape)
#intensities are halved in order to agree with the iteration figure.  This way, the total intensity is the sum of all four images
im0  = A.DMcmd2Intensity(cmd0,   pmat='dom'   ,return_grad=False).reshape((256,256))[i1:i2,i1:i2]
im0 = np.log10(im0/2 + 1.e-8)
im1  = A.DMcmd2Intensity(cmd0,   pmat='dom2'   ,return_grad=False).reshape((256,256))[i1:i2,i1:i2]
im1 = np.log10(im1/2 + 1.e-8)
im2  = A.DMcmd2Intensity(cmd0,   pmat='cross' ,return_grad=False).reshape((256,256))[i1:i2,i1:i2]
im2 = np.log10(im2/2 + 1.e-11)
im3  = A.DMcmd2Intensity(cmd0,   pmat='cross2' ,return_grad=False).reshape((256,256))[i1:i2,i1:i2]
im3 = np.log10(im3/2 + 1.e-11)

im4  = A.DMcmd2Intensity(sol45a,   pmat='dom'   ,return_grad=False).reshape((256,256))[i1:i2,i1:i2]
im4 = np.log10(im4/2 + 1.e-11)
im5  = A.DMcmd2Intensity(sol45a,   pmat='dom2'   ,return_grad=False).reshape((256,256))[i1:i2,i1:i2]
im5 = np.log10(im5/2 + 1.e-11)
im6  = A.DMcmd2Intensity(sol45a,   pmat='cross' ,return_grad=False).reshape((256,256))[i1:i2,i1:i2]
im6 = np.log10(im6/2 + 1.e-11)
im7  = A.DMcmd2Intensity(sol45a,   pmat='cross2' ,return_grad=False).reshape((256,256))[i1:i2,i1:i2]
im7 = np.log10(im7/2 + 1.e-11)

im8  = A.DMcmd2Intensity(sol45b,   pmat='dom'   ,return_grad=False).reshape((256,256))[i1:i2,i1:i2]
im8 = np.log10(im8/2 + 1.e-11)
im9  = A.DMcmd2Intensity(sol45b,   pmat='dom2'   ,return_grad=False).reshape((256,256))[i1:i2,i1:i2]
im9 = np.log10(im9/2 + 1.e-11)
im10  = A.DMcmd2Intensity(sol45b,   pmat='cross' ,return_grad=False).reshape((256,256))[i1:i2,i1:i2]
im10 = np.log10(im10/2 + 1.e-11)
im11  = A.DMcmd2Intensity(sol45b,   pmat='cross2' ,return_grad=False).reshape((256,256))[i1:i2,i1:i2]
im11 = np.log10(im11/2 + 1.e-11)

im12  = A.DMcmd2Intensity(solxaa,   pmat='dom'   ,return_grad=False).reshape((256,256))[i1:i2,i1:i2]
im12 = np.log10(im12/2 + 1.e-9)
im13  = A.DMcmd2Intensity(solxaa,   pmat='dom2'   ,return_grad=False).reshape((256,256))[i1:i2,i1:i2]
im13 = np.log10(im13/2 + 1.e-9)
im14  = A.DMcmd2Intensity(solxaa,   pmat='cross' ,return_grad=False).reshape((256,256))[i1:i2,i1:i2]
im14 = np.log10(im14/2 + 1.e-11)
im15  = A.DMcmd2Intensity(solxaa,   pmat='cross2' ,return_grad=False).reshape((256,256))[i1:i2,i1:i2]
im15 = np.log10(im15/2 + 1.e-11)

im16  = A.DMcmd2Intensity(solxab,   pmat='dom'   ,return_grad=False).reshape((256,256))[i1:i2,i1:i2]
im16 = np.log10(im16/2 + 1.e-9)
im17  = A.DMcmd2Intensity(solxab,   pmat='dom2'   ,return_grad=False).reshape((256,256))[i1:i2,i1:i2]
im17 = np.log10(im17/2 + 1.e-9)
im18  = A.DMcmd2Intensity(solxab,   pmat='cross' ,return_grad=False).reshape((256,256))[i1:i2,i1:i2]
im18 = np.log10(im18/2 + 1.e-11)
im19  = A.DMcmd2Intensity(solxab,   pmat='cross2' ,return_grad=False).reshape((256,256))[i1:i2,i1:i2]
im19 = np.log10(im19/2 + 1.e-11)

imagelist = [im0, im1, im2, im3, im4, im5, im6, im7, im8, im9,
          im10, im11, im12, im13, im14, im15, im16, im17, im18, im19]
#%%
make_5x4_figure(imagelist,left_margin=0.1, right_margin=0.8, top_margin=0.0, bottom_margin=0.02,
caption_space=0,dotcolor='green',centerdot=centerdot)
