#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 17:12:22 2023
@author: Richard Frazin

This is for analysis of the my LightTrans simulations of a coronagraph
with a quilt pattern
"""
import numpy as np
import matplotlib.pyplot as plt

#data_dir = "/home/rfrazin/Optics/SpaceCoronagraphy/LightTransSimData/LyotStopExcluded/"
data_dir = "E:/MyOpticalSetups/OAPstuff/CoatedSilver/"
fnbase = "OffAxNoLyot_8ld_45deg_Oversamp7gridless8acc-1"
figstr = "[8 $\lambda/D$, 45deg]"
extent = (-1, 1, -1, 1)  #corners of image

l1 = "lamD8_coat473nm_908nm_"; l2= "_WL1_908nm_"
sxx = np.load(data_dir + l1 + "Xinput" + l2 + "Ex.npy")
sxy = np.load(data_dir + l1 + "Xinput" + l2 + "Ey.npy")
syx = np.load(data_dir + l1 + "Yinput" + l2 + "Ex.npy")
syy = np.load(data_dir + l1 + "Yinput" + l2 + "Ey.npy")


#calculate Stokes Parameters
sI  = np.abs(sxx)**2 + np.abs(sxy)**2 + np.abs(syx)**2 + np.abs(syy)**2
sQ  = np.abs(sxx)**2 - np.abs(sxy)**2 + np.abs(syx)**2 - np.abs(syy)**2
sU  =  2*np.real( sxx*np.conj(sxy) + syx*np.conj(syy) )
sV  = -2*np.imag( sxx*np.conj(sxy) + syx*np.conj(syy) )

if False:   # I was hoping the rotated basis would make Q less noisy.  It doesn't
   bb = np.cos(np.pi/4); 
   sxa =    bb*sxx + bb*sxy; sya =    bb*syx + bb*syy 
   sxb = -1*bb*sxx + bb*sxy; syb = -1*bb*syx + bb*syy
   sQa = -2*np.real( np.conj(sxa)*sxb + np.conj(sya)*syb )
   sUa = np.abs(sxa)**2 - np.abs(sxb)**2 + np.abs(sya)**2 - np.abs(syb)**2 
   sVa =  2*np.imag( np.conj(sxa)*sxb + np.conj(sya)*syb  )

#polarization intensity and fraction
pI = sQ**2 + sU**2 + sV**2 
pf = pI/sI**2

#my polarization fraction
#mypf = (np.abs(sQ) + np.abs(sU) + np.abs(sV) )/sI

#plt.figure(); plt.imshow(np.log10(np.abs(sxx)**2),extent=extent,cmap='seismic',origin='lower',vmax=1,vmin=-8  );
#plt.colorbar();plt.title('|XX|^2');plt.xlabel('mm');plt.ylabel('mm');

#plt.figure(); plt.imshow(np.log10(np.abs(sxy)**2),extent=extent,cmap='seismic',origin='lower',vmax=-5,vmin=-8 );
#plt.colorbar();plt.title('|XY|^2');plt.xlabel('mm');plt.ylabel('mm');

#plt.figure(); plt.imshow(np.log10(np.abs(sxy)**2),extent=extent,cmap='seismic',origin='lower',vmax=-8,vmin=-12);
#plt.colorbar();plt.title('|XY|^2');plt.xlabel('mm');plt.ylabel('mm');

plt.figure(); plt.imshow(np.log10(1.e-5+sI), extent=extent, cmap='seismic', origin='lower');
plt.colorbar(); plt.title('Stokes I ' + figstr );plt.xlabel('mm');plt.ylabel('mm');

#plt.figure(); plt.imshow(np.log10(np.abs(sQ)),extent=extent, cmap='seismic', origin='lower',vmax=-2,vmin=-7);
#plt.colorbar(); plt.title('|Stokes Q|');plt.xlabel('mm');plt.ylabel('mm');

#plt.figure(); plt.imshow(np.log10(np.abs(sU)),extent=extent, cmap='seismic', origin='lower',vmax=-2,vmin=-7);
#plt.colorbar(); plt.title('|Stokes U|');plt.xlabel('mm');plt.ylabel('mm');

#plt.figure(); plt.imshow(np.log10(np.abs(sV)),extent=extent, cmap='seismic', origin='lower',vmax=-2,vmin=-7);
#plt.colorbar(); plt.title('|Stokes V|');plt.xlabel('mm');plt.ylabel('mm');


#===========================

axv = np.linspace(-1000,1000,1024)

plt.figure(); a1 = plt.subplot(1,1,1);
p1 = a1.plot(axv,sI[511,:],'ko-',label="Stokes I")
p2 = a1.plot(axv,sQ[511,:]*1000,'rx', label="(Stokes Q)*1000")
p3 = a1.plot(axv,sU[511,:]*1000,'c*', label="(Stokes U)*1000")
p4 = a1.plot(axv,sV[511,:]*1000,'gh', label="(Stokes V)*1000")
a1.legend();
plt.title("horizontal profile " + figstr);
plt.xlabel("dist from center of profile (microns)");
plt.ylabel("intensity");

plt.figure(); a2 = plt.subplot(1,1,1);
p1 = a2.plot(axv,sI[:,511],'ko-',label="Stokes I")
p2 = a2.plot(axv,sQ[:,511]*1000,'rx', label="(Stokes Q)*1000")
p3 = a2.plot(axv,sU[:,511]*1000,'c*', label="(Stokes U)*1000")
p4 = a2.plot(axv,sV[:,511]*1000,'gh', label="(Stokes V)*1000")
a2.legend()
plt.title("vertical profile " + figstr);
plt.xlabel("dist from center of profile (microns)");
plt.ylabel("intensity");




