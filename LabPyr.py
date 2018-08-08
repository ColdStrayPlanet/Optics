#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 14:55:31 2018
@author:

This code is a descendant of Pyr.py.   This version is designed to accurately
reproduce lab measurements, and so allows for different paddings of the two 
FFTs and treats alignments (and possibly other) errors with free parameters.  

params is a dictionary of the basic parameters of the numerical model

"""

import numpy as np
from scipy.signal import convolve as conv
from scipy.interpolate import interp1d, interp2d
import matplotlib.pyplot as plt
import FourierOptics

# dict of nominal optical system paramers
#  units: lengths are in microns, angles are in degrees
pyrparams = dict()
pyrparams['beam_diameter'] = 1.e3 # input beam diameter (microns)
pyrparams['wavelength'] = 0.6328 # wavelength (microns)
pyrparams['indref'] = 1.5 # pyramid index of refraction
pyrparams['pyramid_slope_deg'] = 3# 10.5  # slope of pyramid faces relative to horizontal (degrees)
pyrparams['pyramid_roofsize'] = 16 # length of one side of square pyramid roof (microns)
pyrparams['pyramid_height'] = 1.e4 # (microns)  height of pyramid
pyrparams['n_starting_points'] = 150  # number of resolution elements in initital beam diameter
pyrparams['D_e_2_l1'] = 200.e3 # nominal distance from entrance pupil to lens1 (microns)
pyrparams['f1'] = 100.e3 # focal length of lens #1 (focuses light on pyramid tip)
pyrparams['lens1_fill_diameter'] = 3.e3  #  computational beam width at lens#1.  This matters!
pyrparams['beam_diameter_at_pyramid'] = 1.e3  # focal spot size at pyramid width
pyrparams['D_l1_2_pyr'] = 100.e3 # distance from lens1 to pyramid tip
pyrparams['D_l1_2_detector'] = 200.e3  # distance from lens to detector in 4f system
pyrparams['apex_diam'] = 8 # diameter of focal spot at pyramid apex in units of lambda_over_D (set by stop)
pyrparams['D_foc_2_l2'] = 50.e3 # distance from focus to lens #2 (includes effective OPL thru prism)
pyrparams['f2'] = 5023.e3 # focal length of lens #2
pyrparams['diam_lens2'] = 6.e3 # effective diameter of lens2 (set by a stop)
pyrparams['D_l2_2_detector'] = 10.e3 # distrance from lens2 to detector
pyrparams['detector_width'] = 6.e3 # width of detector
pyrparams['max_chirp_step_deg'] = 90  # maximum allowed value (degrees) in chirp step for Fresnel prop
pyrparams['max_lens_step_deg'] = 20 # maximum step size allowed for lens phase screen
pyrparams['max_plane_step_deg'] = 20  # (degrees) maximum phase step allowed for planar phase screen
pyrparams['max_pyramid_phase_step'] = 20  # maximum step (degrees) allowed for pyramid phase ramp
pyrparams['interp_style'] = 'cubic'  # type of 2d interpolator

def sq(field):
    return(np.real(field*np.conj(field)))

class OpticalModels():
    def __init__(self, params=pyrparams):

        self.params = params
        self.grads = dict()  # dictionary of gradients
        nx = self.params['n_starting_points']
        self.field_Start1D = np.ones(nx)  # initial pupil field

        # make a circular input field
        s = np.linspace(-1 + .5/nx, 1 - .5/nx, nx)
        [sx, sy] = np.meshgrid(s, s, indexing='xy')
        cr = np.where(sx*sx + sy*sy > 1)
        f = np.ones((nx,nx))
        f[cr[0],cr[1]] = 0.
        
        self.field_Start2D = f
        diam0 = self.params['beam_diameter']
        dx = diam0/nx
        self.x_Start = np.linspace(-diam0/2 + dx/2, diam0/2 - dx/2, nx)  # initial spatial grid

        return

    def PropBeamNoFocus(self, PhaseScreenTest=False):
        FO = FourierOptics.FourierOptics(pyrparams)
        diam = self.params['beam_diameter']
        diam1 = 1.e3 + diam
        z = self.params['D_e_2_l1'] + self.params['D_l1_2_detector']
        x = 1.*self.x_Start
        field1D = 1.*self.field_Start1D

        if PhaseScreenTest:
            diam1 += 1.5e3
            aa = 1.e3/z
            norm = np.array([aa, np.sqrt(1 - aa*aa)])
            field1D, xout = FO.ApplyPlanarPhaseScreen1D(field1D, x, norm, nA=1, nB=2)
            #undo most of phase screen by simulating a wedge
            field1D = FO.ApplySnellAtPlane1D(field1D, xout, [0,1], nA=2, nB=1)

        field1D, xout = FO.ConvFresnel1D(field1D, x, diam1, z, set_dx=True, return_derivs=False)

        plt.figure()
        intensity = np.real(field1D*np.conj(field1D))
        plt.plot(xout/1.e3, intensity/np.max(intensity), 'x-')
        plt.title('unfocused beam at detector (full res)')
        plt.xlabel('x (mm)')
        plt.ylabel('intensity')

        x = 1.*self.x_Start
        field2D = 1.*self.field_Start2D
        if PhaseScreenTest:
            aa = 1.e3/z
            norm = np.array([0, aa, np.sqrt(1 - aa*aa)])  # vertical displacement
            field2D, xout = FO.ApplyPlanarPhaseScreen2D(field2D, x, norm, nA=1, nB=2)

        field2D, xout = FO.ConvFresnel2D(field2D, xout, diam1, z, set_dx=True, return_derivs=False)
        br2D = sq(field2D)
        br2D /= np.max(br2D)

        xoutmm = xout/1.e3
        plt.figure()
        plt.imshow(br2D, extent=[xoutmm[0], xoutmm[-1], xoutmm[0], xoutmm[-1]]); plt.colorbar();

        return

    #This focuses a beam on a wedge.  When the angle tip_deg == 0, the normal to the
    #  exit surface is in the z-direction.
    #Simulates a 4f system with the wedge at the focal spot.
    #tip_deg - tip angle of wedge (degrees)
    #dist_err - (microns) error in z position of wedge (should be at focal spot)
    def FocusBeamOnInclinedWedge4F(self, tip_deg=0, dist_error=0):
        tip = tip_deg*np.pi/180
        ind_ref=1.5  # index of refaction
        WedgeThickness=5.e4  # wedge width at entry point (microns)
        WedgeAngle = 2*np.pi/180  #  angle between entry surface and exit surface
        focal_length = 10.e4
        diam = self.params['beam_diameter']
        lam = self.params['wavelength']
        d0 = focal_length + dist_error #distance from entrance to wedge
        d1 = focal_length - ind_ref*WedgeThickness - dist_error # distance from wedge exit to detector

        #make normal vectors for wedge entrance and exit.  rotate to include tip
        norm0 = [-np.sin(WedgeAngle), np.cos(WedgeAngle)]  #  normal at wedge entrace (note +z comp.)
        norm1 = [0, 1]  # normal at wedge exit
        rmat = np.array([[np.cos(tip), np.sin(-tip)], [np.sin(tip), np.cos(tip)]])
        norm0 = rmat.dot(norm0)
        norm1 = rmat.dot(norm1)

        x1d = 1.*self.x_Start
        field1d = 1.*self.field_Start1D
        FO = FourierOptics.FourierOptics(pyrparams)

        #prop to lens, dist = 2*focal_length, apply lens
        field1d, x1d = FO.ConvFresnel1D(field1d, x1d, 4*diam, 2*focal_length, set_dx=True, return_derivs=False)
        field1d, x1d = FO.ApplyThinLens1D(field1d, x1d, 0., focal_length, return_derivs=False)
        #prop to wedge
        spot_size = 4*2*focal_length*lam/diam + 2*np.abs(dist_error)*diam/focal_length
        field1d, x1d = FO.ConvFresnel1D(field1d, x1d, spot_size, d0, set_dx=True, return_derivs=False)
        #refract at wedge entrance
        field1d = FO.ApplySnellAtPlane1D(field1d, x1d, norm0, nA=1, nB=ind_ref)
        #propagate through wedge
        spot_size = 2*WedgeAngle*WedgeThickness*(ind_ref-1) + 2*(dist_error + WedgeThickness)*diam/focal_length
        field1d, x1d = FO.ConvFresnel1D(field1d, x1d, spot_size, WedgeThickness,
                                        index_of_refraction=ind_ref, set_dx=True, return_derivs=False)
        #refract at wedge exit
        field1d = FO.ApplySnellAtPlane1D(field1d, x1d, norm1, nA=ind_ref, nB=1)
        #prop to detector plane
        spot_size = diam + 2*focal_length*(ind_ref - 1)*WedgeAngle
        field1d, x1d = FO.ConvFresnel1D(field1d, x1d, spot_size, d1, index_of_refraction=1,
                                        set_dx=True, return_derivs=False)

        return

    #This simulates propagation of a beam to a lens and then to a plane
    #   that is defocus from the focal length of the lens
    #defocus - the signed distance (microns) from the focal plane
    #   negative distances imply the plane is closer to the lens than
    #     the focal plane.
    def PropBeamFocus(self, defocus=0):
        FO = FourierOptics.FourierOptics(pyrparams)
        diam0 = self.params['beam_diameter'] + 1.e3
        z = self.params['D_e_2_l1']
        field1d, x1d = FO.ConvFresnel1D(self.field_Start1D, self.x_Start, diam0, z, set_dx=True, return_derivs=False)
        field2d, x2d = FO.ConvFresnel2D(self.field_Start2D, self.x_Start, diam0, z, set_dx=True, return_derivs=False)
        br1d = sq(field1d)
        br2d = sq(field2d)

        dx = x1d[1] - x1d[0]
        plt.figure()
        plt.plot(x1d, br1d/np.max(br1d),'x-')
        plt.title('intensity at lens1, z = ' + str(z/1.e4) +  ' cm, $\Delta x$ = ' + str(dx))
        plt.xlabel('x (microns)')
        plt.ylabel('intensity')

        plt.figure()
        plt.imshow(br2d, extent=[x2d[0], x2d[-1], x2d[0], x2d[-1]])
        plt.title('intensity at lens1, z = ' + str(z/1.e4) +  ' cm, $\Delta x$ = ' + str(dx))
        plt.xlabel('x (microns)')
        plt.ylabel('y (microns)')

        focal_length = self.params['f1']
        lens_center = 0
        field1d, x1d = FO.ApplyThinLens1D(field1d, x1d, lens_center, focal_length, return_derivs=False)
        lens_center = [0, 0]
        field2d, x2d = FO.ApplyThinLens2D(field2d, x2d, lens_center, focal_length, return_derivs=False)

        diam1 = 350 + defocus*self.params['beam_diameter']/focal_length
        z = focal_length + defocus
        field1d, x1d = FO.ConvFresnel1D(field1d, x1d, diam1, z, set_dx=True, return_derivs=False)
        field2d, x2d = FO.ConvFresnel2D(field2d, x2d, diam1, z, set_dx=True, return_derivs=False)
        br1d = sq(field1d)
        br2d = sq(field2d)

        dx = x1d[1] - x1d[0]
        plt.figure()
        plt.plot(x1d/5.3, br1d/np.max(br1d),'ko-')
        plt.title('detector intensity, $\Delta x$ = ' + str(dx) + ', defocus = ' + str(defocus/1.e3) + ' mm')
        plt.xlabel('x (pixels)')
        plt.ylabel('intensity')

        plt.figure()
        plt.imshow(np.sqrt(br2d/np.max(br2d)), extent=[x2d[0]/5.3, x2d[-1]/5.3, x2d[0]/5.3, x2d[-1]/5.3])
        plt.title('sqrt brightness')
        plt.xlabel('pixels')
        plt.ylabel('pixels')
        plt.colorbar()

        return


    #This simulates an f4 optical system, in which (image distance)=(object distance)=2f
    def PropF4(self, include_pyramid=False):
        FO = FourierOptics.FourierOptics(pyrparams)
        obd = self.params['D_e_2_l1'] # nominal distance from entrance pupil to lens1 (microns)
        imd = self.params['D_l1_2_detector']
        diam0 = self.params['lens1_fill_diameter']
        

        # propagate field to lens
        z = obd
        field1d, x1d = FO.ConvFresnel1D(self.field_Start1D, self.x_Start, diam0, z, set_dx=True, return_derivs=False)
        field2d, x2d = FO.ConvFresnel2D(self.field_Start2D, self.x_Start, diam0, z, set_dx=True, return_derivs=False)
        br1d = sq(field1d); br1d /= np.max(br1d)
        br2d = sq(field2d); br2d /= np.max(br2d)

        dx = x1d[1] - x1d[0]
        plt.figure()
        plt.plot(x1d, br1d,'x-')
        plt.title('intensity at lens1, z = ' + str(z/1.e4) +  ' cm, $\Delta x$ = ' + str(dx))
        plt.xlabel('x (microns)')
        plt.ylabel('intensity')
        plt.figure()
        dx = x2d[1] - x2d[0]
        plt.imshow(br2d, extent=[x2d[0], x2d[-1], x2d[0], x2d[-1]])
        plt.colorbar();
        plt.title('intensity at lens1, z = ' + str(z/1.e4) +  ' cm, $\Delta x$ = ' + str(dx))
        plt.xlabel('x (microns)')
        plt.ylabel('y (microns)')

        #apply phase screen due to lens
        focal_length = self.params['f1']
        lens_center = 0
        field1d, x1d = FO.ApplyThinLens1D(field1d, x1d, lens_center, focal_length, return_derivs=False)
        lens_center = [0, 0]
        field2d, x2d = FO.ApplyThinLens2D(field2d, x2d, lens_center, focal_length, return_derivs=False)

        #propagate to image plane
        z = imd
        diam1 = self.params['beam_diameter'] + 200
        field1d, x1d = FO.ConvFresnel1D(field1d, x1d, diam1, z, set_dx=True, return_derivs=False)
        field2d, x2d = FO.ConvFresnel2D(field2d, x2d, diam1, z, set_dx=True, return_derivs=False)
        br1d = sq(field1d); br1d /= np.max(br1d)
        br2d = sq(field2d); br2d /= np.max(br2d)

        zd = obd + imd
        dx = x1d[1] - x1d[0]
        plt.figure()
        plt.plot(x1d, br1d,'x-')
        plt.title('intensity at detector, z = ' + str(zd/1.e4) +  ' cm, $\Delta x$ = ' + str(dx))
        plt.xlabel('x (microns)')
        plt.ylabel('intensity')
        plt.figure()
        dx = x2d[1] - x2d[0]
        plt.imshow(br2d, extent=[x2d[0], x2d[-1], x2d[0], x2d[-1]])
        plt.colorbar();
        plt.title('intensity at detector, z = ' + str(zd/1.e4) +  ' cm, $\Delta x$ = ' + str(dx))
        plt.xlabel('x (microns)')
        plt.ylabel('y (microns)')

        return

    #This simulates an f4 optical system, in which (image distance)=(object distance)=2f,
    #   except now there is an pyramid placed at a distance f from the lens
    #center, rotate, tip, tilt are passed to FO.ApplyPyramidPhaseRamp2D
    def PropF4Pyramid(self, fig_list=None, pyr_dist_error=0, center=[0,0], rotate=0, tip=0, tilt=0):
        if len(center) != 2:
            raise Exception("PropF4Pyramid: center parameter must have len=2.")
        FO = FourierOptics.FourierOptics(pyrparams)
        obd = self.params['D_e_2_l1'] # nominal distance from entrance pupil to lens1 (microns)
        focal_length = self.params['f1'] # focal length of lens #1 (focuses light on pyramid tip)
        d2pyr = self.params['D_l1_2_pyr'] + pyr_dist_error # distance from lens1 to pyramid tip
        d_pyr_det = self.params['D_l1_2_detector'] - d2pyr
        diam0 = self.params['lens1_fill_diameter']
        # angle between beams, see Eq. 21 in Sec. 4.1 of Depierraz' thesis
        beta =  2*(self.params['indref'] - 1)*self.params['pyramid_slope_deg']*np.pi/180
        detector_diam = 2*focal_length*np.tan(beta/2) + self.params['beam_diameter'] + 550

        # propagate field to lens
        z = obd
        field1d, x1d = FO.ConvFresnel1D(self.field_Start1D, self.x_Start, diam0, z, set_dx=9.9, return_derivs=False)
        print("at lens: delta x = " + str(x1d[1] - x1d[0]) + " microns, len(x) = " + str(len(x1d)))
        field2d, x2d = FO.ConvFresnel2D(self.field_Start2D, self.x_Start, diam0, z, set_dx=9.9, return_derivs=False)
        br1d = np.abs(field1d); br1d /= np.max(br1d)
        br2d = np.abs(field2d); br2d /= np.max(br2d)

        if 'lens' in fig_list:
            dx = x1d[1] - x1d[0]
            plt.figure()
            plt.plot(x1d, br1d,'x-')
            plt.title('|field| at lens1, z = ' + str(z/1.e4) +  ' cm, $\Delta x$ = ' + str(dx))
            plt.xlabel('x (microns)')
            plt.ylabel('intensity')
            plt.figure()
            dx = x2d[1] - x2d[0]
            plt.imshow(br2d, extent=[x2d[0], x2d[-1], x2d[0], x2d[-1]])
            plt.colorbar();
            plt.title('|field| at lens1, z = ' + str(z/1.e4) +  ' cm, $\Delta x$ = ' + str(dx))
            plt.xlabel('x (microns)')
            plt.ylabel('y (microns)')

        #apply phase screen due to lens
        lens_center = 0
        field1d, x1d = FO.ApplyThinLens1D(field1d, x1d, lens_center, focal_length, return_derivs=False)
        print("after lens: delta x = " + str(x1d[1] - x1d[0]) + " microns, len(x) = " + str(len(x1d)))
        lens_center = [0, 0]
        field2d, x2d = FO.ApplyThinLens2D(field2d, x2d, lens_center, focal_length, return_derivs=False)

        #propagate to pyramid tip
        z = d2pyr
        diam_pyr = self.params['beam_diameter_at_pyramid']
        field1d, x1d = FO.ConvFresnel1D(field1d, x1d, diam_pyr, z, set_dx=0.885, return_derivs=False)
        print("at pyramid: delta x = " + str(x1d[1] - x1d[0]) + " microns, len(x) = " + str(len(x1d)))
        field2d, x2d = FO.ConvFresnel2D(field2d, x2d, diam_pyr, z, set_dx=0.885, return_derivs=False)
        br1d = np.abs(field1d); br1d /= np.max(br1d)
        br2d = np.abs(field2d); br2d /= np.max(br2d)

        if 'pyr_apex' in fig_list:
            dx = x1d[1] - x1d[0]
            plt.figure()
            plt.plot(x1d, br1d,'x-')
            plt.title('|field| at pyramid apex, z = ' + str(d2pyr/1.e4) +  ' cm, $\Delta x$ = ' + str(dx))
            plt.xlabel('x (microns)')
            plt.ylabel('intensity')
            plt.figure()
            dx = x2d[1] - x2d[0]
            plt.imshow(br2d, extent=[x2d[0], x2d[-1], x2d[0], x2d[-1]])
            plt.colorbar();
            plt.title('|field| at pyramid apex, z = ' + str(d2pyr/1.e4) +  ' cm, $\Delta x$ = ' + str(dx))
            plt.xlabel('x (microns)')
            plt.ylabel('y (microns)')

        #apply pyramid phase screen to simulate entering pyramid glass
        field1d, x1d = FO.ApplyPyramidPhaseRamp1D(field1d, x1d, center=0, no_apex=False, return_derivs=False)
        print("after pyramid apex: delta x1d = " + str(x1d[1] - x1d[0]) + " microns, len(x1d) = " + str(len(x1d)))
        #the columns of pyr_ax are unit vectors of the pyramid axes
        field2d, x2d, pyr_ax = FO.ApplyPyramidPhaseRamp2D(field2d, x2d,
                           center=center, rotate=rotate, tip=tip, tilt=tilt, return_derivs=False)
        normal = pyr_ax[:,2]  # normal to bottom face of pyramid glass
        print("after pyramid apex: delta x2d = " + str(x2d[1] - x2d[0]) + " microns, len(x2d) = " + str(len(x2d)))

        #propagate within pyramid glass (!)
        nr = self.params['indref']
        z = self.params['pyramid_height']
        beta = 2*self.params['pyramid_slope_deg']*(nr - 1)*np.pi/180
        diam_pyr = 1.e3  #self.params['beam_diameter'] + z*beta
        field1d, x1d = FO.ConvFresnel1D(field1d, x1d, diam_pyr, z, index_of_refraction=nr,
                                        set_dx=True, return_derivs=False)
        print("after pyramid exit: delta x1d = " + str(x1d[1] - x1d[0]) + " microns, len(x1d) = " + str(len(x1d)))
        field2d, x2d = FO.ConvFresnel2D(field2d, x2d, diam_pyr, z, index_of_refraction=nr,
                                        set_dx=True, return_derivs=False)
        print("after pyramid exit: delta x2d = " + str(x2d[1] - x2d[0]) + " microns, len(x2d) = " + str(len(x2d)))


        if 'pyr_exit' in fig_list:
            dx = x1d[1] - x1d[0]
            br1d = np.abs(field1d); br1d /= np.max(br1d)
            plt.figure()
            plt.plot(x1d/1.e3, br1d,'x-')
            plt.title('|field| at pyramid exit, $\Delta x$ = ' + str(dx))
            plt.xlabel('x (mm)')
            plt.ylabel('intensity')
            plt.figure()
            br2d = np.abs(field2d); br2d /= np.max(br2d)
            plt.imshow(br2d, extent=[x2d[0], x2d[-1], x2d[0], x2d[-1]])
            plt.colorbar();
            plt.title('|field| at pyramid exit, $\Delta x$ = ' + str(dx))
            plt.xlabel('x (microns)')
            plt.ylabel('y (microns)')

        
        #apply phase screen to simulate exit from pyramid bottom
        field2d, x2d = FO.ApplyPlanarPhaseScreen2D(field2d, x2d, normal, nA=nr, nB=1)

        z = d_pyr_det - self.params['pyramid_height']
        field1d, x1d = FO.ConvFresnel1D(field1d, x1d, detector_diam, z, set_dx=True, return_derivs=False)
        print("at detector: delta x = " + str(x1d[1] - x1d[0]) + " microns, len(x) = " + str(len(x1d)))
        field2d, x2d = FO.ConvFresnel2D(field2d, x2d, detector_diam, z, set_dx=True, return_derivs=False)
        br1d = sq(field1d); br1d /= np.max(br1d)
        br2d = sq(field2d); br2d /= np.max(br2d)

        if 'detector' in fig_list:
            dx = x1d[1] - x1d[0]
            plt.figure()
            plt.plot(x1d/1.e3, br1d,'x-')
            plt.title('intensity at detector, $\Delta x$ = ' + str(dx))
            plt.xlabel('x (mm)')
            plt.ylabel('intensity')
            plt.figure()
            plt.imshow(br2d, extent=[x2d[0], x2d[-1], x2d[0], x2d[-1]])
            plt.colorbar();
            plt.title('intensity at detector, $\Delta x$ = ' + str(dx))
            plt.xlabel('x (microns)')
            plt.ylabel('y (microns)')

        return



    # This calculates the field near the focal plane of lens1, which is near the apex of the pyramid
    # pfield - pupil plane field (assume evenly sampled)
    def PropPyr1D(self):
        FO = FourierOptics.FourierOptics(pyrparams)
        diam0 = self.params['beam_diameter']
        print("initial x.shape= " + str(self.x_Start.shape))

        # Fresnel prop to lens #1
        z = self.params['D_e_2_l1']
        [f, df_dparam, x] = FO.ConvFresnel1D(self.field_Start1D, self.x_Start,1.2*diam0, z, return_derivs=True)
        self.grads['D_e_2_l1'] = df_dparam

        print("after prop to lens1 x.shape= " + str(x.shape))
        plt.figure(); plt.plot(x, np.real(f*np.conj(f))); 
        plt.title('Intensity impinging on Lens #1'); plt.xlabel("distance ($\mu$m)")

        # apply lens #1
        foc = self.params['f1']
        for key in self.grads:  # update gradients
            self.grads[key], _ = FO.ApplyThinLens1D(self.grads[key], x, 0, foc, return_derivs=False)
        f, df_dparam, x = FO.ApplyThinLens1D(f, x, 0, foc, return_derivs=True) # update field
        self.grads['l1_center'] = df_dparam

        print("post lens1 x.shape= " + str(x.shape))

        # Fresnel prop to focus
        lambda_over_D =  self.params['wavelength']*foc/diam0  
        diam1 = self.params['apex_diam']*lambda_over_D
        for key in self.grads:  # propagate gradients
            self.grads[key], _ = FO.ConvFresnel1D(self.grads[key], x, diam1, foc, return_derivs=False)
        f, df_dparam, x = FO.ConvFresnel1D(f, x, diam1, foc, return_derivs=True) # prop field
        self.Focal_field = f*1.0 # ensure deep copy
        self.grads['D_l1_2_apex'] = df_dparam

        print("after prop to focus x.shape= " + str(x.shape))
        plt.figure(); plt.plot(x, np.real(f*np.conj(f)));
        plt.title("Intensity at focus (pyramid apex)"); plt.xlabel("distance ($\mu$m)")

        # Apply phase ramp caused by pyramid
        for key in self.grads:
            self.grads[key], _ = FO.ApplyPyramidPhaseRamp1D(self.grads[key], x, 0., return_derivs=False, no_apex=False)
        f, df_dparam, x = FO.ApplyPyramidPhaseRamp1D(f, x, 0, return_derivs=True, no_apex=False)
        self.grads['pyramid_center'] = df_dparam

        print("after pyramid x.shape= " + str(x.shape))

        # Fresnel prop to lens #2
        z = self.params['D_foc_2_l2']
        diam2 = self.params['diam_lens2']
        for key in self.grads:  # propagate gradients
            self.grads[key], _ = FO.ConvFresnel1D(self.grads[key], x, diam2, z, return_derivs=False)
        f, df_dparam, x = FO.ConvFresnel1D(f, x, diam2, z, return_derivs=True)
        self.grads['D_foc_2_l2'] = df_dparam

        plt.figure(); plt.plot(x, np.real(f*np.conj(f)));
        plt.title('Intensity impinging on lens2'); plt.xlabel("distance ($\mu$m)")
        print("after prop to lens2 x.shape= " + str(x.shape))

        #apply lens #2
        foc = self.params['f2']
        for key in self.grads:  # update gradients
            self.grads[key], _ = FO.ApplyThinLens1D(self.grads[key], x, 0, foc, return_derivs=False)
        f, df_dparam, x = FO.ApplyThinLens1D(f, x, 0, foc, return_derivs=True) # update field
        self.grads['l2_center'] = df_dparam

        print("post lens2 x.shape= " + str(x.shape))

        # Fresnel prop to detector
        z = self.params['D_l2_2_detector']
        diam2 = self.params['detector_width']

        if False:  #  compare full result to maintain_dx results
            # get full result
            fc, xc = FO.ConvFresnel1D(f, x, diam2, z, return_derivs=False, maintain_dx=False)
            plt.figure(); plt.plot(xc, np.real(fc*np.conj(fc))); plt.title('Intensity at detector, full res'); plt.xlabel("distance ($\mu$m)")
            print("at detector xc.shape= " + str(xc.shape))
            # get maintain_dx result
            fc, xc = FO.ConvFresnel1D(f, x, diam2, z, return_derivs=False, maintain_dx=True)
            plt.figure(); plt.plot(xc, np.real(fc*np.conj(fc))); plt.title('Intensity at detector, maintain_dx'); plt.xlabel("distance ($\mu$m)")
            print("at detector xc.shape= " + str(xc.shape))

        # maintain_dx=True worked well!
        for key in self.grads:  # propagate gradients
            self.grads[key], _ = FO.ConvFresnel1D(self.grads[key], x, diam2, z, return_derivs=False, maintain_dx=True)
        f, df_dparam, x = FO.ConvFresnel1D(f, x, diam2, z, return_derivs=True, maintain_dx=True)
        self.grads['D_l2_2_detector'] = df_dparam

        print("at detector x.shape= " + str(x.shape))
        plt.figure(); plt.plot(x, np.real(f*np.conj(f))); plt.title('Intensity at detector'); plt.xlabel("distance ($\mu$m)")

        return