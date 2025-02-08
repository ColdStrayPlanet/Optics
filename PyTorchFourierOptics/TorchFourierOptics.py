#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:00:02 2025
@author: Richard Frazin

Differentiable Fourier Optics with PyTorch

"""
import numpy as np
import torch
import torch.fft

#lengths are  in microns.  angles are in degrees
example_parameters = {'wavelength': 0.9, 'max_chirp_step_deg':45.0}


################## start TorchFourerOptics Class ######################
class TorchFourierOptics:
    def __init__(self, params):
        self.params = example_parameters

    def GetDxAndLength(self, x):
        nx = x.shape[0]
        dx = x[1] - x[0]
        length = (x[-1] - x[0]) * (1 + 1 / (nx - 1))
        assert length > 0
        assert dx > 0
        return dx, length


    #Resample the tensor g on a new grid.  This relies on the robust routine torch.nn.functional.grid_sample.
    # This routine has input arguments: (input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)
    #   'input' is tensor that is to be interpolated onto a new grid
    #        It  has dimentions (batch_size, number of channels, height, width) for 2D arrays - there is a depth dimension for 3D
    #    'grid' is the set of new coordinates, normalized to the [[-1,1],[-1,1]] range.  It has a shape
    #        (batch_size , output height, output width,2) where the final dimension corresponds to the
    #     Since it assumed that the input coordinate correspond to the [[-1,1],[-1,1]] grid, x_new needs to scaled as:
    #         x_torch =  ( 2*x_new - x.max() - x.min() )/(x.max() - x.min())  where x_torch corresponds a 1D coordinate in the grid array
    #         x and y coords of the new samples
    #  See the pytorch  docs for more detail

    def ResampleField2D(self, g, x, x_new):
        if isinstance(x,np.ndarray):
           x = torch.from_numpy(x)
        if isinstance(x_new, np.ndarray):
           x_new = torch.from_numpy(x_new)

        x_torch = (2*x_new - x.max() -x.min())/(x.max()-x.min())
        grid_x, grid_y = torch.meshgrid(x_torch, x_torch, indexing='xy')
        grid = torch.stack((grid_x, grid_y), dim=-1)  # if grid_x and grid_y are (N,N) this is (N,N,2)
        gnew = torch.nn.functional.grid_sample(g.unsqueeze(0),grid.unsqueeze(0),
                                               mode='bicubic',padding_mode='zeros',align_corners=False )

        return  (gnew[0], x_new)

    #2D Fresenel prop using convolution in the spatial domain
    #NOTE: This routine zeros the Fresnel kernel at distances beyond which the
    #    chirp is not spatially resolved.  See the 'set_dx' argument below
    # g - matrix of complex-valued field in the inital plane
    #       output field should be differentiable with respect to 'g'
    # x - vector of 1D coordinates in initial plane, corresponding to bin center locaitons
    #        it is assumed that the x and y directions have the same sampling.
    # length_out - length (one side of the square) of region to be calculated in output plane
    # z - propagation distance
    # index_of_refaction - isotropic index of refraction of medium
    # set_dx = [True | False | dx_new (same units as x)]
    #     True  - forces full sampling of chirp according to self.params['max_chirp_step_deg']
    #             Note: this can lead to unacceptably large arrays.
    #     False -  the grid spacing of the output plane is the same as that of the input plane
    #     dx_new - grid spacing of the output plane



    def ConvFresnel2D(self, g, x, length_out, z, index_of_refraction=1, set_dx=True):
        if g.shape[1] != x.shape[0]:
            raise ValueError("Input field and grid must have the same sampling.")
        if g.shape[1] != g.shape[2]:
            raise ValueError("Input field array must be square.")

        lam = self.params['wavelength'] / index_of_refraction
        dx, length_input = self.GetDxAndLength(x)
        dx_chirp = (self.params['max_chirp_step_deg'] / 180) * lam * z / (length_input + length_out)

        if isinstance(set_dx, bool):
            if set_dx == False:
                dx_new = x[1] - x[0]
            else:  # set_x is True. Use chirp sampling criterion
                if dx < dx_chirp:
                    dx_new = dx_chirp
        else:  # set_dx is not a bool. Take dx_new to be value of set_dx
            if not isinstance(set_dx, float):
                raise Exception("ConvFresnel2D: set_dx must be a bool or a float.")
            if set_dx <= 0:
                raise Exception("ConvFresnel2D: numerical value of set_dx must be > 0.")
            dx_new = set_dx

        # Resample input field on a grid with spacing dx_new
        nx_new = length_input / dx_new
        x_new = np.linspace(-length_input / 2 + dx_new / 2, length_input / 2 - dx_new / 2, nx_new)

        g, x = self.ResampleField2D(g, x, x_new)
        dx = x[1] - x[0]

        ns = int(torch.round((length_input + length_out) / dx))
        s = torch.linspace(-length_input / 2 - length_out / 2 + dx / 2, length_input / 2 + length_out / 2 - dx / 2, ns)
        sx, sy = torch.meshgrid(s, s, indexing='xy')
        kern = torch.exp(1j * torch.pi * (sx**2 + sy**2) / (lam * z))

        # Apply quadratic cutoff
        r = torch.sqrt(sx**2 + sy**2)
        s_max = lam * z * self.params['max_chirp_step_deg'] / (360 * dx)
        d_small = s_max / 20  # Small distance beyond which the kernel drops to zero

        # Apply quadratic cutoff
        cutoff = torch.ones_like(r)
        # No cutoff for r < s_max
        cutoff[r > s_max] = 1 - (r[r > s_max] - s_max) ** 2 / (d_small ** 2)
        cutoff[cutoff < 0] = 0  # Ensure that the cutoff doesn't go negative
        kern *= cutoff

        h_real = torch.fft.ifft2(torch.fft.fft2(g[0]) * torch.fft.fft2(kern.real)
                                 - torch.fft.fft2(g[1]) * torch.fft.fft2(kern.imag)).real
        h_imag = torch.fft.ifft2(torch.fft.fft2(g[1]) * torch.fft.fft2(kern.real)
                                 + torch.fft.fft2(g[0]) * torch.fft.fft2(kern.imag)).real
        h = torch.stack((h_real, h_imag), dim=0)

        return (h / (lam * z), s)


    def _ConvFresnel2D(self, g, x, length_out, z, index_of_refraction=1, set_dx=True):
        if g.shape[1] != x.shape[0]:
            raise ValueError("Input field and grid must have the same sampling.")
        if g.shape[1] != g.shape[2]:
            raise ValueError("Input field array must be square.")

        lam = self.params['wavelength'] / index_of_refraction
        dx, length_input = self.GetDxAndLength(x)
        dx_chirp = (self.params['max_chirp_step_deg']/180) * lam * z / (length_input + length_out)

        if isinstance(set_dx, bool):
             if set_dx == False:
                dx_new = x[1] - x[0]
             else:  # set_x is True.  Use chirp sampling criterion
                 if dx < dx_chirp:
                     dx_new = dx_chirp
        else:  # set_dx is not a bool. Take dx_new to be value of set_dx
             if not isinstance(set_dx, float):
                 raise Exception("ConvFresnel2D: set_dx must be a bool or a float.")
             if set_dx <= 0:
                 raise Exception("ConvFresnel2D: numerical value of set_dx must be > 0.")
             dx_new = set_dx

        if 64*((length_input + length_out)/dx_new)**2  > 1.5e10:  # 64 bytes per complex number
            print("dx_new=",dx_new,"length_input=",length_input)
            raise Exception("New field size exceeds roughly 20GB, aborting.")

        #resample input field on a grid with spacing dx_new
        nx_new = length_input/dx_new
        x_new = np.linspace(- length_input/2 + dx_new/2, length_input/2 - dx_new/2, nx_new)


        g, x = self.ResampleField2D(g, x, x_new)
        dx = x[1] - x[0]

        ns = int(torch.round((length_input + length_out) / dx))
        s = torch.linspace(-length_input/2 - length_out/2 + dx/2, length_input/2 + length_out/2 - dx/2, ns)
        sx, sy = torch.meshgrid(s, s, indexing='xy')
        kern = torch.exp(1j * torch.pi * (sx**2 + sy**2) / (lam * z))

        if dx > dx_chirp:
            s_max = lam * z * self.params['max_chirp_step_deg'] / (360 * dx)
            kern[torch.sqrt(sx**2 + sy**2) > s_max] = 0

        h_real = torch.fft.ifft2(torch.fft.fft2(g[0])*torch.fft.fft2(kern.real)
                               - torch.fft.fft2(g[1])*torch.fft.fft2(kern.imag)).real
        h_imag = torch.fft.ifft2(torch.fft.fft2(g[1])*torch.fft.fft2(kern.real)
                               + torch.fft.fft2(g[0])*torch.fft.fft2(kern.imag)).real
        h = torch.stack((h_real, h_imag), dim=0)


        return (h/(lam*z), s)



################## end TorchFourerOptics Class ######################

# Convertit un tableau NumPy complexe (N, N) en un tenseur PyTorch de forme (2, N, N)
def numpy_complex_to_torch(real_imag_array: np.ndarray) -> torch.Tensor:
    if not np.iscomplexobj(real_imag_array):
        raise ValueError("L'entrée doit être un tableau NumPy de nombres complexes.")

    real_part = torch.from_numpy(real_imag_array.real).float()
    imag_part = torch.from_numpy(real_imag_array.imag).float()

    return torch.stack((real_part, imag_part), dim=0)

#Convertit un tenseur PyTorch (2, N, N) en un tableau NumPy complexe (N, N).
def torch_to_numpy_complex(torch_tensor: torch.Tensor) -> np.ndarray:
    if torch_tensor.shape[0] != 2:
        raise ValueError("Le tenseur d'entrée doit avoir la forme (2, N, N).")

    real_part = torch_tensor[0].cpu().numpy()
    imag_part = torch_tensor[1].cpu().numpy()

    return real_part + 1j * imag_part
