#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:00:02 2025
@author: Richard Frazin

Differentiable Fourier Optics with PyTorch

"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.fft

#lengths are  in microns.  angles are in degrees
example_parameters = {'wavelength': 0.9, 'max_chirp_step_deg':30.0}


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
       if isinstance(x, np.ndarray):
           x = torch.from_numpy(x).float()
       if isinstance(x_new, np.ndarray):
           x_new = torch.from_numpy(x_new).float()

       x_torch = (2 * x_new - x.max() - x.min()) / (x.max() - x.min())
       grid_x, grid_y = torch.meshgrid(x_torch, x_torch, indexing='xy')
       grid = torch.stack((grid_x, grid_y), dim=-1)  # (len(x_new), len(x_new), 2)
       # Ajustar las dimensiones de `g` para `grid_sample`
       g = g.unsqueeze(0)  # Añadir dimensión de lote: (1, 2, N, N)
       # Unsqueeze en grid si es necesario para asegurar que batch_size es 1
       grid = grid.unsqueeze(0)  # Añadir dimensión de lote: (1, len(x_new), len(x_new), 2)
       gnew = torch.nn.functional.grid_sample(g, grid, mode='bicubic', padding_mode='zeros', align_corners=False)
       return gnew[0], x_new

    """
    Simula una óptica perfecta de transformada de Fourier usando la propagación FFT.

    Args:
    - g (torch.Tensor): Campo de entrada con dos canales (real e imaginario) de tamaño (2, N, N).
    - x (torch.Tensor): Coordenadas 1D del grid en el plano de entrada, en micrómetros, de tamaño (N,).
    - x_out (torch.Tensor): Coordenadas 1D del grid en el plano de salida, en micrómetros.
    - focal_length (float): Distancia focal del sistema óptico en micrómetros.

    Let L = max(x) - min(x), and M be the number of points in the array, padded to have length M+P.
      Define L' = L((M+P)/M). The frequency corresponding to point m in the FFT is m/L' = (m/L)(M/(M+P))
      After undergoing an optical Fourier transform via a lens with focal length f, the spatial frequency
      corresponding to a distance u from the center is u/(f*\lambda), where \lambda is the wavelength. Thus,
      the FFT frequency spacing is (1/L)(M/(M+P)) and the physical frequency spacing is (\delta u)/(f * \lambda).


    Returns:
    - torch.Tensor: Campo en el plano de salida, transformado por la óptica FT.
    """
    def FFT_prop(self, g, x, x_out, focal_length):
      M = len(x)
      assert g[1].shape == (M,M)
      wavelength = self.params['wavelength']
      tru_freq_spacing = (x_out[1]-x_out[0])*(focal_length*wavelength) # physical spatial frequency spacing
      #figure out the amount of padding needed for the FFT to capture the physical spatial frequencies
      MP = 32  # MP is the total number of points in the padded array (1D)
      while True:
         P = MP - M  #  P is the number of zero-pad values
         if P < 0: continue
         FFT_spacing = (M/MP)/(x.max() - x.min())
         if tru_freq_spacing/FFT_spacing >=2:
            break
         else: MP *= 2

      g_pad = torch.zeros((2,MP,MP))  # create padded array
      k1 = MP//2 - M//2;  k2 = k1 + M
      g_pad[0, k1:k2, k1:k2] = g[0]
      g_pad[1, k1:k2, k1:k2] = g[1]
      g_pad = torch.complex(g_pad[0],g_pad[1])
      f_complex = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(g_pad)))
      ff = torch.stack(torch.real(f_complex), torch.imag(f_complex) ,dim=0)

      x_pad = torch.linspace(x.min(),x.max(),MP)
      f = self.ResampleField2D(ff, x_pad, x_out)
      return f


    #2D Fresenel prop using convolution in the spatial domain
    #NOTE: This routine zeros the Fresnel kernel at distances beyond which the
    #    chirp is not spatially resolved.  See the 'set_dx' argument below
    # g - matrix of complex-valued field in the inital plane
    #    g has dimensions (2,N,N) where g[0] is the real part and g[1] is the imag part.
    #    the output field should be differentiable with respect to 'g'
    # x - vector of 1D coordinates in initial plane (where g is defined), corresponding to bin center locaitons
    #        it is assumed that the x and y directions have the same sampling.
    # length_out - length (one side of the square) of region to be calculated in output plane
    # z - propagation distance
    # index_of_refaction - isotropic index of refraction of medium
    # set_dx = [True | False | dx_new (same units as x)]
    #     True  - forces full sampling of chirp according to self.params['max_chirp_step_deg']
    #             Note: this can lead to unacceptably large arrays.
    #     False -  the grid spacing of the output plane is the same as that of the input plane
    #     dx_new - grid spacing of the output plane

    def ConvFresnel2D(self, g, x, length_out, z,  set_dx=True, index_of_refraction=1):
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
                    dx_new = dx_chirp
        else:  # set_dx is not a bool. Take dx_new to be value of set_dx
            if not isinstance(set_dx, float):
                raise Exception("ConvFresnel2D: set_dx must be a bool or a float.")
            if set_dx <= 0:
                raise Exception("ConvFresnel2D: numerical value of set_dx must be > 0.")
            dx_new = set_dx

        # Resample input field on a grid with spacing dx_new
        nx_new = int(length_input / dx_new)
        x_new = np.linspace(-length_input/2 + dx_new/2, length_input/2 - dx_new/2, nx_new)
        g, x = self.ResampleField2D(g, x, x_new)
        dx = x[1] - x[0]

        ns = int(torch.round((length_input + length_out) / dx))
        s = torch.linspace(-length_input / 2 - length_out / 2 + dx / 2, length_input / 2 + length_out / 2 - dx / 2, ns)
        sx, sy = torch.meshgrid(s, s, indexing='xy')
        kern = torch.exp(1j * torch.pi * (sx**2 + sy**2) / (lam*z))
        kern /= torch.sum(torch.abs(kern))  #apply normalization
        kernphase_deriv = lambda r: 2*torch.pi*r/(lam*z)  # radial gradient of the phase of the chirp kernel

        # Apply quadratic cutoff
        r = torch.sqrt(sx**2 + sy**2)
        s_max = lam * z * self.params['max_chirp_step_deg'] / (360 * dx)
        d_small = 1./kernphase_deriv(s_max)  # Small distance beyond which the kernel drops to zero
        cutoff = torch.ones_like(r)
        # No cutoff for r < s_max
        cutoff[r > s_max] = 1 - (r[r > s_max] - s_max) ** 2 / (d_small ** 2)
        cutoff[cutoff < 0] = 0  # Ensure that the cutoff doesn't go negative
        kern *= cutoff

        # Aplicar padding a `g` para igualar el tamaño de `kern`.
#        pad_size = [((kern.size(0) - g.size(1)) + 1) // 2, ((kern.size(1) - g.size(2)) + 1) // 2]
#        g_padded_real = torch.nn.functional.pad(g[0], pad=pad_size)
#        g_padded_imag = torch.nn.functional.pad(g[1], pad=pad_size)


        # Aplicar padding a `g` para igualar el tamaño de `kern`.
        pad_y = (kern.size(0) - g.size(1)) // 2
        pad_x = (kern.size(1) - g.size(2)) // 2
        # El padding se hace simétrico
        g_padded_real = torch.nn.functional.pad(g[0], (pad_x, pad_x, pad_y, pad_y))
        g_padded_imag = torch.nn.functional.pad(g[1], (pad_x, pad_x, pad_y, pad_y))

        # Separar la parte real e imaginaria del kernel
        kern_real_shifted = torch.fft.fftshift(kern.real)
        kern_imag_shifted = torch.fft.fftshift(kern.imag)

        #print("Tamaño de g (padded real):", g_padded_real.size())
        #print("Tamaño de g (padded imag):", g_padded_imag.size())
        #print("Tamaño de kern (real):", kern_real_shifted.size())
        #print("Tamaño de kern (imag):", kern_imag_shifted.size())
        #return None


        # Aplicar fftshift en la señal g paddeada
        g_real_shifted = torch.fft.fftshift(g_padded_real)
        g_imag_shifted = torch.fft.fftshift(g_padded_imag)

        # Realizar transformadas de Fourier en las partes reales e imaginarias
        G_real = torch.fft.fft2(g_real_shifted)
        G_imag = torch.fft.fft2(g_imag_shifted)
        K_real = torch.fft.fft2(kern_real_shifted)
        K_imag = torch.fft.fft2(kern_imag_shifted)

        # Calcular la convolución en el dominio de la frecuencia
        H_real_shifted = G_real * K_real - G_imag * K_imag
        H_imag_shifted = G_imag * K_real + G_real * K_imag

        # Aplicar la transformada inversa de Fourier
        h_real = torch.fft.ifftshift(torch.fft.ifft2(H_real_shifted).real)
        h_imag = torch.fft.ifftshift(torch.fft.ifft2(H_imag_shifted).real)

        # Reconstruir la señal resultante compleja
        h = torch.stack((h_real, h_imag), dim=0)

        return (h / (lam * z), s)

    #Apply the thin lens phase transformation
    #g - two channel input field (each channel is 2D)
    #x - 1D coordinates: len(x) = g.shape[0]=g.shape[1]
    # set_dx = [True | False | dx_new (same units as x)]
    #     True  - forces full sampling of quadratic phase according to self.params['max_chirp_step_deg']
    #     False -  the grid spacing of the output plane is the same as that of the input plane, but it
    #                will not execute if the quadaratic phase is not resolved as above
    #     dx_new - grid spacing of the output plane, but it
    #                will not execute if the quadratic phase is not resolved as above
    #center - two numbers center_x and center_y indicating the center of the
    #     lens with respect to 'x'.  Can be  a tuple, array or list
    #focal_length - same units as x

    def ApplyThinLens2D(self, g, x, set_dx, center, focal_length):
          if g.size(1) != x.size(0):
              raise ValueError("El campo de entrada y las coordenadas deben tener el mismo tamaño.")
          wavelength = self.params['wavelength']
          philim = self.params['max_chirp_step_deg']*torch.pi/180
          max_step = philim*wavelength*focal_length/(2*torch.pi*x.max())

          if isinstance(set_dx, bool):
            if set_dx == False:
                dx_new = x[1] - x[0]
                if dx_new > max_step:
                   print("ApplyThinLens2D: You chose set_dx to be False, but the current sampling is too coarse.")
                   print("max dx =",max_step,"current dx =", dx_new)
                   raise Exception()
            else:  # set_x is True. Use chirp sampling criterion
                    dx_new = max_step
          else:  # set_dx is not a bool. Take dx_new to be value of set_dx
            if not isinstance(set_dx, float):
                raise Exception("ApplyThinLens2D: set_dx must be a bool or a float.")
            if set_dx <= 0:
                raise Exception("ApplyThinsLens2D: numerical value of set_dx must be > 0.")
            dx_new = set_dx
            if dx_new > max_step:
                   print("ApplyThinLens2D: You chose set_dx to be,", set_dx ," but that is too coarse.  max dx =",max_step)
                   raise Exception()

      apply resampling


          center_x, center_y = center

          # Crear mallas de coordenadas
          X, Y = torch.meshgrid(x - center_x, x - center_y, indexing='xy')

          # Calcular la fase cuadrática aplicada por el lente delgado
          phase_factor = torch.exp(-1j * torch.pi * (X**2 + Y**2) / (wavelength * focal_length))

          # Separar canales real e imaginario de g
          g_real = g[0]
          g_imag = g[1]

          # Crear el campo complejo a partir de sus componentes reales e imaginarias
          g_complex = torch.complex(g_real, g_imag)

          # Aplicar la fase del lente
          g_transformed = g_complex * phase_factor

          # Dividir el campo resultado en sus partes real e imaginaria
          g_transformed_real = g_transformed.real
          g_transformed_imag = g_transformed.imag

          # Reconstruir la señal resultante en forma de tensor de dos canales
          h = torch.stack((g_transformed_real, g_transformed_imag), dim=0)

          return h

    """
        Aplica una combinación de un stop y una apertura con amortiguamiento a un campo óptico 2D.

        Args:
        - g (torch.Tensor): Campo de entrada con dos canales (real e imaginario) de tamaño (2, N, N).
        - x (torch.Tensor): Coordenadas 1D del grid en micrómetros, de tamaño (N,).
        - center (tuple or list): Coordenadas (x, y) del centro del stop y la apertura.
        - r_stop (float): Radio del stop en micrómetros.
        - r_ap (float): Radio de la apertura en micrómetros, debe ser mayor que r_stop.
        - damping_length (float): Longitud de amortiguamiento para suavizar el paso de stop a apertura.
        - shape (str): Forma de la apertura ('circle' o 'square').

        Returns:
        - torch.Tensor: Campo truncado resultante con dos canales (real e imaginario).
    """
    def ApplyStopAndAperture(self, g, x, center, r_stop, r_ap, damping_length, shape='circle'):

       X, Y = torch.meshgrid(x - center[0], x - center[1], indexing='xy')

       # Calcular la distancia radial del centro
       if shape == 'circle':
           r = torch.sqrt(X**2 + Y**2)
       elif shape == 'square':
           r = torch.max(torch.abs(X), torch.abs(Y))
       else:
           raise ValueError("Shape must be 'circle' or 'square'.")

       # Crear el factor de máscara
       mask = torch.ones_like(r)

       # Apagar la luz en el interior del stop
       mask[r <= r_stop] = 0

       # Aplicar amortiguamiento cuadrático entre r_stop y r_ap
       transition_zone = (r > r_stop) & (r < r_ap)
       mask[transition_zone] = ((r[transition_zone] - r_stop) / (r_ap - r_stop)) ** 2

       # También asegúrate de que la máscara sea 0 para r > r_ap
       mask[r >= r_ap] = 0

       # Aplicar la máscara al campo
       g_real = g[0] * mask
       g_imag = g[1] * mask

       # Reconstruir el campo complejo modificado
       h = torch.stack((g_real, g_imag), dim=0)

       return h


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

def test_fresnel_propagation():
    # Configurar el campo de entrada
    N = 256  # Resolución del campo de entrada
    L = 120  # Longitud de lado del campo en micrómetros
    x = np.linspace(-L/2, L/2, N)  # Coordenadas en el plano de entrada - campo cuadrado
    X, Y = np.meshgrid(x, x)
    wavelength = 0.9  # Longitud de onda en micrómetros
    z = 1000  # Distancia de propagación en micrómetros

    # Crear un campo de entrada cuadrado uniforme (por simplicidad)
    input_field = np.ones((N, N), dtype=np.complex64)

    # Convertir el campo de entrada en un tensor de PyTorch complejo (2, N, N)
    g = numpy_complex_to_torch(input_field)

    # Instanciar la clase de óptica de Fourier
    params = {'wavelength': wavelength, 'max_chirp_step_deg': 30.0}
    optics = TorchFourierOptics(params)

    # Llamar a la función de propagación de Fresnel
    output_field, s = optics.ConvFresnel2D(g, torch.from_numpy(x).float(), L, z)

    # Convertir el resultado a un array NumPy complejo
    output_field_np = torch_to_numpy_complex(output_field)

    # Mostrar el campo de entrada y el campo de salida
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(input_field), extent=(x.min(), x.max(), x.min(), x.max()))
    plt.title('Campo de entrada')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(output_field_np), extent=(s[0].item(), s[-1].item(), s[0].item(), s[-1].item()))
    plt.title('Campo de salida (propagado)')
    plt.colorbar()

    plt.show()
