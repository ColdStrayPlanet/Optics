#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:02:47 2024

@author: Richard Frazin
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset


if not torch.cuda.is_available():
   print("Read the Sign.  No GPU, No Service!")
   assert False
device = torch.device('cuda')

in_channels = 2; out_channels = 2  # the images are complex valued.

class UNetWithSkip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Encoder with modified channels and larger kernels
        self.enc1 = self.conv_block(in_channels, 16, kernel_size=5)
        self.enc2 = self.conv_block(16, 32, kernel_size=5)
        self.enc3 = self.conv_block(32, 64, kernel_size=3)
        self.enc4 = self.conv_block(64, 128, kernel_size=3)

        # Bottleneck
        self.bottleneck = self.conv_block(128, 64, kernel_size=3)

        # Decoder with skip connections
        self.dec3 = self.conv_block(64, 64, kernel_size=3)  # Skip connection added
        self.dec2 = self.conv_block(192, 32, kernel_size=3)  # Skip connection added
        self.dec1 = self.conv_block(96, 16, kernel_size=3)  # Skip connection added

        # Output layer
        self.out_conv = nn.Conv2d(48, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, kernel_size=3):
        """Basic convolution block with optional kernel size adjustment"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        debug0 = False
        debug1 = False
        if debug0:
            print(f"Input shape: {x.shape}")

        # Encoder
        enc1 = self.enc1(x)
        if debug0:
            print(f"After enc1: {enc1.shape}")
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        if debug0:
            print(f"After enc2: {enc2.shape}")
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        if debug0:
            print(f"After enc3: {enc3.shape}")
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        if debug1:
            print(f"After enc4: {enc4.shape}")

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        if debug1:
            print(f"After bottleneck: {bottleneck.shape}")

        # Decoder with skip connections
        dec3 = self.dec3(F.interpolate(bottleneck, size=enc4.shape[2:], mode='bilinear', align_corners=False))
        dec3 = torch.cat([dec3, enc4], dim=1)  # Concatenate skip connection from enc4
        if debug1:
            print(f"After dec3 (upscaled and concatenated with enc4): {dec3.shape}")

        dec2 = self.dec2(F.interpolate(dec3, size=enc3.shape[2:], mode='bilinear', align_corners=False))
        dec2 = torch.cat([dec2, enc3], dim=1)  # Concatenate skip connection from enc3
        if debug1:
            print(f"After dec2 (upscaled and concatenated with enc3): {dec2.shape}")
        dec2_upscaled = F.interpolate(dec2, size=(31, 31), mode='bilinear', align_corners=False)
        if debug1:
            print(f"After upsampling dec2 to (31, 31): {dec2_upscaled.shape}")
        # Upsample enc2 to (31, 31) to match the size of dec1
        enc2_upscaled = F.interpolate(enc2, size=(31, 31), mode='bilinear', align_corners=False)
        if debug1:
            print(f"After upsampling enc2 to (31, 31): {enc2_upscaled.shape}")

        dec1 = self.dec1(dec2_upscaled)
        dec1 = torch.cat([dec1, enc2_upscaled], dim=1)  # Concatenate skip connection from enc2 (upsampled)
        if debug1:
            print(f"After dec1 (upscaled and concatenated with enc2): {dec1.shape}")

        #dec1 = self.dec1(F.interpolate(dec2, size=enc2.shape[2:], mode='bilinear', align_corners=False))
        #dec1 = torch.cat([dec1, enc2], dim=1)  # Concatenate skip connection from enc2
        #if debug1:
        #    print(f"After dec1 (upscaled and concatenated with enc2): {dec1.shape}")

        # Output
        out = self.out_conv(dec1)
        if debug1:
            print(f"Output shape: {out.shape}")

        return out


class ComplexImageDataSet(torch.utils.data.Dataset):
    def __init__(self, input_images, target_images, transform=None):
        """
        Args:
            input_images (array-like or tensor): Complex-valued input images.
            target_images (array-like or tensor): Complex-valued target images.
            transform (callable, optional): Optional transform to be applied to the input and target.
        """
        self.input_images = input_images  # List or array of complex-valued input images
        self.target_images = target_images  # List or array of complex-valued target images
        self.transform = transform  # Transform function to be applied to each image (input & target)

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.input_images)

    def __getitem__(self, idx):
       """
       Args:
           idx (int): Index of the sample to retrieve.

       Returns:
           input_img (tensor): A tensor of shape (2, H, W) where the first channel is the real part,
                                and the second channel is the imaginary part.
           target_img (tensor): Same as input_img but for the target image.
       """
       debug = False
       input_img = self.input_images[idx]  # Get the complex-valued input image
       target_img = self.target_images[idx]  # Get the complex-valued target image

       # Check the shape of the images before stacking
       if debug:
           print("Input image shape before stacking:", input_img.shape)
           print("Target image shape before stacking:", target_img.shape)

       # Convert the complex-valued images into two-channel tensors (real and imaginary)
       input_real = torch.tensor(input_img[0,:,:])
       input_imag = torch.tensor(input_img[1,:,:])
       target_real = torch.tensor(target_img[0,:,:])
       target_imag = torch.tensor(target_img[1,:,:])

       input_img = torch.stack([input_real, input_imag], dim=0)  # Correctly create a tensor with shape [2, H, W]
       target_img = torch.stack([target_real, target_imag], dim=0)  # Same for target

       # Check shapes after concatenation
       if debug:
           print("Input shape after torch.stack:", input_img.shape)
           print("Target shape after torch.stack:", target_img.shape)

       # Apply any transformations (if provided)
       if self.transform:
           input_img, target_img = self.transform(input_img, target_img)

       return input_img, target_img


#  Helper Functions for Saving and Loading Checkpoints and Models
def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save the model and optimizer state dict to a file.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at epoch {epoch}, loss {loss:.4f} to {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """
    Load the model and optimizer state dict from a checkpoint file.
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {filepath}, epoch {epoch}, loss {loss:.4f}")
    return epoch, loss

def save_model_for_inference(model, filepath):
    """
    Save the trained model for inference.
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved for inference to {filepath}")
    return None

def load_model_for_inference(model, filepath):
    """
    Load a trained model for inference.
    """
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Model loaded for inference from {filepath}")
    return None

# 5. Training and Checkpoint Saving Logic
def train_model(model, train_loader, optimizer, criterion, epochs, checkpoint_dir, checkpoint_freq):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device).float(), targets.to(device).float()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Save checkpoint at specified frequency
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_filepath = f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, checkpoint_filepath)
    return None

# 6. Visualization Function
# input_image and target_image are numpy arrays of shape (2, m , m)
def visualize_results(model, input_image, target_image):


# %%
    shape = input_image.shape;  ss1 = shape[0]; ss2=shape[1]; ss3=shape[2]
    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(input_image.reshape((1,ss1,ss2,ss3)), dtype=torch.float32).to(device))

    output_image = output[0].cpu().numpy()

    # Plot real and imaginary parts for input, target, and output images
    fig, axs = plt.subplots(2, 3, figsize=(11, 7))

    input_re = axs[0,0].imshow(input_image[0], cmap='seismic');
    axs[0,0].set_title('Input Image (Real part)')

    target_re = axs[0,1].imshow(target_image[0], cmap='seismic')
    axs[0,1].set_title('Target Image (Real part)')

    output_re = axs[0,2].imshow(output_image[0], cmap='seismic')
    axs[0,2].set_title('Predicted Image (Real part)')

    input_im = axs[1,0].imshow(input_image[1], cmap='seismic');
    axs[1,0].set_title('Input Image (Imaginary part)')

    target_im = axs[1,1].imshow(target_image[1], cmap='seismic')
    axs[1,1].set_title('Target Image (Imaginary part)')

    output_im = axs[1,2].imshow(output_image[1], cmap='seismic')
    axs[1,2].set_title('Predicted Image (Imaginary part)')

    # Barra de color para la primera fila: input_image tiene su propia escala
    fig.colorbar(input_re, ax=axs[0, 0], orientation='vertical', fraction=0.02, pad=0.04)
    # Barra de color compartida para target_image y output_image
    fig.colorbar(target_re, ax=[axs[0, 1], axs[0, 2]], orientation='vertical', fraction=0.02, pad=0.04)
    # Barra de color para la segunda fila: input_image (imaginario)
    fig.colorbar(input_im, ax=axs[1, 0], orientation='vertical', fraction=0.02, pad=0.04)
    # Barra de color compartida para target_image (imaginario) y output_image (imaginario)
    fig.colorbar(target_im, ax=[axs[1, 1], axs[1, 2]], orientation='vertical', fraction=0.02, pad=0.04)
    # Ajustar el diseño para evitar superposición
    #plt.tight_layout()





# %%

def conv2d_output_size(input_size, kernel_size, stride=1, padding=0):
    """
    Calculer la taille de la sortie d'une couche de convolution 2D (torch.nn.Conv2d).

    Args:
        input_size (int): La taille de l'entrée (hauteur ou largeur).
        kernel_size (int): La taille du noyau de convolution (int).
        stride (int, optional): Le stride de la convolution (par défaut 1).
        padding (int, optional): Le padding ajouté autour de l'image (par défaut 0).

    Returns:
        int: La taille de la sortie après la convolution.
    """

    # Appliquer la formule de calcul de la taille de sortie
    output_size = 1 + (input_size - kernel_size + 2 * padding) // stride
    return output_size

































class UNetNoSkip(nn.Module):
    def __init__(self, in_channels, out_channels):
       super().__init__()

       # Encoder with modified channels and larger kernels
       self.enc1 = self.conv_block(in_channels, 16, kernel_size=5)
       self.enc2 = self.conv_block(16, 32, kernel_size=5)
       self.enc3 = self.conv_block(32, 64, kernel_size=3)
       self.enc4 = self.conv_block(64, 128, kernel_size=3)

       # Bottleneck
       self.bottleneck = self.conv_block(128, 64, kernel_size=3)

       # Decoder with reduced channels (no skip connections)
       self.dec3 = self.conv_block(64, 64, kernel_size=3)  # Pas de concaténation
       self.dec2 = self.conv_block(64, 32, kernel_size=3)
       self.dec1 = self.conv_block(32, 16, kernel_size=3)

       # Output layer
       self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, kernel_size=3):
       """Basic convolution block with optional kernel size adjustment"""
       return nn.Sequential(
           nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
           nn.ReLU(inplace=True),
           nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
           nn.ReLU(inplace=True)
       )

    def forward(self, x):
       debug = False
       if debug:
           print(f"Input shape: {x.shape}")

       # Encoder
       enc1 = self.enc1(x)
       if debug:
           print(f"After enc1: {enc1.shape}")
       enc2 = self.enc2(F.max_pool2d(enc1, 2))
       if debug:
           print(f"After enc2: {enc2.shape}")
       enc3 = self.enc3(F.max_pool2d(enc2, 2))
       if debug:
           print(f"After enc3: {enc3.shape}")
       enc4 = self.enc4(F.max_pool2d(enc3, 2))
       if debug:
           print(f"After enc4: {enc4.shape}")

       # Bottleneck
       bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
       if debug:
           print(f"After bottleneck: {bottleneck.shape}")

       # Decoder (sans skip connections)
       dec3 = self.dec3(F.interpolate(bottleneck, size=(7,7), mode='bilinear', align_corners=False))
       if debug:
           print(f"After dec3 (upscaled): {dec3.shape}")
       dec2 = self.dec2(F.interpolate(dec3, size=(15,15), mode='bilinear', align_corners=False))
       if debug:
           print(f"After dec2 (upscaled): {dec2.shape}")
       dec1 = self.dec1(F.interpolate(dec2, size=(31,31), mode='bilinear', align_corners=False))
       if debug:
           print(f"After dec1 (upscaled): {dec1.shape}")

       # Output
       out = self.out_conv(dec1)
       if debug:
           print(f"Output shape: {out.shape}")

       return out
