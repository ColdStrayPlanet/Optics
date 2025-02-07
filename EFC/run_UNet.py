#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:04:49 2024

@author: Richard Frazin rfrazin@umich.edu
"""


machine = "OfficeWindows"
if machine == "OfficeWindows":
  import os
  os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
This treats an error I'm encountering on my windows machine.  Specifically,
the call to GFP(), see below, leads to an Python kernel crash.  Below is from UMGPT:
The error you're encountering is related to the OpenMP runtime known as libiomp5md.dll.
This happens when multiple copies of OpenMP are initialized in the same process, which
can cause conflicts. This issue is common when using libraries that internally use
OpenMP, such as PyTorch and potentially other libraries you're importing.
"""

import torch
from torch.utils.data import DataLoader
from SampleGen import GenerateFieldPairs as GFP
import UNet4CrossField as UN

if not torch.cuda.is_available():
   print("Read the Sign.  No GPU, No Service!")
   assert False


(dom, cro, scales) = GFP(1000)
inputscale = scales[0];  targetscale = scales[1]
dom /= inputscale;  cro /= targetscale
input_images = dom; target_images = cro  # dominant and cross polarization fields


cids = UN.ComplexImageDataSet(input_images, target_images)
train_loader = DataLoader(cids, batch_size=10, shuffle=True)

# Hyperparameters and configuration
in_channels  = input_images.shape[ 1]  # Real and imaginary parts
out_channels = target_images.shape[1]  # Real and imaginary parts
learning_rate = 1e-4
epochs = 20
checkpoint_dir = './checkpoints'  # Directory to save checkpoints
checkpoint_freq = 5  # Save every 5 epochs
device = torch.device('cuda')
# %%

# Initialize the model and optimizer
#model = UN.UNet(in_channels=in_channels, out_channels=out_channels).to(device)
model = UN.UNetWithSkip(in_channels=in_channels, out_channels=out_channels).to(device)

optimizer = UN.optim.Adam(model.parameters(), lr=learning_rate)
criterion = UN.nn.MSELoss()

# %%

# Train the model and save checkpoints
UN.train_model(model, train_loader, optimizer, criterion, epochs, checkpoint_dir, checkpoint_freq)
# %%
# Optionally, save the final model for inference
UN.save_model_for_inference(model, './final_model.pth')

# After training, visualize results for the first image in the dataset
#input_image = torch.tensor(input_images[0:1]).to(device)  # Get the first image
#target_image = torch.tensor(target_images[0:1]).to(device)  # Get the first target

UN.visualize_results(model, input_image, target_image)
