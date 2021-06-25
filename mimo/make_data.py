import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch import cuda

from torch.distributions.uniform import Uniform

class SimpleDataset(data.Dataset):
    def __init__(self, input, target):
        self.x = input
        self.y = target

    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[[idx],:], self.y[[idx], :]

def create_data(n, data_dim=1, data_noise=0.02, batch_size=8, support=(-1., 1.)):
  
  # Create 1D features X0.
  lower, upper = support
  X0 = torch.rand(n,1)* (upper-lower) + lower
  noise = torch.rand(n, 1) * data_noise

  # Generate response. 
  y = X0 + 0.3 * torch.sin(2*np.pi * (X0 + noise)) + 0.3 * torch.sin(4*np.pi * (X0 + noise)) + noise

  # Embed X0 into high-dimensional space.
  X = X0
  if data_dim > 1:
    Proj_mat = torch.randn(1, data_dim)
    X = torch.matmul(X0, Proj_mat)

  # Produce high-dimensional dataset.
  torch_dataset = SimpleDataset(X, y)
  torch_dataloader = data.DataLoader(torch_dataset, batch_size, shuffle=True)

  return X0, y, X, torch_dataloader