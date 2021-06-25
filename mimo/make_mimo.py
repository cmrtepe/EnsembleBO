import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch import cuda

device = "cuda" if cuda.is_available() else "cpu"

def create_mimo(architecture, data_dim=1, ens_size=1):
    """Create a MIMO model by expanding input/ouput layer by ensemble size."""
    # The only modification needed by MIMO: expand input/output layer by ensemble size
    num_logits = 1  # Since this is a regression problem.
    inputs_size = data_dim * ens_size
    outputs_size = num_logits * ens_size
    
    layers = []
    prev = inputs_size
    layers.append(nn.Flatten())
    for units in architecture:
        layers.append(nn.Linear(prev, units))
        layers.append(nn.ReLU())
        prev = units
    layers.append(nn.Linear(prev, outputs_size))

    return nn.Sequential(*layers)

def train(model, data_loader, optimizer, ens_size):

    model.train()

    total_loss = 0

    for batch_id, (input, target) in enumerate(data_loader):

        input = input.to(device).float()
        target = target.to(device).float()
        inputs = [input]
        targets = [target]
        for _ in range(ens_size - 1):
            idx = torch.randperm(target.size(0))
            shuffled_in = input[idx]
            shuffled_trg = target[idx]
            inputs.append(shuffled_in)
            targets.append(shuffled_trg)
        inputs = torch.cat(inputs, dim=1)
        targets = torch.cat(targets, dim=1).squeeze(-1)
        optimizer.zero_grad()

        loss_fun = nn.MSELoss()
        pred = model(inputs)
        loss = loss_fun(pred, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss/len(data_loader)

def test(model, data_loader):
    
    model.eval()

    test_loss = 0

    with torch.no_grad():
        for batch_id, (input, target) in enumerate(data_loader):
           
            input = input.to(device).float()
            target = target.to(device).float()

            output = model(input)

            loss_fun = nn.MSELoss()
            pred = model(input)
            loss = loss_fun(pred, target)

            test_loss += loss.item()
            
    return test_loss/len(data_loader)