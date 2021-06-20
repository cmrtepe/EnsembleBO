import torch
import torch.nn as nn
import numpy as np


class SRM(nn.Module):

    def __init__(self, n_layers=8, n_units=512, input_size=6, output_size=201):
        super(SRM, self).__init__()
        layers = []
        if n_layers > 1:
            layers.append(nn.Linear(input_size, n_units))
            layers.append(nn.ReLU())
            for i in range(n_layers-2):
                layers.append(nn.Linear(n_units,n_units))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_units, output_size))
        else:
            layers = [nn.Linear(input_size, output_size), nn.ReLU()]
        self.model = nn.Sequential(*layers[:-1])
        self.final = layers[-1]

    def forward(self, inputs):

        return self.final(self.model(inputs))


class EnsembleNet(nn.Module):
  def __init__(self, model_list, hidden_size, num_of_ens):
      super(EnsembleNet, self).__init__()
      self.ensemble = nn.ModuleList()

      for model in model_list:
          model.final = nn.Identity()
          for param in model.parameters():
              param.requires_grad = False
          self.ensemble.append(model)

      self.last_layer = nn.Sequential(nn.Linear(num_of_ens*hidden_size, hidden_size, bias=False),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size, 201, bias=False))

  def forward(self, input):
      pred = torch.cat([model(input) for model in self.ensemble], dim=-1)
      pred = self.last_layer(pred)
      return pred

