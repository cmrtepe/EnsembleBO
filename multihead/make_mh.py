import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

class MultiHead(nn.Module):
    
    def __init__(self, ens_size=5, n_units=256, n_layers=8, in_dim=6, out_dim=201):
        super(MultiHead, self).__init__()
        # list of models in the ensemble
        models = nn.ModuleList()
        # architecture of each model is the same
        arch = [n_units]*n_layers
        # create models
        for i in range(ens_size):
            mdl = self.creat_model(arch, in_dim, out_dim)
            models.append(mdl)
        
        self.models = models
    
    def forward(self, input):
        
        # use //with torch.no_grad():// before using forward ! 
        return torch.mean(torch.vstack([model(input) for model in self.models]), dim=0)

    def predict_posterior(self, input):

        return torch.stack([model(input) for model in self.models], dim=-1)

    def reset_weights(self):
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data)
        for model in self.models:
            model.apply(weights_init)
    

    def creat_model(self, arch, in_size, out_size):
        
        layers = []
        # add layers given in arch
        if len(arch) > 2:
            prev = in_size
            for units in arch:
                layers.append(nn.Linear(prev, units))
                layers.append(nn.ReLU())
                prev = units
            layers.append(nn.Linear(prev, out_size))
        
        else:
            layers.append(nn.Linear(in_size, out_size))

        return nn.Sequential(*layers)
    
    def train(self, n_epochs, batch_size, input, target, opts, scheds, lr=0.01):

        tdataset = SimpleDataset(input, target)
        tloader = data.DataLoader(tdataset, batch_size, shuffle=True)
        
        losses = []

        for i in range(len(self.models)):
            model = self.models[i]
            optimizer = opts[i]
            scheduler = scheds[i]
            for epoch in range(n_epochs):
                model.train()

                total_loss = 0

                for batch_id, (input, target) in enumerate(tloader):

                    optimizer.zero_grad()

                    loss_fun = nn.MSELoss()

                    pred = model(input)
                    loss = loss_fun(pred, target)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    
                    losses.append(total_loss/len(tloader))
                scheduler.step()
        return losses

class SimpleDataset(data.Dataset):
    def __init__(self, input, target):
        self.x = input
        self.y = target

    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[[idx],:], self.y[[idx], :]
    def add(self, x_pt, y_pt):
        self.x = torch.cat([self.x, x_pt], dim=0)
        self.y = torch.cat([self.y, y_pt], dim=0)








