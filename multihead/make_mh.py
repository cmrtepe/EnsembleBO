import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch import cuda

device = "cuda:3" if cuda.is_available() else "cpu"
class MultiHead(nn.Module):
    
    def __init__(self, ens_size=10, n_units=256, n_layers=8, in_dim=6, out_dim=201):
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
    
    def train(self, n_epochs, batch_size, input, target, opts, scheds, train_loader=None):

        tdataset = SimpleDataset(input, target)
        # if data_loader is given use that, if not, use the input and target values
        tloader = train_loader if train_loader else data.DataLoader(tdataset, batch_size, shuffle=True) 
        
        losses = []
        preds = []

        # loop over models in the ensemble to trian them separately in a sequential order
        for i in range(len(self.models)):
            model = self.models[i]

            model.train()

            opt = opts[i]
            sched = scheds[i]
            
            for epoch in range(n_epochs):
                for batch_id, (input, target) in enumerate(tloader):
                    input = input.to(device).float()
                    target = target.to(device).float()
                    
                    opt.zero_grad()

                    total_loss = 0
                    loss_fun = nn.MSELoss()

                    pred = model(input)

                    loss = loss_fun(pred, target)

                    loss.backward()
                    # optimizer step
                    opt.step()

                    total_loss += loss.item()
                    # take the first pred for analyzing the model's performance
                    if batch_id == 0 and n_epochs == 1:
                        preds.append(pred.squeeze().detach().cpu().numpy())
                # step the scheduler for each epoch
                sched.step()
        return preds



def train(model, n_epochs, batch_size, input, target, optimizer, scheduler):
    # train method for ensemble created using 1d convolutional layers 

    # torch dataset and data loader for batching
    tdataset = SimpleDataset(input, target)
    tloader = data.DataLoader(tdataset, batch_size, shuffle=True)

    model.train()

    losses = []
    for _ in range(n_epochs):
        total_loss = 0
        for batch_id, (input, target) in enumerate(tloader):

            optimizer.zero_grad()

            loss_fun = nn.MSELoss()
            preds = model(input)
            loss = 0
            for pred in preds:
                loss += loss_fun(pred, target)
                pred.unsqueeze(1)
            loss = loss/len(preds)
            loss.backward()
            optimizer.step()

            total_loss += loss
        scheduler.step()
        losses.append(total_loss.item()/len(tloader))
    return preds

class ConvFC(nn.Module):
    # 1D convolution layers used for representing linear layers since they are easier to
    # parallize compared to linear layers in torch. 
    def __init__(self, ens_size=10, n_units=256, n_layers=8, in_dim=6, out_dim=201):
        super(ConvFC, self).__init__()
        
        self.out_dim = out_dim # output dimensions

        self.ens_size = ens_size # number of models in the ensemble

        arch = [n_units]*n_layers # architechture of the model 
        
        layers = [] # initialize the layers as an empty list
        
        if len(arch) > 2:
            prev = in_dim 
            for units in arch:
                # in_channels chosen to make convolution layer the same as an ensemble
                # of linear layers with in_size = prev and out_size = units e.g.
                # nn.Linear(pre, units) 
                in_channels = prev*ens_size  
                out_channels = units*ens_size

                layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, groups=ens_size))
                # ReLU activation
                layers.append(nn.ReLU())
                # new prev for the next layer
                prev = units
            # last layer of the model
            layers.append(nn.Conv1d(prev*ens_size, out_dim*ens_size, kernel_size=1, groups=ens_size))
        
        else:
            layers.append(nn.Conv1d(in_dim*ens_size, out_dim*ens_size, kernel_size=1, groups=ens_size))
        
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        # mean over ensemble dimension
        input = torch.cat([input]*self.ens_size, dim=-1).view(input.size(0), -1, 1)
        return torch.mean(self.model(input).view(input.size(0), -1, self.ens_size), dim=-1)

    def predict_posterior(self, input):
        # output of the model without the mean for sampling from the output
        input = torch.cat([input]*self.ens_size, dim=-1).view(input.size(0), -1, 1)
        return self.model(input).view(input.size(0), self.out_dim, self.ens_size)
    

class SimpleDataset(data.Dataset):
    # Simple input target dataset with additional add method to add
    # new datapoint to the set
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








