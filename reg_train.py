from multihead import make_mh
from scattering import calc_spectrum
from bagging.get_data import get_problem, get_obj

import torch
import torch.nn as nn
from torch.utils import data

import numpy as np
from torch import cuda
device = torch.device("cuda:3") if cuda.is_available() else "cpu"


from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.utils.extmath import randomized_svd
from matplotlib import pyplot as plt

import pickle

# Optimization using an ensemble where the loss of each model in the ensemble is averaged
# which is then used for backpropagation similar to a stacking type of ensemble

def main(a, n_epochs, n_data, check_point=5, batch_size=16, lr=0.001, bbs=(0.9,0.999), ens_size=10, 
         n_layers=8, n_units=256, x_dim=6, out_dim=201, objective="orange"):

    params = calc_spectrum.MieScattering(n_layers=x_dim)
    prob_fun = get_problem(params, objective=objective)
    obj_fun = get_obj(params, objective=objective)
    if a == 0:
        
        x_train = params.sample_x(n_data)
        z_train = prob_fun(x_train)
        x_train = torch.from_numpy(x_train)
        z_train = torch.from_numpy(z_train)
        
        torch.save(x_train, "tensors/x_train.pt")
        torch.save(z_train, "tensors/z_train.pt")

        x_val = params.sample_x(n_data)
        z_val = prob_fun(x_val)
        x_val = torch.from_numpy(x_val)
        z_val = torch.from_numpy(z_val)
        
        torch.save(x_val, "tensors/x_val.pt")
        torch.save(z_val, "tensors/z_val.pt")
        
        a += 1
    else:
        x_train = torch.load("tensors/x_train.pt")
        z_train = torch.load("tensors/z_train.pt")

        x_val = torch.load("tensors/x_val.pt")
        z_val = torch.load("tensors/z_val.pt")

    
    
    # torch.manual_seed(5)
    # multihead = make_mh.MultiHead(ens_size=ens_size, n_units=n_units, n_layers=n_layers, in_dim=x_dim, out_dim=out_dim).to(device)
    # optimizer = torch.optim.Adam(multihead.parameters(), lr=lr, betas=bbs)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=125)

    
    # torch.manual_seed(5)
    # multihead2 = make_mh.MultiHead(ens_size=ens_size, n_units=n_units, n_layers=n_layers, in_dim=x_dim, out_dim=out_dim).to(device)
   
    # optimizer2 = torch.optim.Adam(multihead2.parameters(), lr=lr, betas=(0.71,0.92))
    # scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=125)

    torch.manual_seed(0)
    multihead = make_mh.ConvFC(ens_size=ens_size, n_units=n_units, n_layers=n_layers, in_dim=x_dim, out_dim=out_dim).to(device)
    optimizer = torch.optim.Adam(multihead.parameters(), lr=lr, betas=bbs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=125)

    
    torch.manual_seed(0)
    multihead2 = make_mh.ConvFC(ens_size=ens_size, n_units=n_units, n_layers=n_layers, in_dim=x_dim, out_dim=out_dim).to(device)
    
    optimizer2 = torch.optim.Adam(multihead2.parameters(), lr=0.001, betas=(0.75,0.999))
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=125)

    tdataset = SimpleDataset(x_train, z_train)
    tloader = data.DataLoader(tdataset, batch_size=batch_size, shuffle=True, drop_last=True)

    vdataset = SimpleDataset(x_val, z_val)
    vloader = data.DataLoader(tdataset, batch_size=batch_size, shuffle=True, drop_last=True)
    losses = []
    pred_es = []
    pred_es2 = []
    corr_loss = []
    corrs = []
    tsne = TSNE(perplexity=50.0)
    for epoch in tqdm(range(n_epochs)):

        # preds = multihead.train(1, 8, x_train, z_train, optimizer, scheduler, tloader)
        # preds2 = multihead2.train(1, 8, x_train, z_train, optimizer2, scheduler2, tloader)
        preds = conv_train(multihead, tloader, optimizer, scheduler)
        preds2 = conv_train(multihead2, tloader, optimizer2, scheduler2)
        val_loss = np.log(test(multihead, vloader))
        corr = np.corrcoef(preds[:,0].reshape(-1,), preds[:,2].reshape(-1,))
        corr_loss.append(corr[0,1]/(val_loss))
        corrs.append(corr[0,1])
        # if epoch % check_point == 0:
        #     if epoch == 0:
        #         torch.save({"check" + str(epoch): multihead}, "run1.pt")
        #     else:
        #         checkp = torch.load("run1.pt")
        #         checkp["check"+str(epoch)] = multihead
        #         torch.save(checkp, "run1.pt")
        if epoch % check_point == 0:
            pred_a = preds[:,0]
            pred_b = preds2[:,0]
            pred_a = pred_a.reshape((-1,201))
            pred_es.append(pred_a)
            pred_b = pred_b.reshape((-1,201))
            pred_es2.append(pred_b)
        
        # val_loss2 = test(multihead2, vloader)
        losses.append(val_loss)
        if epoch % 10 == 0:
            print('[{:4d}] validation sq. loss {:0.8f}'.format(epoch, val_loss))
    
    
    f = plt.figure(1, figsize=(9,6))
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    ax1.plot([i for i in range(len(corrs))], corrs, marker="o")
    ax2.plot(losses, corr_loss, marker="o")
    plt.savefig("corr-val_loss2conv.png")


    plt.figure(2, figsize=(9,6))
    n = np.vstack(pred_es).shape[0]
    pred_es = pred_es + pred_es2
    pred_es, _, _ = randomized_svd(np.vstack(pred_es), n_components=30, n_iter=300, random_state=None)
    print(pred_es.shape)
    embedded = tsne.fit_transform(pred_es)
    plt.plot(embedded[:n, 0], embedded[:n, 1], linestyle="dashed", marker="o")
    plt.plot(embedded[n:,0], embedded[n:, 1], linestyle="dashdot", marker="o")
    plt.savefig("t-sne2t2convperp50.png")
    # plt.figure()
    # losses_l  = "res_lists/losses_l" + str(lr) + "-" + str(bbs[0]) + "-" + str(bbs[1])  + ".txt" 
    
    # with open(losses_l, "wb+") as fb:
    #         pickle.dump(losses, fb)
    

def conv_train(model, data_loader, optimizer, scheduler):
    
    model.train()
    preds = None
    losses = []
    total_loss = 0
    for batch_id, (input, target) in enumerate(data_loader):

        input = input.to(device).float()
        target = target.to(device).float()

        optimizer.zero_grad()

        loss_fun = nn.MSELoss()
        pred = model(input).squeeze()
        if batch_id == 0:
            preds = model.predict_posterior(input).detach()[0,:,:]
        loss = loss_fun(pred, target.squeeze())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    scheduler.step()
    losses.append(total_loss/len(data_loader))
    return preds.squeeze().cpu().numpy()

def train(model, data_loader, optimizer):

    model.train()

    total_loss = 0
    predss = []
    for batch_id, (input, target) in enumerate(data_loader):
        input = input.to(device).float()
        target = target.to(device).float()
        optimizer.zero_grad()

        loss_fun = nn.MSELoss()
        preds = model(input).squeeze()
        loss = 0
        for pred in preds:
            loss += loss_fun(pred, target)
        loss = loss/len(preds)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predss.append(preds[0].squeeze().detach().cpu().numpy())
        predss.append(preds[9].squeeze().detach().cpu().numpy())
        
        

    return total_loss/len(data_loader), predss

def test(model, data_loader):

    model.eval()

    test_loss = 0

    with torch.no_grad():
        for batch_id, (input, target) in enumerate(data_loader):
           
            input = input.to(device).float()
            target = target.to(device).float()

            pred = model(input)

            loss_fun = nn.MSELoss()
            loss = loss_fun(pred, target.squeeze())
            test_loss += loss.item()
    # with torch.no_grad():
    #     for batch_id, (input, target) in enumerate(data_loader):
    #         preds = []
    #         input = input.to(device).float()
    #         target = target.to(device).float()
    #         for model in model.models:
    #             pred = model(input)
    #             preds.append(pred)
    #         loss_fun = nn.MSELoss()
    #         loss = 0
    #         for pred in preds:
    #             loss += loss_fun(pred, target)
    #         loss = loss/len(preds)
    #         test_loss += loss.item()

            
    return test_loss/len(data_loader)

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

if __name__=="__main__":

    a = 1
    main(a, 500, 1000, lr=0.0001, bbs=(0.9, 0.999))