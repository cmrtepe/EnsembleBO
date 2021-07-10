from multihead import make_mh
from scattering import calc_spectrum
from scattering import acquisition as bo
from bagging.get_data import get_problem, get_obj

import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import cuda

import argparse 

import pickle
from tqdm import tqdm


# Optimization using an ensemble with each model trained separately. 

def main(n_batch, n_epochs, batch_size=8, n_train=250, acquisition="ei", objective="orange",
        x_dim=6, trials=1, bbs=(0.9,0.999), ens_size=10, n_start=5, lr=0.01, dev=0):
    
    device = "cuda:"+ str(dev) if cuda.is_available() else "cpu"

    # Create a MieScattering class with x_dim number of layers and default parameters
    params = calc_spectrum.MieScattering(n_layers=x_dim)

    # Given the MieScattering class, produces the problem function which takes thickness values
    # of nanoparticle layers and returns the corresponding spectra
    prob_fun = get_problem(params, objective=objective)

    # Takes the spectra produced by the prob_fun and returns the maximum value for some objective.
    # For "orange" objective, the obj_fun is used to maximize the scattering in 600-640 nm range 
    # and minimize the rest of the spectrum.
    obj_fun = get_obj(params, objective=objective)


    for _ in range(trials):
        # Small number of starting dataset with auxiliary information z_train
        x_train = params.sample_x(n_start)
        z_train = prob_fun(x_train)
        y_train = obj_fun(z_train)

        # To torch and gpu
        x_train = torch.from_numpy(x_train).to(device).float()
        z_train = torch.from_numpy(z_train).to(device).float().squeeze()

        # Create the model with given parameters
        torch.manual_seed(0)
        multihead = make_mh.ConvFC(ens_size=ens_size, n_units=256, n_layers=8, in_dim=6, out_dim=201).to(device)
        multihead2 = make_mh.ConvFC(ens_size=ens_size, n_units=256, n_layers=8, in_dim=6, out_dim=201).to(device)
        # Given sampled inputs, returns the posterior given by the current model
        def f(samples):
            samples = torch.from_numpy(samples).to(device).float()
            return multihead.predict_posterior(samples)
        
        # Current best value
        y_best_i = np.argmax(y_train, axis=0)
        x_best = x_train[[y_best_i], :]
        y_best = np.max(y_train)

        n_data_list, y_best_list = [], []
        # opts = []
        # scheds = []
        # Create separate optimizers and schedulers for models in the ensemble
        # for i in range(ens_size):
        #     opts.append(torch.optim.Adam(multihead.models[i].parameters(), lr=lr, betas=bbs))
        #     scheds.append(torch.optim.lr_scheduler.CosineAnnealingLR(opts[i], T_max=125))
        opt = torch.optim.Adam(multihead.parameters(), lr=lr, betas=bbs)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=125)
        # Optimization iterations
        for i in tqdm(range(n_train)):
            # Number of epochs is 200 for the first iteration and n_epochs for the consequent iterations 
            nepo = 200 if i == 0 else n_epochs
            
            # Train the model
            #_ = multihead.train(nepo, batch_size, x_train, z_train, opts, scheds)
            _ = train(multihead, nepo, ens_size, batch_size, x_train, z_train, opt, sched, dev)

            # Draw a new set of samples
            x_sample = params.sample_x(int(1e4))

            # Select the new input value from the set of samples using the acquisition function
            _, x_new = bo.ei_mc(x_sample, f, y_best, obj_fun)
            z_new = prob_fun(x_new)
            

            # to torch tensor
            x_new = torch.from_numpy(x_new).to(device).float()
            z_new = torch.from_numpy(z_new).to(device).float()

            # Create the new dataset with the new datapoint 
            x_train = torch.vstack([x_train, x_new])
            z_train = torch.vstack([z_train, z_new])
            
            # New best value
            i_best = np.argmax(obj_fun(z_train.detach().cpu().numpy())) # new best idx
            y_best = obj_fun(z_train.detach().cpu().numpy())[i_best]

            n_data_list.append(x_train.size(0))
            y_best_list.append(y_best)
            print("Trained with %d data points. Best value=%f" % (x_train.size(0), y_best))
            
        print("Trained with %d data points. Best value=%f" % (x_train.size(0), y_best))
        # File paths for saving the y_best and number of data points lists for each iteration
        n_list_f = "res_lists/n_list" + str(lr) + "-" + str(bbs[0]) + "-" + str(bbs[1]) + "-" + "conv2" + ".txt"    
        y_list_f = "res_lists/y_list" + str(lr) + "-" + str(bbs[0]) + "-" + str(bbs[1]) + "-" + "conv2"+ ".txt"   
        with open(n_list_f, "wb+") as fb:
            pickle.dump(n_data_list, fb)
        with open(y_list_f, "wb+") as fb:
            pickle.dump(y_best_list, fb)
        

def train(model, n_epochs, ens_size, batch_size, input, target, optimizer, scheduler, device):

    # torch dataset and data loader
    tdataset = make_mh.SimpleDataset(input, target)
    tloader = data.DataLoader(tdataset, batch_size, shuffle=True)
    
    device = "cuda:" + str(device) 
    
    # training mode
    model.train()
    
    preds = None
    losses = []
    
    for _ in range(n_epochs):
        total_loss = 0
        for batch_id, (input, target) in enumerate(tloader):
            # take each batch to gpu
            input = input.to(device).float()
            target = target.to(device).float()

            optimizer.zero_grad()
            
            # mean square error loss for regression
            loss_fun = nn.MSELoss()
            
            pred = model(input).squeeze()
            
            # first prediction of the model in this epoch 
            if batch_id == 0:
                preds = model.predict_posterior(input).detach()
            
            loss = loss_fun(pred, target.squeeze())
            loss.backward()
            
            optimizer.step()

            total_loss += loss
        # step scheduler each epoch
        scheduler.step()
        
        losses.append(total_loss.item()/len(tloader))
    
    return preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Averaged multihead optimization")
    parser.add_argument("--enssize", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs["enssize"], kwargs["device"])
    main(8, 10, lr=0.004, bbs=(0.7, 0.95), ens_size=kwargs["enssize"], dev=kwargs["device"])
    # for i in [0.0005, 0.001, 0.004, 0.007, 0.01]:
    #     for j in [0.7, 0.75, 0.85, 0.9]:
    #         for k in [0.95, 0.999]:
    #             main(8, 10, lr=i, bbs=(j, k), ens_size=kwargs["enssize"], device=kwargs["device"])
