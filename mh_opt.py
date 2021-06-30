from multihead import make_mh
from scattering import calc_spectrum
from scattering import acquisition as bo
from bagging.get_data import get_problem, get_obj

import torch
from torch.utils import data
import numpy as np
from torch import cuda
device = "cuda" if cuda.is_available() else "cpu"

import pickle
from tqdm import tqdm

def main(n_batch, n_epochs, batch_size=8, n_train=250, acquisition="ei", objective="orange",
        x_dim=6, trials=1, architecture=(32,128,256,128), ens_size=10, n_start=5, lr=0.01):

    params = calc_spectrum.MieScattering(n_layers=x_dim)

    prob_fun = get_problem(params, objective=objective)
    obj_fun = get_obj(params, objective=objective)

    

    for _ in range(trials):

        x_train = params.sample_x(n_start)
        z_train = prob_fun(x_train)
        y_train = obj_fun(z_train)

        x_train = torch.from_numpy(x_train).to(device).float()
        z_train = torch.from_numpy(z_train).to(device).float().squeeze()

        multihead = make_mh.MultiHead(ens_size=ens_size, n_units=256, n_layers=8, in_dim=6, out_dim=201).to(device)
        multihead.reset_weights()
        def f(samples):
            samples = torch.from_numpy(samples).to(device).float()
            return multihead.predict_posterior(samples)

        y_best_i = np.argmax(y_train, axis=0)
        x_best = x_train[[y_best_i], :]
        y_best = np.max(y_train)

        n_data_list, y_best_list = [], []
        opts = []
        scheds = []
        for i in range(ens_size):
            opts.append(torch.optim.Adam(multihead.models[i].parameters(), lr=lr))
            scheds.append(torch.optim.lr_scheduler.CosineAnnealingLR(opts[i], 50))
        for i in tqdm(range(n_train)):
	    
            nepo = 200 if i == 0 else n_epochs
            
            _ = multihead.train(nepo, batch_size, x_train, z_train, opts, scheds)

            # new sample
            x_sample = params.sample_x(int(1e4))
            _, x_new = bo.ei_mc(x_sample, f, y_best, obj_fun)
            z_new = prob_fun(x_new)
            

            # to torch tensor
            x_new = torch.from_numpy(x_new).to(device).float()
            z_new = torch.from_numpy(z_new).to(device).float()

            # new dataset 
            x_train = torch.vstack([x_train, x_new])
            z_train = torch.vstack([z_train, z_new])
            
            i_best = np.argmax(obj_fun(z_train.detach().cpu().numpy())) # new best idx
            y_best = obj_fun(z_train.detach().cpu().numpy())[i_best]

            n_data_list.append(x_train.size(0))
            y_best_list.append(y_best)
            print("Trained with %d data points. Best value=%f" % (x_train.size(0), y_best))
            
        if lr == 0.01:
            with open("n_list_newt.txt", "wb+") as fb:
                pickle.dump(n_data_list, fb)
            with open("y_list_newt.txt", "wb+") as fb:
                pickle.dump(y_best_list, fb)
        else:
            with open("n_list_newtnewlr.txt", "wb+") as fb:
                pickle.dump(n_data_list, fb)
            with open("y_list_newtnewlr.txt", "wb+") as fb:
                pickle.dump(y_best_list, fb)
        print(x_best)
        print(np.max(y_best_list))



if __name__ == "__main__":

    main(8, 10)
    main(8, 10, lr=0.005)
