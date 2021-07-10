from multihead import make_mh
from scattering import calc_spectrum
from scattering import acquisition as bo
from bagging.get_data import get_problem, get_obj

import torch
from torch.utils import data
import torch.nn.functional as F
import numpy as np
from torch import cuda


import pickle
from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.utils.extmath import randomized_svd
from matplotlib import pyplot as plt

import argparse

def main(n_batch, n_epochs, batch_size=8, n_train=250, acquisition="ei", objective="orange",
        x_dim=6, trials=1, bbs=(0.9,0.999), ens_size=10, n_start=5, lr=0.01, device=0):

    # Select the default device
    device = "cuda:"+ str(device) if cuda.is_available() else "cpu"

    params = calc_spectrum.MieScattering(n_layers=x_dim)

    prob_fun = get_problem(params, objective=objective)
    obj_fun = get_obj(params, objective=objective)


    for _ in range(trials):

        x_train = params.sample_x(n_start)
        z_train = prob_fun(x_train)
        y_train = obj_fun(z_train)

        x_train = torch.from_numpy(x_train).to(device).float()
        z_train = torch.from_numpy(z_train).to(device).float().squeeze()

        multihead = make_mh.MhAverage(ens_size=ens_size, n_units=256, n_layers=8, in_dim=6, out_dim=201, act="prelu").to(device)
        def f(samples):
            samples = torch.from_numpy(samples).to(device).float()
            return multihead.predict_posterior(samples)

        y_best_i = np.argmax(y_train, axis=0)
        x_best = x_train[[y_best_i], :]
        y_best = np.max(y_train)

        n_data_list, y_best_list = [], []
        optimizer = torch.optim.Adam(multihead.parameters(), lr=lr, betas=bbs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=125)
        #tsne = TSNE()
        preds = []
        for i in tqdm(range(n_train)):
	    
            nepo = 200 if i == 0 else n_epochs
            
            _ = make_mh.train(multihead, nepo, batch_size, x_train, z_train, optimizer, scheduler) 
            for p1, p2 in zip(multihead.ensemble[0].parameters(), multihead.ensemble[2].parameters()):
                if p1.data.ne(p2.data).sum() > 0:
                    print("not the same")
                    break
            # new sample
            x_sample = params.sample_x(int(1e4))
            # pred = torch.mean(torch.stack(pred, dim=-1), dim=-1).squeeze(1).detach().cpu().numpy()
            # print(pred.shape)
            # ypred = obj_fun(pred).reshape((1,-1))
            # preds.append(ypred)
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
        # plt.figure(figsize=(9,6))
        # print(np.vstack(preds).shape)
        # preds, _, _ = randomized_svd(np.vstack(preds), n_components=50, n_iter=300, random_state=None)
        # embedded = tsne.fit_transform(preds)
        # plt.plot(embedded[:, 0], embedded[:, 1], marker="o")
        # plt.savefig("t-sne.png")
        if ens_size == 10:
            n_list_f = "res_lists/n_list_averaged_t2" + str(lr) + "-" + str(bbs[0]) + "-" + str(bbs[1]) + ".txt"    
            y_list_f = "res_lists/y_list_averaged_t2" + str(lr) + "-" + str(bbs[0]) + "-" + str(bbs[1]) + ".txt"
        else:
            n_list_f = "res_lists/n_list_averaged_t2" + str(lr) + "-" + str(bbs[0]) + "-" + str(bbs[1]) + "-" + str(ens_size) + ".txt"    
            y_list_f = "res_lists/y_list_averaged_t2" + str(lr) + "-" + str(bbs[0]) + "-" + str(bbs[1]) + "-" + str(ens_size) + ".txt" 
        with open(n_list_f, "wb+") as fb:
            pickle.dump(n_data_list, fb)
        with open(y_list_f, "wb+") as fb:
            pickle.dump(y_best_list, fb)
        
        print(x_best)
        print(np.max(y_best_list))



if __name__ == "__main__":
    main(8, 10, bbs=(0.71,0.92), ens_size=7, lr=0.01, device=3)
    # parser = argparse.ArgumentParser(description="Averaged multihead optimization")
    # parser.add_argument("--mode", type=int, default=0)
    # args = parser.parse_args()
    # kwargs = vars(args)
    # if kwargs["mode"] == 0:
    #     for i in [0.0005, 0.001, 0.003, 0.005, 0.008, 0.01]:
    #         for j in [0.6, 0.7, 0.8, 0.9]:
    #             for k in [0.92, 0.999]:
    #                 print(i, j, k)
    #                 main(8, 10, lr=i, bbs=(j, k))
    # else:
    #     for i in [0.0005, 0.001, 0.003, 0.005, 0.008, 0.01]:
    #         for j in [0.67, 0.69, 0.71, 0.73]:
    #             for k in [0.9, 0.92]:
    #                 print(i, j, k)
    #                 main(8, 10, lr=i, bbs=(j, k))
