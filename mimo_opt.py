from mimo import make_mimo, make_data
from scattering import calc_spectrum
from scattering import acquisition as bo
from bagging.get_data import get_problem, get_obj

import torch
from torch.utils import data
import numpy as np

def main(n_batch, n_epochs, batch_size=8, n_train=200, acquisition="ei", objective="orange",
        x_dim=6, trials=1, architecture=(32,128,256,128), ens_size=10, n_start=5):

    params = calc_spectrum.MieScattering(n_layers=x_dim)

    prob_fun = get_problem(params, objective=objective)
    obj_fun = get_obj(params, objective=objective)

    

    for _ in range(trials):

        x_train = params.sample_x(n_start)
        z_train = prob_fun(x_train)
        y_train = obj_fun(z_train)

        x_train = torch.from_numpy(x_train)
        z_train = torch.from_numpy(z_train).squeeze()

        training_dataset = make_data.SimpleDataset(x_train, z_train)
        training_dlr = data.DataLoader(training_dataset, batch_size, shuffle=True)

        mimo_mlp = make_mimo.create_mimo(architecture, data_dim=x_dim, ens_size=ens_size, num_logits=z_train.size(-1))

        def f(samples):
            samples = torch.from_numpy(samples)
            sampless = [samples]
            for _ in range(ens_size - 1):
                idx = torch.randperm(samples.size(0))
                shuffled_in = samples[idx]
                sampless.append(shuffled_in)
            samples = torch.cat(sampless, dim=1)
            return make_mimo.predict_posterior(mimo_mlp, x_dim, ens_size, samples.float())

        y_best_i = np.argmax(y_train, axis=0)
        x_best = x_train[[y_best_i], :]
        y_best = np.max(y_train)

        n_data_list, y_best_list = [], []

        for i in range(n_train):

            optimizer = torch.optim.Adam(mimo_mlp.parameters(), lr=0.01, weight_decay=1e-5)
            for epoch in range(n_epochs):
                _ = make_mimo.train(mimo_mlp, training_dlr, optimizer, ens_size)

            # new sample
            x_sample = params.sample_x(int(1e4))
            _, x_new = bo.ei_mc(x_sample, f, y_best, obj_fun)
            z_new = prob_fun(x_new)
            

            # to torch tensor
            x_new = torch.from_numpy(x_new)
            z_new = torch.from_numpy(z_new)

            # new dataset 
            x_train = torch.vstack([x_train, x_new])
            z_train = torch.vstack([z_train, z_new])
            training_dataset = make_data.SimpleDataset(x_train, z_train)
            training_dlr = data.DataLoader(training_dataset, batch_size, shuffle=True)
            
            i_best = np.argmax(obj_fun(z_train.detach().cpu().numpy())) # new best idx
            y_best = obj_fun(z_train.detach().cpu().numpy())[i_best]

            n_data_list.append(len(training_dataset))
            y_best_list.append(y_best)
            print("Trained with %d data points. Best value=%f" % (len(training_dataset), y_best))
        
        print(x_best)
        print(np.max(y_best_list))


if __name__ == "__main__":

    main(8, 100)



