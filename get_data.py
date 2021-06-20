from scattering import calc_spectrum
import numpy as np
import torch
from torch.utils import data

def get_obj(params, objective='orange'):
    if objective == 'orange':
        # Maximize the min scattering in 600-640nm range, minimize the max scattering outside of that
        lam = params.lam
        i1, = np.where(lam == 600)
        i2, = np.where(lam == 640)  # non-inclusive
        i1 = i1[0]
        i2 = i2[0]

        def obj_fun(y):
            return np.sum(y[:, i1:i2], axis=1) / np.sum(np.delete(y, np.arange(i1, i2), axis=1), axis=1)
    elif objective == 'hipass':
        lam = params.lam
        i1, = np.where(lam == 600)
        i1 = i1[0]

        def obj_fun(y):
            return np.sum(y[:, i1:], axis=1) / np.sum(y[:, :i1], axis=1)
    else:
        raise ValueError("Could not find an objective function with that name.")
    return obj_fun


def get_problem(params, objective="orange"):
    """Get objective function and problem parameters"""

    # Different objective functions to maximize during optimization
    if objective == "orange" or objective == 'hipass':
        def prob(x):
            return params.calc_data(x)

    else:
        raise ValueError("No objective function specified.")

    return prob

class ScatteringDataset(data.Dataset):
    def __init__(self, input, target, z_target=None):
        if target.ndim == 1:
            target = target[:, np.newaxis]
        if z_target is not None and z_target.ndim == 1:
            z_target = z_target[:, np.newaxis]
        self.x = input
        self.y = target
        self.z = z_target

    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[[idx],:], self.y[[idx], :]
    #def add(self, input_target_pair):

def get_data(number_of_dp):

    params = calc_spectrum.MieScattering(n_layers=6)
    prob_fun = get_problem(params, objective="orange")
    input = params.sample_x(number_of_dp)
    target = prob_fun(input)[:, np.newaxis]
    val_ratio = 0.1
    n, _ = input.shape
    training_id = int(n*(1-val_ratio))
    training_input = torch.from_numpy(input[:training_id,:])
    training_target = torch.from_numpy(target[:training_id,:]).squeeze()
    val_input = torch.from_numpy(input[training_id:,:])
    val_target = torch.from_numpy(target[training_id:,:]).squeeze()
    return training_input, training_target, val_input, val_target