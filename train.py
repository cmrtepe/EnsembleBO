import argparse
from ensemble import SRM, EnsembleNet
import torch
import torch.nn as nn
from torch.utils import data
from torch import cuda
from get_data import get_data, ScatteringDataset

device = "cuda" if cuda.is_available() else "cpu"
def train_mod(model, data_loader, optimizer):
    model.train()

    total_loss = 0

    for batch_id, (input, target) in enumerate(data_loader):

        input = input.to(device).float()
        target = target.to(device).float()
        optimizer.zero_grad()

        loss_fun = nn.MSELoss()
        pred = model(input)
        loss = loss_fun(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss
    return total_loss.item()/len(data_loader)

def test_mod(model, data_loader):
    
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

            test_loss += loss
            
    return test_loss.item()/len(data_loader)


def train_ens(n_models, n_units, n_layers, train_loader, val_loader, train_loader_ens, val_loader_ens,
              epochs=500):
    model_ls = nn.ModuleList()
    if device=="cuda":
        print("using cuda")
    else:
        print("using cpu")
    for i in range(n_models):
        net = SRM(n_layers, n_units).to(device)
        optimizer = torch.optim.Adam(net.parameters(), 
                                    lr=0.001, weight_decay=5e-4)
        for epoch in range(1, epochs+1):
            _ = train_mod(net, train_loader, optimizer)
            model_loss = test_mod(net, val_loader)
        model_ls.append(net)
    ensemble = EnsembleNet(model_ls, n_units, n_models).to(device)
    optimizer = torch.optim.Adam(ensemble.parameters(), 
                                    lr=0.001)
    for epoch in range(1, epochs+1):
        _ = train_mod(ensemble, train_loader_ens, optimizer)
        val_loss = test_mod(ensemble, val_loader_ens)
    return val_loss, model_loss

def main(n_models, n_data, n_epochs, n_layers, n_units, batch_size):
    
    # for pretraining models
    train_data_in, train_data_trg, val_data_in, val_data_trg = get_data(n_data)

    train_data_in, train_data_trg = train_data_in.to(device), train_data_trg.to(device)
    val_data_in, val_data_trg = val_data_in.to(device), val_data_trg.to(device) 

    training_dataset = ScatteringDataset(train_data_in, train_data_trg)
    val_dataset = ScatteringDataset(val_data_in, val_data_trg)

    train_loader = data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size)

    # for training the ensemble
    train_in_ens, train_trg_ens, val_in_ens, val_trg_ens = get_data(n_data)

    train_in_ens, train_trg_ens = train_in_ens.to(device), train_trg_ens.to(device)
    val_in_ens, val_trg_ens = val_in_ens.to(device), val_trg_ens.to(device)

    training_ens_dataset = ScatteringDataset(train_in_ens, train_trg_ens)
    val_ens_dataset = ScatteringDataset(val_in_ens, val_trg_ens)

    train_loader_ens = data.DataLoader(training_ens_dataset, batch_size=batch_size, shuffle=True)
    val_loader_ens = data.DataLoader(val_ens_dataset, batch_size=batch_size)

    val_loss = []

    for i in range(1, 4):
        val_loss_s, _ = train_ens(n_models, n_units*2**i, n_layers, train_loader, val_loader,
                            train_loader_ens, val_loader_ens, n_epochs)
        val_loss.append(val_loss_s)

    print(val_loss)
    return val_loss
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble training")
    parser.add_argument("--n-models", type=int, default=10)
    parser.add_argument("--n-data", type=int, default=10000)
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--n-units", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=8)

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
