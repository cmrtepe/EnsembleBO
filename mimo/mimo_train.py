from tqdm import tqdm
from make_data import create_data
from make_mimo import create_mimo, train
import argparse
import torch
from torch import cuda

device = "cuda" if cuda.is_available() else "cpu"

def run_mimo(architecture=(32, 128), ens_sizes=(1, 2, 3, 4, 5),
                         lr=0.01, batch_size=4, n_epochs=2000,
                         data_dim=1, data_noise=0.08, 
                         n_train=64, n_test=3000, num_reps=20,
                         eval_epoch=2, print_epoch=-1):
    torch.manual_seed(0)

    # Create testing data.
    Xtest0, ytest, Xtest, _ = create_data(n_test, data_dim=data_dim, 
                                    data_noise=data_noise, support=(-1.,1.))
    Xtest0_wide, ytest_wide, Xtest_wide, _ = create_data(n_test, data_dim=data_dim, 
                                                data_noise=data_noise, support=(-2.,2.))

    # Train MIMO models with different ensemble sizes over multiple random seeds.
    ytest_pred = {}
    ytest_wide_pred = {}

    for rep in range(num_reps):
        print('Repetition', rep)

        _, y, X, training_loader = create_data(n_train, data_dim=data_dim, 
                                            data_noise=data_noise, batch_size=batch_size)

        ytest_pred[rep] = {}
        ytest_wide_pred[rep] = {}

        for ens_size_id in tqdm(range(len(ens_sizes))):
        # Specified ensemble size.
            ens_size = ens_sizes[ens_size_id]
            
            ytest_pred[rep][ens_size] = {}
            ytest_wide_pred[rep][ens_size] = {}

            # Train a MIMO model.
            
            mimo_mlp = create_mimo(architecture, data_dim=data_dim,
                                    ens_size=ens_size).to(device)
            print(mimo_mlp)
            
            optimizer = torch.optim.Adam(mimo_mlp.parameters(), lr=lr)

            for epoch in range(n_epochs):
                loss = train(mimo_mlp, training_loader, optimizer, ens_size)

                if print_epoch > 0 and epoch % print_epoch == 0:
                    print('[{:4d}] train sq. loss {:0.3f}'.format(epoch, loss))

                if epoch % eval_epoch == 0:
                # Save testing performance.
                    with torch.no_grad():
                        per_ens_member_ytest_pred = mimo_mlp(torch.tile(Xtest.to(device), (1, ens_size)))
                        ytest_pred[rep][ens_size][epoch] = per_ens_member_ytest_pred

                        per_ens_member_ytest_wide_pred = mimo_mlp(torch.tile(Xtest_wide.to(device), (1, ens_size)))
                        ytest_wide_pred[rep][ens_size][epoch] = per_ens_member_ytest_wide_pred

    return Xtest0, ytest, ytest_pred, Xtest0_wide, ytest_wide, ytest_wide_pred

def main(exp_config):

    Xtest, ytest, ytest_pred, Xtest_wide, ytest_wide, ytest_wide_pred = run_mimo(**exp_config)
    # Save the tensors for further analysis
    torch.save(Xtest, "tensors/Xtest.pt")
    torch.save(ytest, "tensors/ytest.pt")
    torch.save(ytest_pred, "tensors/ytest_pred.pt")
    torch.save(Xtest_wide, "tensors/Xtest_wide.pt")
    torch.save(ytest_wide, "tensors/ytest_wide.pt")
    torch.save(ytest_wide_pred, "tensors/ytest_wide_pred.pt")

if __name__ == "__main__":
    # Arguments for training
    parser = argparse.ArgumentParser(description="Ensemble training")
    parser.add_argument("--data-dim", type=int, default=1)
    parser.add_argument("--architecture", type=list, default=(32, 128)) #32 , 128
    parser.add_argument("--n-train", type=int, default=64)
    parser.add_argument("--n-test", type=int, default=3000) # 3000
    parser.add_argument("--num-reps", type=int, default=20) # 20
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n-epochs", type=int, default=2000) # 2000
    parser.add_argument("--ens-sizes", type=list, default=(1,2,3,4,5))

    args = parser.parse_args()
    kwargs = vars(args)
    main(kwargs)

