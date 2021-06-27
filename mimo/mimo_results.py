import torch
import numpy as np
from matplotlib import pyplot as plt

def compute_ens_mean(ens_pred):
    return torch.mean(ens_pred, 1, keepdim=True)

def compute_mean_prediction(y_ens_pred_runs):
  """Computes mean of MIMO model prediction from different experiment runs."""
  ens_pred_list = []
  for y_ens_pred in y_ens_pred_runs.values():
    # Compute mean of ensemble prediction for each repetition
    ens_pred = compute_ens_mean(y_ens_pred)
    ens_pred_list.append(ens_pred)

  return torch.mean(torch.stack(ens_pred_list), dim=0)

def compute_squared_bias(y_test, y_ens_pred_runs):
  """Compute predictive squared bias over experiment repetitions."""
  y_pred_mean = compute_mean_prediction(y_ens_pred_runs)
  return torch.mean((y_test.detach().cpu() - y_pred_mean.detach().cpu())**2)

def compute_variance(y_ens_pred_runs):
  """Compute predictive variance over experiment repetitions."""
  ens_pred_mean = compute_mean_prediction(y_ens_pred_runs).detach().cpu()

  ens_var_list = []
  for y_ens_pred in y_ens_pred_runs.values():
    # Compute variance of ensemble prediction for each repetition
    ens_pred = compute_ens_mean(y_ens_pred).detach().cpu()
    ens_var_list.append((ens_pred_mean - ens_pred)**2)
  return torch.mean(torch.stack(ens_var_list))

def get_simulation_metadata(simu_result):
  """Extract number of repetitions, ensemble sizes and epoch ids from the simulation result."""
  rep_ids = list(simu_result.keys())
  ens_sizes = list(simu_result[rep_ids[0]].keys())
  epoch_ids = list(simu_result[rep_ids[0]][ens_sizes[0]].keys())
  return rep_ids, ens_sizes, epoch_ids
  

def compute_bias_variance_tradeoff(ytest, ytest_pred):
  """Computes bias-variance tradeoff for simulation result contained in ytest_pred."""
  # Extract information about ensemble sizes and epoch ids.
  rep_ids, ens_sizes, epoch_ids = get_simulation_metadata(ytest_pred)
  sq_biases = {}
  variances = {}
  for ens_size in ens_sizes:
    sq_bias_list = []
    var_list = []

    for epoch in epoch_ids:
      # Extract results over repetitions for a specific epoch and ensemble size
      ytest_pred_reps = {rep_id: ytest_pred[rep_id][ens_size][epoch] for rep_id in rep_ids}

      # compute bias and variance
      sq_bias = compute_squared_bias(ytest, ytest_pred_reps)
      variance = compute_variance(ytest_pred_reps)
      sq_bias_list.append(sq_bias.item())
      var_list.append(variance.item())

    sq_biases[ens_size] = np.array(sq_bias_list)
    variances[ens_size] = np.array(var_list)
  
  return sq_biases, variances

def plot_bias_variance_tradeoff(sq_biases, variances, ytest_pred, 
                                plot_type="bias", y_top=None, y_bottom=None, figsize=(9, 6)):
  
  rep_ids, ens_sizes, epoch_ids = get_simulation_metadata(ytest_pred)
  x_axis_epochs = np.array(epoch_ids) + 1

  plt.figure(figsize=figsize)
  for ens_size in ens_sizes:
    if plot_type == "bias":
      plt.plot(x_axis_epochs, np.log10(sq_biases[ens_size]), label='M={}'.format(ens_size))
      plot_ylabel = '$\\log_{10}(\\mathrm{squared\ bias})$'
    elif plot_type == "variance": 
      plt.plot(x_axis_epochs, np.log10(variances[ens_size]), label='M={}'.format(ens_size))
      plot_ylabel = '$\\log_{10}(\\mathrm{variance})$'
    elif plot_type == "loss":
      plt.plot(x_axis_epochs, np.log10(sq_biases[ens_size] + variances[ens_size]), label='ens. size {}'.format(ens_size))
      plot_ylabel = '$\\log_{10}(\\mathcal{E}_{\\mathrm{ens}})$'
    elif plot_type == "loss_diff": 
      single_model_loss = sq_biases[1] + variances[1]
      ensemble_model_loss = sq_biases[ens_size] + variances[ens_size]
      plt.plot(x_axis_epochs, np.log10(ensemble_model_loss) - np.log10(single_model_loss), label='M={}'.format(ens_size))
      plot_ylabel = '$\\log_{10}(\\mathcal{E}_{M})-\\log_{10}(\\mathcal{E}_{1})$'
    else:
      raise ValueError(f"Plot type {plot_type} not supported.")

  plt.legend()
  plt.ylim(top=y_top, bottom=y_bottom)
  plt.xlabel('epochs')
  plt.ylabel(plot_ylabel)
  
  if plot_type == "bias":
      plt.savefig("bias.png")
  else:
      plt.savefig("variance.png")

def main():

    ytest = torch.load("tensors/ytest.pt")
    ytest_pred = torch.load("tensors/ytest_pred.pt")

    sq_biases, variances = compute_bias_variance_tradeoff(ytest, ytest_pred)

    plot_bias_variance_tradeoff(sq_biases, variances, ytest_pred, plot_type="bias")
    plot_bias_variance_tradeoff(sq_biases, variances, ytest_pred, plot_type="variance")
    
    return 0

if __name__ == "__main__":

    main()
