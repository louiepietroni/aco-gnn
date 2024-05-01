import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_dataset
from aco import ACO
import graphnn.neuralnetwork as graphnn
from traintsp import train_variable, validate_best_variable_vary_evap

problem_size = 50
k_sparse = 50
epochs = 20
iterations_per_epoch = 1500
n_ants = 15

min_problem_size = 10
max_problem_size = 100

all_data = {}
SIGNIFICANCE_RUNS=15
STEPS=20
data_to_save = []

if False:
    base_data = []
    model_data = []
    for i in range(SIGNIFICANCE_RUNS):
        heuristic_network = graphnn.GNN(32, 12)
        # train_variable(heuristic_network, 20, 50, 1, iterations_per_epoch, n_ants, k_sparse=None)
        heuristic_network.eval()

        base_run_data = []
        model_run_data = []
        for evap in np.linspace(0, 1, 21):
            val_data = validate_best_variable_vary_evap(heuristic_network, min_problem_size, max_problem_size, n_ants, evap)
            model_run_data.append(val_data)

            val_data = validate_best_variable_vary_evap(None, min_problem_size, max_problem_size, n_ants, evap)
            base_run_data.append(val_data)
        print(f'Completed run {i}/{SIGNIFICANCE_RUNS}')
        base_data.append(base_run_data)
        model_data.append(model_run_data)
    torch.save(model_data, f'results/tsp/evap_model.pt')
    torch.save(base_data, f'results/tsp/evap_model.pt')

fig, ax = plt.subplots()

data_model = torch.load(f'results/tsp/evap_model.pt')
data_base = torch.load(f'results/tsp/evap_base.pt')
x = torch.linspace(0, 1, 21)

mean_model = data_model.mean(dim=0)
std_model = data_model.std(dim=0)
delta_model = 2.131 *  std_model / (SIGNIFICANCE_RUNS ** 0.5)

mean_base = data_base.mean(dim=0)
std_base = data_base.std(dim=0)
delta_base = 2.131 *  std_base / (SIGNIFICANCE_RUNS ** 0.5)

a = ax.plot(x, mean_model, label=f'GNN + ACO')
ax.scatter(mean_model.tolist().index(min(mean_model))/20, min(mean_model))

ax.fill_between(x, (mean_model-delta_model), (mean_model+delta_model), alpha=.2)

a = ax.plot(x, mean_base, label=f'Pure ACO')
ax.scatter(mean_base.tolist().index(min(mean_base))/20, min(mean_base), color='orange')

ax.fill_between(x, (mean_base-delta_base), (mean_base+delta_base), alpha=.2)

plt.xlabel('Evaporation rate')
plt.ylabel('Objective value')
plt.legend()
plt.title(f'Objective value as evarporation rate varies')
plt.show()
