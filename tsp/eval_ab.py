import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_dataset
from aco import ACO
import graphnn.graphnn as graphnn
from traintsp import train_variable, validate_best_variable_vary_evap, train, validate_best, validate_best_variable_vary_ab



problem_size = 50
k_sparse = 50
epochs = 20
iterations_per_epoch = 1500
n_ants = 15

min_problem_size = 10
max_problem_size = 100

data_to_save = []

heuristic_network = graphnn.GNN(32, 12)

variable_values = [0.5, 0.9, 1, 1.5, 2]
if False:
    train_variable(heuristic_network, 20, 50, 1, iterations_per_epoch, n_ants, k_sparse=None)
    heuristic_network.eval()
    heuristic_network = None

    data = torch.zeros(size=(len(variable_values), len(variable_values)))
    for i, alpha in enumerate(variable_values):
        for j, beta in enumerate(variable_values):
            val_data = validate_best_variable_vary_ab(heuristic_network, min_problem_size, max_problem_size, n_ants, alpha, beta)
            data[i][j] = val_data
    torch.save(data, f'results/tsp/abheatmap_base.pt')

data_model = torch.load(f'results/tsp/abheatmap_model.pt')
data_base = torch.load(f'results/tsp/abheatmap_base.pt')

    
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8.5, 5))


for ax, data in zip(axes.flat, [data_base, data_model]):
    im = ax.imshow(data, cmap='PuBu', vmin=5, vmax=17)
    locs = np.arange(0, len(variable_values))
    ax.set(xticks=locs, xticklabels=variable_values, yticks=locs, yticklabels=variable_values)
    ax.set_title('Pure ACO' if (data == data_base).all() else 'GNN + ACO')
    ax.set_xlabel('β - Beta')
    ax.set_ylabel('α - Alpha')

fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.2, hspace=0.02)

cb_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])
cbar = fig.colorbar(im, cax=cb_ax)
plt.show()