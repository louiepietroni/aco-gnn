import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_dataset
from aco import ACO
import graphnn.neuralnetwork as graphnn
from traintsp import train_variable, validate_each_iteration



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
data_base = []
data_model = []
for _ in range(SIGNIFICANCE_RUNS):
    continue
    val_data = validate_each_iteration(None, min_problem_size, max_problem_size, n_ants, avg=True, k_sparse=None)
    data_base.append(val_data)

    heuristic_network = graphnn.GNN(32, 12)
    train_variable(heuristic_network, min_problem_size, max_problem_size, 1, iterations_per_epoch, n_ants, k_sparse=None)
    heuristic_network.eval()

    val_data = validate_each_iteration(heuristic_network, min_problem_size, max_problem_size, n_ants, avg=True, k_sparse=None)
    data_model.append(val_data)

    print(f'Completed significane run {_+1}/{SIGNIFICANCE_RUNS}')


# torch.save(torch.stack(data_base), f'results/tsp/ls-base.pt')
# torch.save(torch.stack(data_model), f'results/tsp/ls-model.pt')

sub = 0

data_base = torch.load('results/tsp/ls-base.pt')[:, sub:]
data_model = torch.load('results/tsp/ls-model.pt')[:, sub:]

fig, ax = plt.subplots()
x = [i for i in range(1, 101 + 20)][sub:]

mean = data_base.mean(dim=0)
std = data_base.std(dim=0)
delta = 2.131 * std / (15 ** 0.5)
ax.plot(x, mean, label=f'Pure ACO')
ax.fill_between(x, (mean-delta), (mean+delta), alpha=.4)

m = mean

mean = data_model.mean(dim=0)
std = data_model.std(dim=0)
delta = 2.131 * std / (15 ** 0.5)
ax.plot(x, mean, label=f'GNN+ACO')
ax.fill_between(x, (mean-delta), (mean+delta), alpha=.4)

plt.axvspan(0, 100, facecolor='green', alpha=0.15, label='ACO rounds')
plt.axvspan(100, 120, facecolor='red', alpha=0.15, label='LS rounds')

plt.xlabel('Rounds of solution search')
plt.ylabel('Objective value')
plt.title(f'Objective value during ACO and Local search')
plt.legend()
plt.show()





sub = 99
data_base = torch.load('results/tsp/ls-base.pt')[:, sub:]
data_model = torch.load('results/tsp/ls-model.pt')[:, sub:]

fig, ax = plt.subplots()
x = [i for i in range(0, 21)]
data_base = 1-(data_base / data_base[:, 0].expand(21, -1).t())
data_base *= 100

mean = data_base.mean(dim=0)
ax.axhline(mean.max(), color='lightblue', linestyle='--', alpha=1, zorder=1)

std = data_base.std(dim=0)
delta = 2.131 * std / (15 ** 0.5)
ax.plot(x, mean, label=f'Pure ACO')
ax.fill_between(x, (mean-delta), (mean+delta), alpha=.3)


data_model = 1-(data_model / data_model[:, 0].expand(21, -1).t())
data_model *= 100

mean = data_model.mean(dim=0)

std = data_model.std(dim=0)
delta = 2.131 * std / (15 ** 0.5)
ax.plot(x, mean, label=f'GNN+ACO')
ax.fill_between(x, (mean-delta), (mean+delta), alpha=.3)

ax.axhline(mean.max(), color='orange', linestyle='--', alpha=.5)
ax.set_xticks(np.linspace(0, 20, 11))
plt.xlabel('Rounds of local search')
plt.ylabel('% Improvement on best ACO solution')
plt.title(f'% Improvement against ACO solution during Local search')
plt.legend()
plt.show()
