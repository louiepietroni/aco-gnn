import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_dataset, load_variable_dataset, load_variable_sols
from aco import ACO
import graphnn.neuralnetwork as graphnn
from traintsp import train_variable, validate_best_variable, train_dataset


problem_size = 50
k_sparse = 50
iterations_per_epoch = 200
n_ants = 15

min_problem_size = 20
max_problem_size = 50

SIGNIFICANCE_RUNS=15
STEPS=20

sup_data = []
sup_data_limited = []
reinf_data = []

for _ in range(SIGNIFICANCE_RUNS):
    data_for_run_reinf = []
    data_for_run_sup =  []
    data_for_run_sup_limited = []

    reinf_network = graphnn.GNN(32, 12)
    sup_network = graphnn.GNN(32, 12)
    sup_limited_network = graphnn.GNN(32, 12)

    dataset = load_variable_dataset('train', 20, 50, quantity=1000)
    sols = load_variable_sols('train-sols', 20, 50)
    order = torch.randperm(len(dataset))
    dataset = [dataset[i] for i in order]
    sols = sols[order]

    for iterations in range(STEPS):
        train_variable(reinf_network, min_problem_size, max_problem_size, 1, iterations_per_epoch, n_ants)
        train_dataset(sup_network, dataset, sols, iterations_per_epoch, n_ants, start=iterations*iterations_per_epoch)
        train_dataset(sup_limited_network, dataset, sols, iterations_per_epoch, n_ants, start=0)
        avg_over_val_data = validate_best_variable(reinf_network, min_problem_size, max_problem_size, n_ants)
        data_for_run_reinf.append(avg_over_val_data)

        avg_over_val_data = validate_best_variable(sup_network, min_problem_size, max_problem_size, n_ants)
        data_for_run_sup.append(avg_over_val_data)

        avg_over_val_data = validate_best_variable(sup_limited_network, min_problem_size, max_problem_size, n_ants)
        data_for_run_sup_limited.append(avg_over_val_data)
    


    reinf_data.append(data_for_run_reinf)
    sup_data.append(data_for_run_sup)
    sup_data_limited.append(data_for_run_sup_limited)
    print(f'Finished sup limited for {_+1}/{SIGNIFICANCE_RUNS}')

# torch.save(torch.tensor(reinf_data), 'results/tsp/sup-reinf-reinf.pt')
# torch.save(torch.tensor(sup_data), 'results/tsp/sup-reinf-sup.pt')
# torch.save(torch.tensor(sup_data_limited), 'results/tsp/sup-reinf-sup-limited.pt')

data_reinf = torch.load('results/tsp/sup-reinf-reinf.pt')
data_sup = torch.load('results/tsp/sup-reinf-sup.pt')
data_sup_limited = torch.load('results/tsp/sup-reinf-sup-limited.pt')

fig, ax = plt.subplots()
x = [(i+1)*iterations_per_epoch for i in range(15)][:8]

mean = data_reinf.mean(dim=0)[:8]
std = data_reinf.std(dim=0)[:8]
delta = 2.131 * std / (15 ** 0.5)
ax.axhline(mean.min(), linestyle='--', alpha=.5, color='black')
ax.plot(x, mean, label=f'Reinforcement Learning')
ax.fill_between(x, (mean-delta), (mean+delta), alpha=.2)

print(f'Reinf min: {mean.min()}')

x = [(i+1)*iterations_per_epoch for i in range(STEPS)]

mean = data_sup.mean(dim=0)
std = data_sup.std(dim=0)
delta = 2.131 * std / (15 ** 0.5)
ax.plot(x, mean, label=f'Supervised Learning')
ax.fill_between(x, (mean-delta), (mean+delta), alpha=.2)
print(f'Sup min: {mean.min()}')

mean = data_sup_limited.mean(dim=0)
std = data_sup_limited.std(dim=0)
delta = 2.131 * std / (15 ** 0.5)
ax.plot(x, mean, label=f'Supervised Learning limited instances')
ax.fill_between(x, (mean-delta), (mean+delta), alpha=.2)
print(f'Sup limit min: {mean.min()}')


plt.xlabel('Training iterations')
plt.ylabel('Objective value')
plt.legend()
plt.title(f'Objective value during training')
plt.show()


