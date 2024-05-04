import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_dataset
from aco import ACO
import graphnn.graphnn as graphnn
from traintsp import train_variable, validate_best_variable_vary_evap, train, validate_best, validate_best_variable_vary_ab, validate_best_variable

problem_size = 50
k_sparse = 50
epochs = 20
iterations_per_epoch = 1500
n_ants = 15

min_problem_size = 10
max_problem_size = 100

SIGNIFICANCE_RUNS=15
STEPS=20
if False:
    data_to_save = []

    for problem_size in [10, 20, 50, 100]:
        k_sparse = problem_size
        data_for_model = []
        # Number of runs of each model for statistical significance
        
        heuristic_network = graphnn.GNN(32, 12)

        train(heuristic_network, problem_size, 1, iterations_per_epoch, n_ants, k_sparse=None)
        heuristic_network.eval()
        val_data = validate_best_variable(heuristic_network, min_problem_size, max_problem_size, n_ants, avg=False)

        print(f'Completed run for model size {problem_size}')
        val_data = torch.tensor(val_data)
        data_to_save.append(val_data)
        
    torch.save(torch.stack(data_to_save), f'results/tsp/10-100models_multi.pt')
    data_to_save = []

    for min_s, max_s in zip([20, 10], [50, 100]):
        k_sparse = problem_size
        data_for_model = []
        
        heuristic_network = graphnn.GNN(32, 12)

        train_variable(heuristic_network, min_s, max_s, 1, iterations_per_epoch, n_ants, k_sparse=None)
        heuristic_network.eval()
        val_data = validate_best_variable(heuristic_network, min_problem_size, max_problem_size, n_ants, avg=False)

        print(f'Completed run for model size {min_s}-{max_s}')
        val_data = torch.tensor(val_data)
        data_to_save.append(val_data)
        
    torch.save(torch.stack(data_to_save), f'results/tsp/10-100models_multi.pt')

    base_results = validate_best_variable(None, min_problem_size, max_problem_size, n_ants, avg=False)
    base_data = torch.tensor(base_results)
    torch.save(base_data, f'results/tsp/10-100base.pt')

OPTIMAL10_100_50 = torch.load('results/tsp/models_solutions_2.pt')
fig, ax = plt.subplots()
x = [i for i in range(min_problem_size, max_problem_size+5, 5)]
for data, name in zip(torch.load('results/tsp/10-100models_2.pt'), ['10', '20', '50', '100']):

    data = (data / OPTIMAL10_100_50 - 1) * 100
    data = data.reshape((-1, 50))
    mean = data.mean(dim=1)
    std = data.std(dim=1)
    delta = 2.01 * std / (50 ** 0.5)
    a = ax.plot(x, mean, label=f'TSP{name}')
    ax.fill_between(x, (mean-delta), (mean+delta), alpha=.1)

for data, name in zip(torch.load('results/tsp/10-100models_multi.pt'), ['20-50', '10-100']):

    data = (data / OPTIMAL10_100_50 - 1) * 100
    data = data.reshape((-1, 50))
    mean = data.mean(dim=1)
    std = data.std(dim=1)
    delta = 2.01 * std / (50 ** 0.5)
    a = ax.plot(x, mean, label=f'TSP{name}')
    ax.fill_between(x, (mean-delta), (mean+delta), alpha=.1)


simple = torch.load('results/tsp/10-100base.pt')

data = (simple / OPTIMAL10_100_50 - 1) * 100
data = data.reshape((-1, 50))
mean = data.mean(dim=1)
std = data.std(dim=1)

delta = 2.01 * std / (50 ** 0.5)
a = ax.plot(x, mean, label=f'Expert Heuristic')
ax.fill_between(x, (mean-delta), (mean+delta), alpha=.1)


plt.xlabel('Problem size')
plt.ylabel('Optimality gap')
plt.legend()
plt.title(f'Optimality gap for different sized test instances')
plt.show()

# Plot comparing the variably trained models
fig, ax = plt.subplots()
for data, name in zip(torch.load('results/tsp/10-100models_multi.pt'), ['20-50', '10-100']):

    data = (data / OPTIMAL10_100_50 - 1) * 100
    data = data.reshape((-1, 50))
    mean = data.mean(dim=1)
    std = data.std(dim=1)

    
    delta = 2.01 * std / (50 ** 0.5)

    a = ax.plot(x, mean, label=f'TSP{name}')

    ax.fill_between(x, (mean-delta), (mean+delta), alpha=.1)
plt.xlabel('Problem size')
plt.ylabel('Optimality gap')
plt.legend()
plt.title(f'Optimality gap for different sized test instances')
plt.show()



# Plot comparing tsp10 and tsp100 models
fig, ax = plt.subplots()
for data, name in zip(torch.load('results/tsp/10-100models_2.pt'), ['10', '20', '50', '100']):
    if name in ['20', '50']:
        continue
    data = (data / OPTIMAL10_100_50 - 1) * 100
    data = data.reshape((-1, 50))
    mean = data.mean(dim=1)
    std = data.std(dim=1)

    
    delta = 2.01 * std / (50 ** 0.5)

    a = ax.plot(x, mean, label=f'TSP{name}')

    ax.fill_between(x, (mean-delta), (mean+delta), alpha=.1)
plt.xlabel('Problem size')
plt.ylabel('Optimality gap')
plt.legend()
plt.title(f'Optimality gap for different sized test instances')
plt.show()
