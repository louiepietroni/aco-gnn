import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_dataset
from aco import ACO
import graphnn.neuralnetwork as graphnn
from traintsp import train_variable, validate_best_variable_vary_evap, train, validate_best


problem_size = 50
k_sparse = 50
epochs = 20
iterations_per_epoch = 100
n_ants = 15

min_problem_size = 10
max_problem_size = 100

# Train up to 2000 total iterations to see convergence

SIGNIFICANCE_RUNS=15
STEPS=20
all_models_data = []
for problem_size in [10, 20, 50, 100]:
    continue
    k_sparse = None
    data_for_model = []
    
    for model_number in range(SIGNIFICANCE_RUNS):
        heuristic_network = graphnn.GNN(32, 12)
        data_for_run = []

        for iterations in range(STEPS):
            # print(iterations)
            train(heuristic_network, problem_size, 1, iterations_per_epoch, n_ants, k_sparse=k_sparse)
            # train_variable(heuristic_network, min_problem_size, max_problem_size, 1, iterations_per_epoch, n_ants)
            heuristic_network.eval()
            avg_over_val_data = validate_best(heuristic_network, problem_size, n_ants, k_sparse)
            # avg_over_val_data = validate_best_variable(heuristic_network, min_problem_size, max_problem_size, n_ants)
            data_for_run.append(avg_over_val_data)
        data_for_model.append(data_for_run)
        print(f'Completed run {model_number}/{SIGNIFICANCE_RUNS} for problem size {problem_size}')
    tensor_data_for_model = torch.tensor(data_for_model)
    print(f'DATA FOR {problem_size}, {tensor_data_for_model.size()}')
    mean = tensor_data_for_model.mean(dim=0).tolist()
    std = tensor_data_for_model.std(dim=0).tolist()
    all_models_data.append([mean, std])

# torch.save(all_models_data, 'results/tsp/training_sizes.pt')


fig, ax = plt.subplots()
x = [(i+1)*iterations_per_epoch for i in range(STEPS)]

all_sizes_data = torch.load('results/tsp/training_sizes.pt')
names = ['10', '20', '50', '100', '20-50', '10-100']
i = 0
for data in all_sizes_data:
    mean = data[0]
    std = data[1]

    delta = 2.131 * std / (SIGNIFICANCE_RUNS ** 0.5)
    a = ax.plot(x, mean, label=f'TSP{names[i]}')
    ax.fill_between(x, (mean-delta), (mean+delta), alpha=.4)
    i += 1

plt.xlabel('Training iterations')
plt.ylabel('Objective value')
plt.legend()
plt.title(f'Objective value during training')
plt.show()



