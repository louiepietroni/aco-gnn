import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import matplotlib.pyplot as plt
from utils import load_dataset, generate_variable_dataset
from trainsop import train_variable, validate_best_variable, train, validate_dataset
from aco import ACO
import graphnn.neuralnetwork as graphnn


           

SIGNIFICANCE_RUNS=15
problem_size = 100
k_sparse = 10
epochs = 10
n_ants = 15
iterations_per_epoch = 1000
min_problem_size = 10
max_problem_size = 100
INSTANCES = 20
test_dataset = generate_variable_dataset('test', min_problem_size, max_problem_size, INSTANCES)
if True:
    for problem_size in [20, 50, 100]:
        data_for_model = []
        for _ in range(SIGNIFICANCE_RUNS):
            heuristic_network = graphnn.GNN(32, 12)
            train(heuristic_network, problem_size, 1, iterations_per_epoch, n_ants)
            heuristic_network.eval()
            avg_over_val_data = validate_dataset(heuristic_network, n_ants, test_dataset, avg=False)
            data_for_model.append(avg_over_val_data)
            print(f'Finished run {_+1}/{SIGNIFICANCE_RUNS} for SOP{problem_size}')
        torch.save(torch.tensor(data_for_model), f'results/sop/eval-sizes-{problem_size}.pt')
        

    for min, max in [(20, 50), (10, 100)]:

        data_for_model = []
        for _ in range(SIGNIFICANCE_RUNS):
            heuristic_network = graphnn.GNN(32, 12)
            train_variable(heuristic_network, min, max, 1, iterations_per_epoch, n_ants)
            heuristic_network.eval()
            avg_over_val_data = validate_dataset(heuristic_network, n_ants, test_dataset, avg=False)
            data_for_model.append(avg_over_val_data)
            print(f'Finished run {_+1}/{SIGNIFICANCE_RUNS} for SOP{min}-{max}')
        torch.save(torch.tensor(data_for_model), f'results/sop/eval-sizes-{min}-{max}.pt')

    data_for_model = []
    for _ in range(SIGNIFICANCE_RUNS):
        heuristic_network = graphnn.GNN(32, 12)
        avg_over_val_data = validate_dataset(heuristic_network, n_ants, test_dataset, avg=False)
        data_for_model.append(avg_over_val_data)
        print(f'Finished run {_+1}/{SIGNIFICANCE_RUNS} for base')
    torch.save(torch.tensor(data_for_model), f'results/sop/eval-sizes-base.pt')

data_20 = torch.load('results/sop/eval-sizes-20.pt')
data_50 = torch.load('results/sop/eval-sizes-50.pt')
data_100 = torch.load('results/sop/eval-sizes-100.pt')
data_20_50 = torch.load('results/sop/eval-sizes-20-50.pt')
data_10_100 = torch.load('results/sop/eval-sizes-10-100.pt')
data_base = torch.load('results/sop/eval-sizes-base.pt')


fig, ax = plt.subplots()
x = [i for i in range(min_problem_size, max_problem_size+5, 5)]

for data, name in [(data_20, '20'), (data_50, '50'), (data_100, '100'), (data_20_50, '20-50'), (data_10_100, '10-100')]:
    data = data.t().reshape((-1, 15*20))
    mean = data.mean(dim=1)
    std = data.std(dim=1)
    delta = 1.984 * std / (SIGNIFICANCE_RUNS*INSTANCES ** 0.5)
    ax.plot(x, mean, label=f'SOP{name}')
    ax.fill_between(x, (mean-delta), (mean+delta), alpha=.2)

data = data_base
data = data.t().reshape((-1, 15*20))
mean = data.mean(dim=1)
std = data.std(dim=1)
delta = 1.984 * std / (SIGNIFICANCE_RUNS*INSTANCES ** 0.5)
ax.plot(x, mean, label=f'Expert heuristic')
ax.fill_between(x, (mean-delta), (mean+delta), alpha=.2)

plt.xlabel('Problem size')
plt.ylabel('Objective value')
plt.legend()
plt.title(f'Objective value against problem sizes')
plt.show()
