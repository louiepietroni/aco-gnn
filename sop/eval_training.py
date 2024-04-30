import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import matplotlib.pyplot as plt
from utils import load_dataset, generate_variable_dataset
from trainsop import train_variable, validate_best_variable, validate_dataset
from aco import ACO
import graphnn.neuralnetwork as graphnn





SIGNIFICANCE_RUNS=15
STEPS=10
problem_size = 100
k_sparse = 10
epochs = 10
n_ants = 15
iterations_per_epoch = 100
min_problem_size = 10
max_problem_size = 100
test_dataset = generate_variable_dataset('test', min_problem_size, max_problem_size, 25)

data = []

for _ in range(SIGNIFICANCE_RUNS):
    heuristic_network = graphnn.GNN(32, 12)
    data_for_run = []
    for iterations in range(STEPS):
        # train(heuristic_network, problem_size, 1, iterations_per_epoch, n_ants)

        train_variable(heuristic_network, 20, 50, 1, iterations_per_epoch, n_ants)
        heuristic_network.eval()
        avg_over_val_data = validate_dataset(heuristic_network, n_ants, test_dataset)
        data_for_run.append(avg_over_val_data)
        print(f'Finished step {iterations+1}/{STEPS}')
    data.append(data_for_run)
    print(f'Finished for {_+1}/{SIGNIFICANCE_RUNS}')

# torch.save(torch.tensor(data), 'results/sop/train-iters-20-50.pt')

data_50 = torch.load('results/sop/train-iters-50.pt')
data_100 = torch.load('results/sop/train-iters-100.pt')
data_20_50 = torch.load('results/sop/train-iters-20-50.pt')
data_10_100 = torch.load('results/sop/train-iters-10-100.pt')

fig, ax = plt.subplots()
x = [(i+1)*iterations_per_epoch for i in range(10)]
for data, name in [(data_50, 'SOP50'), (data_100, 'SOP100'), (data_20_50, 'SOP20-50'), (data_10_100, 'SOP10-100')]:
    mean = data.mean(dim=0)
    std = data.std(dim=0)
    delta = 2.131 * std / (SIGNIFICANCE_RUNS ** 0.5)
    ax.plot(x, mean, label=name)
    ax.fill_between(x, (mean-delta), (mean+delta), alpha=.2)

plt.xlabel('Training iterations')
plt.ylabel('Objective value')
plt.legend()
plt.title(f'Objective value during training')
plt.show()


