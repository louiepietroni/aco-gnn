import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import graphnn.neuralnetwork as graphnn
import torch
import matplotlib.pyplot as plt
from utils import generate_variable_dataset
from traincvrp import train_variable, validate_dataset

costs = []
SIGNIFICANCE_RUNS = 15
min_problem_size = 20
max_problem_size = 50
epochs = 10
iterations_per_epoch = 1500
n_ants = 15

min_cap = 10
max_cap = 50

if True:
    test_dataset = generate_variable_dataset('test', min_problem_size, max_problem_size, 45)
    order = torch.randperm(len(test_dataset))
    test_dataset = [test_dataset[i] for i in order]
    test_capacities = [[x for _ in range(35)] for x in range(min_cap, max_cap + 5, 5)]
    test_capacities = torch.tensor(test_capacities).flatten().tolist()
    print('starting')

    heuristic_network = graphnn.GNN(32, 12, node_features=1)
    train_variable(heuristic_network, min_problem_size, max_problem_size, 1, iterations_per_epoch, n_ants)
    print('trained 30')
    heuristic_network.eval()
    data_30 = validate_dataset(heuristic_network, test_dataset, test_capacities, n_ants, avg=False)
    print('evaluated 30')
    torch.save(torch.tensor(data_30), f'results/cvrp/caps-30.pt')

    heuristic_network = graphnn.GNN(32, 12, node_features=1)
    train_variable(heuristic_network, min_problem_size, max_problem_size, 1, iterations_per_epoch, n_ants, min_cap=10, max_cap=50)
    print('trained 10-50')
    heuristic_network.eval()
    data_10_50 = validate_dataset(heuristic_network, test_dataset, test_capacities, n_ants, avg=False)
    print('evaluated 10-50')
    torch.save(torch.tensor(data_10_50), f'results/cvrp/caps-10-50.pt')


    data_base = validate_dataset(None, test_dataset, test_capacities, n_ants, avg=False)
    print('evaluated base')
    torch.save(torch.tensor(data_base), f'results/cvrp/caps-base.pt')


data_30 = torch.load('results/cvrp/caps-30.pt')
data_10_100 = torch.load('results/cvrp/caps-10-50.pt')
data_base = torch.load('results/cvrp/caps-base.pt')

fig, ax = plt.subplots()
x = [i for i in range(min_cap, max_cap+5, 5)]

for data, name in [(data_30, 'Model trained with capacity 30'), (data_10_100, 'Model trained with capacity 10-50'), (data_base, 'Expert heuristic')]:
    data = data.reshape((-1, 35))
    mean = data.mean(dim=1)
    std = data.std(dim=1)
    delta = 2.021 * std / (35 ** 0.5)
    ax.plot(x, mean, label=f'{name}')
    ax.fill_between(x, (mean-delta), (mean+delta), alpha=.2)


plt.xlabel('Vehicle Capacity')
plt.ylabel('Objective value')
plt.legend()
plt.title(f'Objective value against vehicle capacities')
plt.show()