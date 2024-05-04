import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import matplotlib.pylab as plt
import torch
from utils import generate_variable_dataset, solve_dataset
from trainap import train, train_variable, validate_dataset
import graphnn.graphnn as graphnn


costs = []
min_problem_size = 10
max_problem_size = 100
epochs = 10
iterations_per_epoch = 1000
n_ants = 15

min_cap = 10
max_cap = 50
if False:
    test_dataset = generate_variable_dataset('test', min_problem_size, max_problem_size, 50)
    sols = solve_dataset(test_dataset)

    for size in [20, 50, 100]:
        heuristic_network = graphnn.GNN(32, 12, node_features=1)
        train(heuristic_network, size, epochs, iterations_per_epoch, n_ants)
        heuristic_network.eval()

        data = validate_dataset(heuristic_network, n_ants, test_dataset, sols,  avg=False)
        torch.save(torch.tensor(data), f'results/ap/sizes-{size}.pt')

    heuristic_network = graphnn.GNN(32, 12, node_features=1)
    train_variable(heuristic_network, 20, 50, epochs, iterations_per_epoch, n_ants)
    heuristic_network.eval()

    data_20_50 = validate_dataset(heuristic_network, n_ants, test_dataset, sols,  avg=False)
    torch.save(torch.tensor(data_20_50), 'results/ap/sizes-20-50.pt')

    data_base = validate_dataset(heuristic_network, n_ants, test_dataset, sols,  avg=False)
    torch.save(torch.tensor(data_base), 'results/ap/sizes-base.pt')


data_20 = torch.load('results/ap/sizes-20.pt')
data_50 = torch.load('results/ap/sizes-50.pt')
data_100 = torch.load('results/ap/sizes-100.pt')
data_20_50 = torch.load('results/ap/sizes-20-50.pt')
data_base = torch.load('results/ap/sizes-base.pt')

fig, ax = plt.subplots()
x = [i for i in range(min_problem_size, max_problem_size+5, 5)]

for data, name in [(data_20, 'AP20'), (data_50, 'AP50'), (data_100, 'AP100'), (data_20_50, 'AP20-50'), (data_base, 'Expert heuristic')]:
    data = (data - 1) * 100
    data = data.reshape((-1, 50))
    mean = data.mean(dim=1)
    std = data.std(dim=1)
    delta = 2.021 * std / (50 ** 0.5)
    ax.plot(x, mean, label=f'{name}')
    ax.fill_between(x, (mean-delta), (mean+delta), alpha=.2)

plt.xlabel('Problem size')
plt.ylabel('Optimality gap %')
plt.legend()
plt.title(f'Optimality gap against problem sizes')
plt.show()
