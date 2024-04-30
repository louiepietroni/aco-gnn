import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import matplotlib.pyplot as plt
from utils import load_dataset
from trainbpp import train, get_distances, reshape_heuristic, convert_to_pyg_format
from aco import ACO
import graphnn.neuralnetwork as graphnn

test_dataset = load_dataset('test', 25)
SIGNIFICANCE_RUNS=15

problem_size = 100
k_sparse = None
epochs = 5
iterations_per_epoch = 100
n_ants = 15

costs_base = []
costs_model = []

costs = []
test_dataset = load_dataset('test', 100)
for _ in range(SIGNIFICANCE_RUNS):
    heuristic_network = graphnn.GNN(32, 12, node_features=1)
    train(heuristic_network, problem_size, epochs, iterations_per_epoch, n_ants, k_sparse=k_sparse)

    run_costs_base = []
    run_costs_model = []

    for sizes in test_dataset:
        distances = get_distances(sizes)
        pyg_data = convert_to_pyg_format(sizes, distances)

        sim = ACO(n_ants, distances, sizes)
        sim.run(100)
        run_costs_base.append(sim.costs)


        heuristic_vector = heuristic_network(pyg_data)
        heuristics = reshape_heuristic(heuristic_vector, pyg_data)
        sim_heu = ACO(n_ants, distances, sizes, heuristics=heuristics)
        sim_heu.run(100)
        run_costs_model.append(sim_heu.costs)

    costs_base.append(torch.tensor(run_costs_base).mean(dim=0).tolist())
    costs_model.append(torch.tensor(run_costs_model).mean(dim=0).tolist())

# torch.save(torch.tensor(costs_model), 'results/bpp/run-model.pt')
# torch.save(torch.tensor(costs_base), 'results/bpp/run-base.pt')

data_model = torch.load('results/bpp/run-model.pt')
data_base = torch.load('results/bpp/run-base.pt')


fig, ax = plt.subplots()
x = [i for i in range(1, 101)]

mean = data_base.mean(dim=0)
std = data_base.std(dim=0)
delta = 2.131 * std / (SIGNIFICANCE_RUNS ** 0.5)
ax.plot(x, mean, label=f'Pure ACO')
ax.fill_between(x, (mean-delta), (mean+delta), alpha=.2)
print(f'Base {mean}')

mean = data_model.mean(dim=0)
std = data_model.std(dim=0)
delta = 2.131 * std / (SIGNIFICANCE_RUNS ** 0.5)
ax.plot(x, mean, label=f'GNN + ACO')
ax.fill_between(x, (mean-delta), (mean+delta), alpha=.2)
print(f'Model {mean}')

plt.xlabel('Rounds of solution search')
plt.ylabel('Objective value')
plt.title(f'Objective value during rounds of ACO solution construction')
plt.legend()
plt.show()