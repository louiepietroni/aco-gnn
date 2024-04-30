import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import matplotlib.pyplot as plt
from aco import ACO
from utils import load_dataset
from trainvcp import train, get_distances, convert_to_pyg_format, reshape_heuristic
import graphnn.neuralnetwork as graphnn


problem_size = 30
k_sparse = None
epochs = 3
iterations_per_epoch = 100
n_ants = 15


costs_model = []
costs_base = []
test_dataset = load_dataset('test', problem_size)
SIGNIFICANCE_RUNS = 15
for _ in range(SIGNIFICANCE_RUNS):
    heuristic_network = graphnn.GNN(32, 12, node_features=1)
    train(heuristic_network, problem_size, epochs, iterations_per_epoch, n_ants, k_sparse=k_sparse)
    heuristic_network.eval()
    run_costs_model = []
    run_costs_base = []
    for edges in test_dataset:
        distances = get_distances(edges, problem_size)
        pyg_data = convert_to_pyg_format(distances)

        sim = ACO(n_ants, distances, edges)
        sim.run(150)
        run_costs_base.append(sim.costs)

        heuristic_vector = heuristic_network(pyg_data)
        heuristics = reshape_heuristic(heuristic_vector, pyg_data)
        sim_heu = ACO(n_ants, distances, edges, heuristics=heuristics)
        sim_heu.run(150)
        run_costs_model.append(sim_heu.costs)
    
    costs_model.append(torch.tensor(run_costs_model).mean(dim=0).tolist())
    costs_base.append(torch.tensor(run_costs_base).mean(dim=0).tolist())

# torch.save(torch.tensor(costs_model), 'results/vcp/run-model.pt')
# torch.save(torch.tensor(costs_base), 'results/vcp/run-base.pt')


data_model = torch.load('results/vcp/run-model.pt')[:, :75]
data_base = torch.load('results/vcp/run-base.pt')[:, :75]



fig, ax = plt.subplots()
x = [i for i in range(1, 76)]

mean = data_base.mean(dim=0)
std = data_base.std(dim=0)
delta = 2.131 * std / (SIGNIFICANCE_RUNS ** 0.5)
ax.plot(x, mean, label=f'Expert heuristic')
ax.fill_between(x, (mean-delta), (mean+delta), alpha=.2)
print(f'Updated architecture {mean}')

mean = data_model.mean(dim=0)
std = data_model.std(dim=0)
delta = 2.131 * std / (SIGNIFICANCE_RUNS ** 0.5)
ax.plot(x, mean, label=f'GNN + ACO')
ax.fill_between(x, (mean-delta), (mean+delta), alpha=.2)
print(f'Updated architecture {mean}')


plt.xlabel('ACO iterations')
plt.ylabel('Objective value')
plt.legend()
plt.title(f'Objective value against ACO rounds for VCP')
plt.show()
