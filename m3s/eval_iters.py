import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import matplotlib.pyplot as plt
from aco import ACO
from utils import load_dataset
from trainm3s import train, get_distances, convert_to_pyg_format, reshape_heuristic
import graphnn.graphnn as graphnn


costs_base = []
costs_model = []
SIGNIFICANCE_RUNS=15

problem_size = 50
k_sparse = None
epochs = 1
iterations_per_epoch = 700
n_ants = 15
test_dataset = load_dataset('test', problem_size)


for _ in range(SIGNIFICANCE_RUNS):
    continue
    heuristic_network = graphnn.GNN(32, 12, node_features=1)
    train(heuristic_network, problem_size, epochs, iterations_per_epoch, n_ants, k_sparse=k_sparse)

    run_costs_base = []
    run_costs_model = []

    fail = False
    for clauses in test_dataset:
        distances = get_distances(clauses, problem_size)
        pyg_data = convert_to_pyg_format(clauses, distances)

        sim = ACO(n_ants, distances, clauses)
        sim.run(750)

        heuristic_vector = heuristic_network(pyg_data)
        heuristics = reshape_heuristic(heuristic_vector, pyg_data)

        sim_heu = ACO(n_ants, distances, clauses, heuristics=heuristics)
        sim_heu.run(750)

        run_costs_base.append(sim.costs)
        run_costs_model.append(sim_heu.costs)


    costs_base.append(torch.tensor(run_costs_base).mean(dim=0).tolist())
    costs_model.append(torch.tensor(run_costs_model).mean(dim=0).tolist())

# torch.save(torch.tensor(costs_model), 'results/m3s/new-run-model-old.pt')
# torch.save(torch.tensor(costs_base), 'results/m3s/new-run-base.pt')

data_model = torch.load('results/m3s/new-run-model.pt')
data_model_old = torch.load('results/m3s/new-run-model-old.pt')
data_base = torch.load('results/m3s/new-run-base.pt')


fig, ax = plt.subplots()
x = [i for i in range(1, 751)]

for data, name in [(data_model, 'Updated architecture'), (data_model_old, 'Original architecture'), (data_base, 'Pure ACO')]:
    mean = data.mean(dim=0)
    std = data.std(dim=0)
    delta = 2.131 * std / (SIGNIFICANCE_RUNS ** 0.5)
    ax.plot(x, mean, label=name)
    ax.fill_between(x, (mean-delta), (mean+delta), alpha=.2)

plt.xlabel('Rounds of solution search')
plt.ylabel('Objective value')
plt.title(f'Objective value during rounds of ACO solution construction')
plt.legend()
plt.show()