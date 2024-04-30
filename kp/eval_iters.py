import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import matplotlib.pyplot as plt
from utils import load_dataset
from aco import ACO
from trainkp import train, get_distances, reshape_heuristic, convert_to_pyg_format
import graphnn.neuralnetwork as graphnn


problem_size = 100
k_sparse = None
epochs = 20
iterations_per_epoch = 100
n_ants = 20

costs_model = []
costs_model_new = []
costs_base = []
test_dataset = load_dataset('test', problem_size)
SIGNIFICANCE_RUNS = 15

if False:
    for _ in range(SIGNIFICANCE_RUNS):
        original_network = graphnn.GNN(32, 12)
        new_network = graphnn.GNN(32, 12, updated_layers=True)
        train(original_network, problem_size, epochs, iterations_per_epoch, n_ants, k_sparse=k_sparse)
        train(new_network, problem_size, epochs, iterations_per_epoch, n_ants, k_sparse=k_sparse)

        run_costs_model = []
        run_costs_model_new = []
        run_costs_base = []
        for data in test_dataset:
            weights, values = get_distances(data)
            pyg_data = convert_to_pyg_format(data, weights, values)

            sim = ACO(n_ants, weights, values)
            sim.run(50)
            run_costs_base.append(sim.costs)

            heuristic_vector = original_network(pyg_data)
            heuristics = reshape_heuristic(heuristic_vector, pyg_data)
            sim_heu = ACO(n_ants, weights, values, heuristics=heuristics)
            sim_heu.run(50)
            run_costs_model.append(sim_heu.costs)

            heuristic_vector = new_network(pyg_data)
            heuristics = reshape_heuristic(heuristic_vector, pyg_data)
            sim_heu_new = ACO(n_ants, weights, values, heuristics=heuristics)
            sim_heu_new.run(50)
            run_costs_model_new.append(sim_heu_new.costs)
        
        costs_model.append(torch.tensor(run_costs_model).mean(dim=0).tolist())
        costs_model_new.append(torch.tensor(run_costs_model_new).mean(dim=0).tolist())
        costs_base.append(torch.tensor(run_costs_base).mean(dim=0).tolist())

    torch.save(torch.tensor(costs_model), 'results/kp/run-model.pt')
    torch.save(torch.tensor(costs_model_new), 'results/kp/run-model-new.pt')
    torch.save(torch.tensor(costs_base), 'results/kp/run-base.pt')

data_model_new = torch.load('results/kp/run-model-new.pt')
data_model = torch.load('results/kp/run-model.pt')
data_base = torch.load('results/kp/run-base.pt')


fig, ax = plt.subplots()
x = [i for i in range(1, 51)]

for data, name in [(data_base, 'Expert heuristic'), (data_model_new, 'Updated architecture'), (data_model, 'Orginial architecture')]:
    mean = data.mean(dim=0)
    std = data.std(dim=0)
    delta = 2.131 * std / (SIGNIFICANCE_RUNS ** 0.5)
    ax.plot(x, mean, label=name)
    ax.fill_between(x, (mean-delta), (mean+delta), alpha=.5)


plt.xlabel('ACO iterations')
plt.ylabel('Objective value')
plt.legend()
plt.title(f'Objective value against ACO rounds for KP')
plt.show()
