import torch
import numpy as np
from aco import ACO
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import neuralnetwork
from tqdm import trange


def convert_to_pyg_format(nodes, distances,k_sparse=None):
    if k_sparse:
        # Update distances by setting non k closest connections to 0
        nearest_k_values, nearest_k_inds = torch.topk(distances, k_sparse, -1, largest=False)
        row_indices = torch.arange(distances.size(0)).unsqueeze(1).expand_as(nearest_k_inds)

        distances = torch.zeros_like(distances)
        distances[row_indices, nearest_k_inds] = nearest_k_values

    x = nodes
    edge_data = distances.to_sparse()
    edge_index = edge_data.indices()
    edge_attr = edge_data.values().reshape(-1, 1)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def generate_problem_instance(size, k_sparse=None):
    nodes = torch.rand(size=(size, 2))
    distances = torch.sqrt(((nodes[:, None] - nodes[None, :]) ** 2).sum(2))
    distances[torch.arange(size), torch.arange(size)] = 1e9
    pyg_data = convert_to_pyg_format(nodes, distances, k_sparse=k_sparse)
    return pyg_data, distances


def generateLoss(tour_costs, tour_log_probs):
    mean = tour_costs.mean()
    size = tour_costs.size()[0] # #ants
    loss = torch.sum((tour_costs - mean) * tour_log_probs.sum(dim=0)) / size
    return loss

def reshape_heuristic(heursitic_vector, pyg_data):
    problem_size = pyg_data.x.shape[0]
    heuristic_matrix = torch.full((problem_size, problem_size), 1e-10)

    heuristic_matrix[pyg_data.edge_index[0], pyg_data.edge_index[1]] = heursitic_vector.squeeze(dim = -1)
    return heuristic_matrix
    

def train_iteration(network, optimiser, instance_data, distances, n_ants, k_sparse=None):
    network.train()
    heuristic_vector = network(instance_data)
    heuristics = reshape_heuristic(heuristic_vector, instance_data)
    
    acoInstance = ACO(n_ants, distances, heuristics=heuristics)
    _, tour_costs, tour_log_probs = acoInstance.generate_paths_and_costs(gen_probs=True) # Ignore actual paths

    loss = generateLoss(tour_costs, tour_log_probs)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()


def train(network, problem_size, epochs, iterations_per_epoch, n_ants, k_sparse=None, lr=1e-4):
    optimiser = torch.optim.AdamW(network.parameters(), lr=lr)

    for epoch in range(epochs):
        for _ in (pbar := trange(iterations_per_epoch)):
            pyg_data, distances = generate_problem_instance(problem_size, k_sparse=k_sparse)
            train_iteration(network, optimiser, pyg_data, distances, n_ants, k_sparse=k_sparse)
            pbar.set_description(f'Epoch {epoch+1}')


heuristic_network = neuralnetwork.GNN(32, 20)
problem_size = 25
k_sparse = 10
epochs = 10
iterations_per_epoch = 100
n_ants = 15
train(heuristic_network, problem_size, epochs, iterations_per_epoch, n_ants, k_sparse=k_sparse)

heuristic_network.eval()
costs_base = []
costs_heu = []

for _ in range(10):
    data, distances = generate_problem_instance(problem_size)
    sim = ACO(n_ants, distances)
    sim.run(50)
    costs_base.append(sim.costs)

    heuristic_vector = heuristic_network(data)
    heuristics = reshape_heuristic(heuristic_vector, data)
    sim_heu = ACO(n_ants, distances, heuristics=heuristics)
    sim_heu.run(50)
    costs_heu.append(sim_heu.costs)

costs_base = np.column_stack(tuple(costs_base))
costs_heu = np.column_stack(tuple(costs_heu))

fig, ax = plt.subplots()
ax.plot(np.mean(costs_base, axis=1), label='Base')
ax.plot(np.mean(costs_heu, axis=1), label='Heu')

plt.xlabel('No. Iterations')
plt.ylabel('Path Length')
plt.legend()
plt.title(f'TSP {problem_size}')
plt.show()

#  # Extrapolate
# costs_base = []
# costs_heu = []

# for _ in range(10):
#     data, distances = generate_problem_instance(problem_size * 2)
#     sim = ACO(15, distances)
#     sim.run(50)
#     costs_base.append(sim.costs)

#     heuristic_vector = heuristic_network(data)
#     heuristics = reshape_heuristic(heuristic_vector, data)
#     sim_heu = ACO(15, distances, heuristics=heuristics)
#     sim_heu.run(50)
#     costs_heu.append(sim_heu.costs)

# costs_base = np.column_stack(tuple(costs_base))
# costs_heu = np.column_stack(tuple(costs_heu))

# fig, ax = plt.subplots()
# ax.plot(np.mean(costs_base, axis=1), label='Base')
# ax.plot(np.mean(costs_heu, axis=1), label='Heu')

# plt.xlabel('No. Iterations')
# plt.ylabel('Path Length')
# plt.legend()
# plt.title(f'TSP Extrapolated')
# plt.show()
