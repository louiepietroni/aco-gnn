import torch
import numpy as np
from aco import ACO
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import neuralnetwork
from tqdm import trange


def convert_to_pyg_format(nodes, distances):
    x = nodes
    edge_data = distances.to_sparse()
    edge_index = edge_data.indices()
    edge_attr = edge_data.values().reshape(-1, 1)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def generate_problem_instance(size):
    nodes = torch.rand(size=(size, 2))
    distances = torch.sqrt(((nodes[:, None] - nodes[None, :]) ** 2).sum(2))
    distances[torch.arange(size), torch.arange(size)] = 1e9
    pyg_data = convert_to_pyg_format(nodes, distances)
    return pyg_data, distances

# def example(size):
#     pyg, dis, nodes = generate_problem_instance(size)
#     pyg_old, d_old = gen_pyg_data(nodes, size)
#     print(pyg)
#     print('XXX')
#     print(pyg_old)

# example(3)
# exit()

def generateLoss(tour_costs, tour_log_probs):
    simulation_size = tour_costs.size()[0]
    mean = tour_costs.mean()
    loss = torch.sum((tour_costs - mean) * tour_log_probs) / simulation_size
    return loss

def reshape_heuristic(heursitic_vector, pyg_data):
    problem_size = pyg_data.x.shape[0]
    heuristic_matrix = torch.full((problem_size, problem_size), 1e-10)

    heuristic_matrix[pyg_data.edge_index[0], pyg_data.edge_index[1]] = heursitic_vector.squeeze(dim = -1)
    return heuristic_matrix

    

def train_iteration(network, optimiser, instance_data, distances):
    network.train()
    heuristic_vector = network(instance_data)
    heuristics = reshape_heuristic(heuristic_vector, instance_data)
    
    acoInstance = ACO(15, distances.detach().numpy(), heuristics=heuristics.clone().detach().numpy())
    _, tour_costs, tour_log_probs = acoInstance.generate_paths_and_costs(gen_probs=True) # Ignore actual paths
    iteration_loss = generateLoss(torch.from_numpy(tour_costs), torch.from_numpy(tour_log_probs))
    optimiser.zero_grad()
    iteration_loss.requires_grad = True
    iteration_loss.backward()
    optimiser.step()


def train(network, problem_size, epochs, iterations_per_epoch):
    optimiser = torch.optim.AdamW(network.parameters(), lr=1e-4)

    for epoch in range(epochs):
        print(F'Starting epoch {epoch+1}')
        for _ in trange(iterations_per_epoch):
            pyg_data, distances = generate_problem_instance(problem_size)
            train_iteration(network, optimiser, pyg_data, distances)

heuristic_network = neuralnetwork.GNN(32, 12)
problem_size = 50
epochs = 5
iterations_per_epoch = 100
train(heuristic_network, problem_size, epochs, iterations_per_epoch)

heuristic_network.eval()
costs_base = []
costs_heu = []
for _ in range(5):
    data, distances = generate_problem_instance(problem_size)
    sim = ACO(15, distances.numpy())
    sim.run(50)
    costs_base.append(sim.costs)

    heuristic_vector = heuristic_network(data)
    heuristics = reshape_heuristic(heuristic_vector, data)
    sim_heu = ACO(15, distances.numpy(), heuristics=heuristics.clone().detach().numpy())
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
plt.title(f'TSP')
plt.show()
