import torch
import numpy as np
from aco import ACO
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import neuralnetwork
from tqdm import trange
from utils import generate_dataset, visualiseWeights, load_dataset, generate_problem_instance, convert_to_pyg_format

from tsp_solver.greedy import solve_tsp

def evaluate_iteration(network, instance_data, distances, n_ants, k_sparse=None):
    network.eval()
    heuristic_vector = network(instance_data)
    heuristics = reshape_heuristic(heuristic_vector, instance_data)
    
    acoInstance = ACO(n_ants, distances, heuristics=heuristics)
    _, initial_tour_costs, _ = acoInstance.generate_paths_and_costs() # Ignore paths and probs

    acoInstance.run(10, verbose=False)
    _, simulated_tour_costs, _ = acoInstance.generate_paths_and_costs() # Ignore paths and probs

    initial_best_tour = torch.min(initial_tour_costs)
    initial_mean_tour = torch.mean(initial_tour_costs)

    simulated_best_tour = torch.min(simulated_tour_costs)
    simulated_mean_tour = torch.mean(simulated_tour_costs)

    iteration_validation_data = torch.tensor([initial_best_tour, initial_mean_tour, simulated_best_tour, simulated_mean_tour])

    # return initial_best_tour, initial_mean_tour, simulated_best_tour, simulated_mean_tour
    return iteration_validation_data

def get_instance_data(nodes, k_sparse=None):
    size = nodes.size()[0]
    distances = torch.sqrt(((nodes[:, None] - nodes[None, :]) ** 2).sum(2))
    distances[torch.arange(size), torch.arange(size)] = 1e9
    pyg_data = convert_to_pyg_format(nodes, distances, k_sparse=k_sparse)
    return pyg_data, distances



def validate(network, problem_size, n_ants, k_sparse=None):
    dataset = load_dataset('val', problem_size)
    # dataset_size = dataset.size()[0]
    validation_data = []
    for instance_nodes in dataset:
        pyg_data, distances = get_instance_data(instance_nodes, k_sparse)
        # init_best, init_mean, sim_best, sim_mean = evaluate_iteration(network, pyg_data, distances, n_ants)
        # avg_init_best += init_best
        # avg_init_mean += init_mean
        # avg_sim_best += sim_best
        # avg_sim_mean += sim_mean
        iteration_data = evaluate_iteration(network, pyg_data, distances, n_ants)
        validation_data.append(iteration_data)
    validation_data = torch.stack(validation_data)
    validation_data = torch.mean(validation_data, dim=0)
    return validation_data
    # avg_init_best /= dataset_size
    # avg_init_mean /= dataset_size
    # avg_sim_best /= dataset_size
    # avg_sim_mean /= dataset_size

    # return avg_init_best, avg_init_mean, avg_sim_best, avg_sim_mean




def generate_path_costs(paths, distances):
    length = 0
    paths.append(paths[0])
    for i in range(len(paths)-1):
        u = paths[i]
        v = paths[i+1]
        length += distances[u, v]
    return length


def plot_validation_data(validation_data):
    fig, ax = plt.subplots()
    ax.plot(validation_data[:, 0], label='Best initial path cost')
    ax.plot(validation_data[:, 1], label='Average initial path cost')
    ax.plot(validation_data[:, 2], label='Best post simulation path cost')
    ax.plot(validation_data[:, 3], label='Average post simulation path cost')

    plt.xlabel('Epoch')
    plt.ylabel('Path Length')
    plt.legend()
    plt.title(f'Path Lengths per Epoch')
    plt.show()
    print(validation_data)


def generateLoss(tour_costs, tour_log_probs):
    mean = tour_costs.mean()
    size = tour_costs.size()[0] # #ants
    loss = torch.sum((tour_costs - mean) * tour_log_probs.sum(dim=0)) / size
    return loss

def generateSupervisedLoss(tour_costs, tour_log_probs, optimal):
    
    size = tour_costs.size()[0] # #ants
    loss = torch.sum((tour_costs - optimal) * tour_log_probs.sum(dim=0)) / size
    return loss

def reshape_heuristic(heursitic_vector, pyg_data):
    problem_size = pyg_data.x.shape[0]
    heuristic_matrix = torch.full((problem_size, problem_size), 1e-10)

    heuristic_matrix[pyg_data.edge_index[0], pyg_data.edge_index[1]] = heursitic_vector.squeeze(dim = -1)
    return heuristic_matrix
    

def train_iteration(network, optimiser, instance_data, distances, precedences, n_ants, k_sparse=None):
    network.train()
    heuristic_vector = network(instance_data)
    heuristics = reshape_heuristic(heuristic_vector, instance_data)
    
    acoInstance = ACO(n_ants, distances, precedences, heuristics=heuristics)
    _, tour_costs, tour_log_probs = acoInstance.generate_paths_and_costs(gen_probs=True) # Ignore actual paths

    loss = generateLoss(tour_costs, tour_log_probs)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()


def train(network, problem_size, epochs, iterations_per_epoch, n_ants, k_sparse=None, lr=1e-4):
    optimiser = torch.optim.AdamW(network.parameters(), lr=lr)
    validation_data = []
    # validation_data.append(validate(network, problem_size, n_ants, k_sparse))
    for epoch in range(epochs):
        for _ in (pbar := trange(iterations_per_epoch)):
            nodes, precedences = generate_problem_instance(problem_size, 0.1)
            distances = torch.sqrt(((nodes[:, None] - nodes[None, :]) ** 2).sum(2))
            distances[torch.arange(problem_size), torch.arange(problem_size)] = 0
            pyg_data = convert_to_pyg_format(nodes, distances, precedences)

            train_iteration(network, optimiser, pyg_data, distances, precedences, n_ants, k_sparse=k_sparse)
            pbar.set_description(f'Epoch {epoch+1}')
        # validation_data.append(validate(network, problem_size, n_ants, k_sparse))
    # validation_data = torch.stack(validation_data)
    # plot_validation_data(validation_data)

heuristic_network = neuralnetwork.GNN(32, 12)
# heuristic_network = neuralnetwork.GNN(40, 20)
problem_size = 50
k_sparse = 10
epochs = 10
iterations_per_epoch = 100
# iterations_per_epoch = 200
n_ants = 15
train(heuristic_network, problem_size, epochs, iterations_per_epoch, n_ants, k_sparse=k_sparse)

heuristic_network.eval()
costs_base = []
costs_heu = []

costs = []


for _ in range(20):
    nodes, precedences = generate_problem_instance(problem_size, 0.1)
    distances = torch.sqrt(((nodes[:, None] - nodes[None, :]) ** 2).sum(2))
    distances[torch.arange(problem_size), torch.arange(problem_size)] = 1e9
    pyg_data = convert_to_pyg_format(nodes, distances, precedences)

    sim = ACO(n_ants, distances, precedences)
    sim.run(50)
    costs_base.append(sim.costs)








    # visualiseWeights(data.x, sim.heuristics)
    # visualiseWeights(data.x, sim.pheromones * sim.heuristics)
    # visualiseWeights(nodes, sim.pheromones * sim.heuristics, path=sim.generate_best_path())

    heuristic_vector = heuristic_network(pyg_data)
    heuristics = reshape_heuristic(heuristic_vector, pyg_data)
    sim_heu = ACO(n_ants, distances, precedences, heuristics=heuristics)
    sim_heu.run(50)
    costs_heu.append(sim_heu.costs)

    # visualiseWeights(nodes, sim_heu.pheromones * sim_heu.heuristics, path=sim_heu.generate_best_path())


costs_base = np.column_stack(tuple(costs_base))
costs_heu = np.column_stack(tuple(costs_heu))

fig, ax = plt.subplots()
ax.plot(np.mean(costs_base, axis=1), label='Base')
ax.plot(np.mean(costs_heu, axis=1), label='Heu')

plt.xlabel('No. Iterations')
plt.ylabel('Path Length')
plt.legend()
plt.title(f'SOP {problem_size}')
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
