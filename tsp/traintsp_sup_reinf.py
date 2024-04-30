import torch
import numpy as np
from aco import ACO
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import neuralnetwork
import random
from tqdm import trange
from utils import generate_dataset, visualiseWeights, load_dataset, load_variable_dataset, load_variable_sols

from tsp_solver.greedy import solve_tsp

def evaluate_iteration(network, instance_data, distances, n_ants, k_sparse=None):
    network.eval()
    heuristic_vector = network(instance_data)
    heuristics = reshape_heuristic(heuristic_vector, instance_data)
    
    acoInstance = ACO(n_ants, distances, heuristics=heuristics)
    _, initial_tour_costs, _ = acoInstance.generate_paths_and_costs() # Ignore paths and probs

    acoInstance.run(100, verbose=False)
    _, simulated_tour_costs, _ = acoInstance.generate_paths_and_costs() # Ignore paths and probs

    initial_best_tour = torch.min(initial_tour_costs)
    initial_mean_tour = torch.mean(initial_tour_costs)

    simulated_best_tour = torch.min(simulated_tour_costs)
    simulated_mean_tour = torch.mean(simulated_tour_costs)

    iteration_validation_data = torch.tensor([initial_best_tour, initial_mean_tour, simulated_best_tour, simulated_mean_tour])

    # return initial_best_tour, initial_mean_tour, simulated_best_tour, simulated_mean_tour
    return iteration_validation_data

def evaluate_iteration_best(network, instance_data, distances, n_ants, k_sparse=None):
    network.eval()
    heuristic_vector = network(instance_data)
    heuristics = reshape_heuristic(heuristic_vector, instance_data)
    
    acoInstance = ACO(n_ants, distances, heuristics=heuristics)
    acoInstance.run(100, verbose=False)
    return acoInstance.best_cost

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
        iteration_data = evaluate_iteration(network, pyg_data, distances, n_ants)
        validation_data.append(iteration_data)
    validation_data = torch.stack(validation_data)
    validation_data = torch.mean(validation_data, dim=0)
    return validation_data

def validate_best(network, problem_size, n_ants, k_sparse=None):
    dataset = load_dataset('test', problem_size)
    # dataset_size = dataset.size()[0]
    validation_data = []
    for instance_nodes in dataset:
        pyg_data, distances = get_instance_data(instance_nodes, k_sparse)
        iteration_data = evaluate_iteration_best(network, pyg_data, distances, n_ants)
        validation_data.append(iteration_data)
    return sum(validation_data)/len(validation_data)

def validate_best_variable(network, min_problem_size, max_problem_size, n_ants, k_sparse=None):
    dataset = load_variable_dataset('test', min_problem_size, max_problem_size)
    validation_data = []
    # print(dataset.size())
    # n = 0
    # for instance_nodes in dataset:
    #     n += 1
    #     # print(n)
    for instance_nodes in dataset:
        pyg_data, distances = get_instance_data(instance_nodes, k_sparse)
        iteration_data = evaluate_iteration_best(network, pyg_data, distances, n_ants)
        validation_data.append(iteration_data)
    return sum(validation_data)/len(validation_data)


def generate_path_costs(paths, distances):
    length = 0
    paths.append(paths[0])
    for i in range(len(paths)-1):
        u = paths[i]
        v = paths[i+1]
        length += distances[u, v]
    return length

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
    

def train_iteration(network, optimiser, instance_data, distances, n_ants, k_sparse=None, supervised=None):
    network.train()
    heuristic_vector = network(instance_data)
    heuristics = reshape_heuristic(heuristic_vector, instance_data)
    
    acoInstance = ACO(n_ants, distances, heuristics=heuristics)
    _, tour_costs, tour_log_probs = acoInstance.generate_paths_and_costs(gen_probs=True) # Ignore actual paths

    # Inside comments is code for supervised
    # path = solve_tsp(distances)
    # optimal = generate_path_costs(path, distances)
    # loss = generateSupervisedLoss(tour_costs, tour_log_probs, optimal)

    # End
    if supervised is None:
        loss = generateLoss(tour_costs, tour_log_probs)
    else:
        loss = generateSupervisedLoss(tour_costs, tour_log_probs, supervised)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()


def train(network, problem_size, epochs, iterations_per_epoch, n_ants, k_sparse=None, lr=1e-4):
    optimiser = torch.optim.AdamW(network.parameters(), lr=lr)
    # validation_data = []
    # validation_data.append(validate(network, problem_size, n_ants, k_sparse))
    for epoch in range(epochs):
        # for _ in (pbar := trange(iterations_per_epoch)):
        for _ in range(iterations_per_epoch):
            pyg_data, distances = generate_problem_instance(problem_size, k_sparse=k_sparse)
            train_iteration(network, optimiser, pyg_data, distances, n_ants, k_sparse=k_sparse)
            # pbar.set_description(f'Epoch {epoch+1}')

        # validation_data.append(validate(network, problem_size, n_ants, k_sparse))
    # validation_data = torch.stack(validation_data)
    # plot_validation_data(validation_data)

def train_variable(network, min_problem_size, max_problem_size, epochs, iterations_per_epoch, n_ants, k_sparse=None, lr=1e-4):
    optimiser = torch.optim.AdamW(network.parameters(), lr=lr)
    # validation_data = []
    # validation_data.append(validate(network, problem_size, n_ants, k_sparse))
    for epoch in range(epochs):
        # for _ in (pbar := trange(iterations_per_epoch)):
        for _ in range(iterations_per_epoch):
            problem_size = random.randint(min_problem_size, max_problem_size)
            pyg_data, distances = generate_problem_instance(problem_size, k_sparse=k_sparse)
            train_iteration(network, optimiser, pyg_data, distances, n_ants, k_sparse=k_sparse)

def train_dataset(network, dataset, sols, iterations_per_epoch, n_ants, k_sparse=None, lr=1e-4, start=0):
    # print(start, start+iterations_per_epoch)
    optimiser = torch.optim.AdamW(network.parameters(), lr=lr)
    for index, instance_nodes in enumerate(dataset[start:start+iterations_per_epoch]):
        pyg_data, distances = get_instance_data(instance_nodes, k_sparse)
        train_iteration(network, optimiser, pyg_data, distances, n_ants, k_sparse=k_sparse, supervised=sols[start+index])

# heuristic_network = neuralnetwork.GNN(40, 20)
problem_size = 50
k_sparse = 50
epochs = 20
iterations_per_epoch = 200 #WAS 100
n_ants = 15

min_problem_size = 20
max_problem_size = 50

# Train up to 2000 total iterations to see how many are enough

SIGNIFICANCE_RUNS=15
STEPS=20 #WAS 10

sup_data = []
sup_data_limited = []
reinf_data = []

for _ in range(SIGNIFICANCE_RUNS):
    continue
    # heuristic_network = neuralnetwork.GNN(32, 12)
    # data_for_run = []
    # for iterations in range(STEPS):
    #     train_variable(heuristic_network, min_problem_size, max_problem_size, 1, iterations_per_epoch, n_ants)
    #     heuristic_network.eval()
    #     avg_over_val_data = validate_best_variable(heuristic_network, min_problem_size, max_problem_size, n_ants)
    #     data_for_run.append(avg_over_val_data)
    # reinf_data.append(data_for_run)
    # print(f'Finished reinf for {_+1}/{SIGNIFICANCE_RUNS}')

    # heuristic_network = neuralnetwork.GNN(32, 12)
    # data_for_run = []
    # dataset = load_variable_dataset('train', 20, 50, quantity=1000)
    # sols = load_variable_sols('train-sols', 20, 50)
    # order = torch.randperm(len(dataset))
    # dataset = [dataset[i] for i in order]
    # sols = sols[order]

    # # print(len(dataset))
    # # print(sols.size())

    # for iterations in range(STEPS):
    #     print(f'Starting {iterations}')
    #     train_dataset(heuristic_network, dataset, sols, iterations_per_epoch, n_ants, start=iterations*iterations_per_epoch)
    #     heuristic_network.eval()
    #     avg_over_val_data = validate_best_variable(heuristic_network, min_problem_size, max_problem_size, n_ants)
    #     data_for_run.append(avg_over_val_data)
    # sup_data.append(data_for_run)

    heuristic_network = neuralnetwork.GNN(32, 12)
    data_for_run = []
    dataset = load_variable_dataset('train', 20, 50, quantity=1000)
    sols = load_variable_sols('train-sols', 20, 50)
    order = torch.randperm(len(dataset))
    dataset = [dataset[i] for i in order]
    sols = sols[order]

    for iterations in range(STEPS):
        print(f'Starting {iterations}')
        # Use the same training instances each time
        train_dataset(heuristic_network, dataset, sols, iterations_per_epoch, n_ants, start=0)
        heuristic_network.eval()
        avg_over_val_data = validate_best_variable(heuristic_network, min_problem_size, max_problem_size, n_ants)
        data_for_run.append(avg_over_val_data)
    sup_data_limited.append(data_for_run)
    print(f'Finished sup limited for {_+1}/{SIGNIFICANCE_RUNS}')

# torch.save(torch.tensor(reinf_data), 'results/sup-reinf-reinf.pt')
# torch.save(torch.tensor(sup_data), 'results/sup-reinf-sup.pt')
# torch.save(torch.tensor(sup_data_limited), 'results/sup-reinf-sup-limited.pt')

data_reinf = torch.load('results/sup-reinf-reinf.pt')
data_sup = torch.load('results/sup-reinf-sup.pt')
data_sup_limited = torch.load('results/sup-reinf-sup-limited.pt')
# print(data_sup.size())

fig, ax = plt.subplots()
x = [(i+1)*iterations_per_epoch for i in range(15)][:8]

mean = data_reinf.mean(dim=0)[:8]
std = data_reinf.std(dim=0)[:8]
delta = 2.131 * std / (15 ** 0.5)
ax.axhline(mean.min(), linestyle='--', alpha=.5, color='black')
ax.plot(x, mean, label=f'Reinforcement Learning')
ax.fill_between(x, (mean-delta), (mean+delta), alpha=.2)

print(f'Reinf min: {mean.min()}')

x = [(i+1)*iterations_per_epoch for i in range(STEPS)]

mean = data_sup.mean(dim=0)
std = data_sup.std(dim=0)
delta = 2.131 * std / (15 ** 0.5)
ax.plot(x, mean, label=f'Supervised Learning')
ax.fill_between(x, (mean-delta), (mean+delta), alpha=.2)
print(f'Sup min: {mean.min()}')

mean = data_sup_limited.mean(dim=0)
std = data_sup_limited.std(dim=0)
delta = 2.131 * std / (15 ** 0.5)
ax.plot(x, mean, label=f'Supervised Learning limited instances')
ax.fill_between(x, (mean-delta), (mean+delta), alpha=.2)
print(f'Sup limit min: {mean.min()}')


plt.xlabel('Training iterations')
plt.ylabel('Objective value')
plt.legend()
plt.title(f'Objective value during training')
plt.show()


