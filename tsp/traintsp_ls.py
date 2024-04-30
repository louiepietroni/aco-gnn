import torch
import numpy as np
from aco import ACO
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import neuralnetwork
import random
from tqdm import trange
from utils import generate_dataset, visualiseWeights, load_dataset, load_variable_dataset
from localsearch import two_opt

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
    heuristics=None
    if network is not None:
        network.eval()
        heuristic_vector = network(instance_data)
        heuristics = reshape_heuristic(heuristic_vector, instance_data)
    
    acoInstance = ACO(n_ants, distances, heuristics=heuristics)
    best_costs = []
    for _ in range(100):
        acoInstance.run(1, verbose=False)
        best_costs.append(acoInstance.best_cost)
    # updated_paths = acoInstance.most_recent_paths
    updated_paths = acoInstance.best_path
    for _ in range(20):
        updated_paths = two_opt(updated_paths, distances)
        best_costs.append(acoInstance.generate_path_costs(updated_paths).min())
    return best_costs


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

def validate_each_iteration(network, min_problem_size, max_problem_size, n_ants, avg=True, k_sparse=None):
    dataset = load_variable_dataset('test', min_problem_size, max_problem_size, quantity=50)
    validation_data = []
    for instance_nodes in dataset:
        pyg_data, distances = get_instance_data(instance_nodes, k_sparse)
        iteration_data = evaluate_iteration_best(network, pyg_data, distances, n_ants)
        validation_data.append(iteration_data)
    validation_data = torch.tensor(validation_data)
    if avg:
        return validation_data.mean(dim=0)
    else:
        return validation_data


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
        k_sparse = int(distances.size(0) * k_sparse)
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
    

def train_iteration(network, optimiser, instance_data, distances, n_ants, k_sparse=None):
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
    loss = generateLoss(tour_costs, tour_log_probs)
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

# heuristic_network = neuralnetwork.GNN(40, 20)
problem_size = 50
k_sparse = 50
epochs = 20
iterations_per_epoch = 1500
n_ants = 15

min_problem_size = 10
max_problem_size = 100

# Train up to 2000 total iterations to see how many are enough
torch.set_printoptions(precision=5)

all_data = {}
SIGNIFICANCE_RUNS=15
STEPS=20
data_to_save = []
# for problem_size in [10, 20, 50, 100]:
data_base = []
data_model = []
for _ in range(SIGNIFICANCE_RUNS):
    continue
    val_data = validate_each_iteration(None, min_problem_size, max_problem_size, n_ants, avg=True, k_sparse=None)
    data_base.append(val_data)

    # val_data = validate_each_iteration(None, min_problem_size, max_problem_size, n_ants, avg=True, k_sparse=0.25)
    # data_base_spar_25.append(val_data)
    # val_data = validate_each_iteration(None, min_problem_size, max_problem_size, n_ants, avg=True, k_sparse=0.5)
    # data_base_spar_50.append(val_data)

    heuristic_network = neuralnetwork.GNN(32, 12)
    train_variable(heuristic_network, min_problem_size, max_problem_size, 1, iterations_per_epoch, n_ants, k_sparse=None)
    heuristic_network.eval()

    val_data = validate_each_iteration(heuristic_network, min_problem_size, max_problem_size, n_ants, avg=True, k_sparse=None)
    data_model.append(val_data)

    # val_data = validate_each_iteration(heuristic_network, min_problem_size, max_problem_size, n_ants, avg=True, k_sparse=0.1)
    # data_model_spar_10.append(val_data)
    # val_data = validate_each_iteration(heuristic_network, min_problem_size, max_problem_size, n_ants, avg=True, k_sparse=0.25)
    # data_model_spar_25.append(val_data)
    # val_data = validate_each_iteration(heuristic_network, min_problem_size, max_problem_size, n_ants, avg=True, k_sparse=0.50)
    # data_model_spar_50.append(val_data)

    print(f'Completed significane run {_+1}/{SIGNIFICANCE_RUNS}')


# torch.save(torch.stack(data_base), f'results/ls-base.pt')
# torch.save(torch.stack(data_base_spar_10), f'results/spar-base-10.pt')
# torch.save(torch.stack(data_model), f'results/ls-model.pt')
# torch.save(torch.stack(data_model_spar_10), f'results/spar-model-10.pt')
    
# torch.save(torch.stack(data_base_spar_25), f'results/spar-base-25.pt')
# torch.save(torch.stack(data_base_spar_50), f'results/spar-base-50.pt')
# torch.save(torch.stack(data_model_spar_25), f'results/spar-model-25.pt')
# torch.save(torch.stack(data_model_spar_50), f'results/spar-model-50.pt')
sub = 0

data_base = torch.load('results/ls-base.pt')[:, sub:]
# data_base_spar_10 = torch.load('results/spar-base-10.pt')
# data_base_spar_25 = torch.load('results/spar-base-25.pt')
# data_base_spar_50 = torch.load('results/spar-base-50.pt')
data_model = torch.load('results/ls-model.pt')[:, sub:]
# data_model_spar_10 = torch.load('results/spar-model-10.pt')
# data_model_spar_25 = torch.load('results/spar-model-25.pt')
# data_model_spar_50 = torch.load('results/spar-model-50.pt')

# data = (data / OPTIMAL10_100_50 - 1) * 100
fig, ax = plt.subplots()
x = [i for i in range(1, 101 + 20)][sub:]

mean = data_base.mean(dim=0)
std = data_base.std(dim=0)
delta = 2.131 * std / (15 ** 0.5)
ax.plot(x, mean, label=f'Pure ACO')
ax.fill_between(x, (mean-delta), (mean+delta), alpha=.4)
print(f'Base {mean}')


# mean = data_base_spar_10.mean(dim=0)
# std = data_base_spar_10.std(dim=0)
# delta = 2.131 * std / (15 ** 0.5)
# ax.plot(x, mean, label=f'Pure ACO, 10% sparsification')
# ax.fill_between(x, (mean-delta), (mean+delta), alpha=.1)

# mean = data_base_spar_25.mean(dim=0)
# std = data_base_spar_25.std(dim=0)
# delta = 2.131 * std / (15 ** 0.5)
# ax.plot(x, mean, label=f'Pure ACO, 25% sparsification')
# ax.fill_between(x, (mean-delta), (mean+delta), alpha=.1)

# mean = data_base_spar_50.mean(dim=0)
# std = data_base_spar_50.std(dim=0)
# delta = 2.131 * std / (15 ** 0.5)
# ax.plot(x, mean, label=f'Pure ACO, 50% sparsification')
# ax.fill_between(x, (mean-delta), (mean+delta), alpha=.1)
m = mean

mean = data_model.mean(dim=0)
std = data_model.std(dim=0)
delta = 2.131 * std / (15 ** 0.5)
ax.plot(x, mean, label=f'GNN+ACO')
ax.fill_between(x, (mean-delta), (mean+delta), alpha=.4)
print(f'Model {mean}')

# ax.fill_between(x[:100], min(mean), max(m), alpha=.15, label='ACO rounds')
# ax.fill_between(x[99:], min(mean), max(m), alpha=.15, label='LS rounds')
plt.axvspan(0, 100, facecolor='green', alpha=0.15, label='ACO rounds')
plt.axvspan(100, 120, facecolor='red', alpha=0.15, label='LS rounds')


# mean = data_model_spar_10.mean(dim=0)
# std = data_model_spar_10.std(dim=0)
# delta = 2.131 * std / (15 ** 0.5)
# ax.plot(x, mean, label=f'GNN+ACO, 10% sparsification')
# ax.fill_between(x, (mean-delta), (mean+delta), alpha=.1)

# mean = data_model_spar_25.mean(dim=0)
# std = data_model_spar_25.std(dim=0)
# delta = 2.131 * std / (15 ** 0.5)
# ax.plot(x, mean, label=f'GNN+ACO, 25% sparsification')
# ax.fill_between(x, (mean-delta), (mean+delta), alpha=.1)

# mean = data_model_spar_50.mean(dim=0)
# std = data_model_spar_50.std(dim=0)
# delta = 2.131 * std / (15 ** 0.5)
# ax.plot(x, mean, label=f'GNN+ACO, 50% sparsification')
# ax.fill_between(x, (mean-delta), (mean+delta), alpha=.1)
plt.xlabel('Rounds of solution search')
plt.ylabel('Objective value')
plt.title(f'Objective value during ACO and Local search')
plt.legend()
plt.show()





sub = 99
data_base = torch.load('results/ls-base.pt')[:, sub:]
data_model = torch.load('results/ls-model.pt')[:, sub:]

fig, ax = plt.subplots()
x = [i for i in range(0, 21)]
data_base = 1-(data_base / data_base[:, 0].expand(21, -1).t())
data_base *= 100

mean = data_base.mean(dim=0)
ax.axhline(mean.max(), color='lightblue', linestyle='--', alpha=1, zorder=1)

std = data_base.std(dim=0)
delta = 2.131 * std / (15 ** 0.5)
ax.plot(x, mean, label=f'Pure ACO')
ax.fill_between(x, (mean-delta), (mean+delta), alpha=.3)
print(f'Base {mean}')


data_model = 1-(data_model / data_model[:, 0].expand(21, -1).t())
data_model *= 100


mean = data_model.mean(dim=0)

std = data_model.std(dim=0)
delta = 2.131 * std / (15 ** 0.5)
ax.plot(x, mean, label=f'GNN+ACO')
ax.fill_between(x, (mean-delta), (mean+delta), alpha=.3)
print(f'Model {mean}')
ax.axhline(mean.max(), color='orange', linestyle='--', alpha=.5)

ax.set_xticks(np.linspace(0, 20, 11))
plt.xlabel('Rounds of local search')
plt.ylabel('% Improvement on best ACO solution')
plt.title(f'% Improvement against ACO solution during Local search')
plt.legend()
plt.show()
