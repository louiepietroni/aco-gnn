import torch
import numpy as np
from aco import ACO
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import neuralnetwork
from tqdm import trange
import random
from cvrputils import generate_dataset, visualiseWeights, load_dataset, generate_problem_instance, generate_variable_dataset


def evaluate_iteration(network, instance_data, distances, demands, n_ants, k_sparse=None):
    network.eval()
    heuristic_vector = network(instance_data)
    heuristics = reshape_heuristic(heuristic_vector, instance_data)
    
    acoInstance = ACO(n_ants, distances, demands, heuristics=heuristics)
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

def evaluate_iteration_best(network, instance_data, distances, n_ants, demands, cap, k_sparse=None):
    heuristics = None
    if network is not None:
        network.eval()
        heuristic_vector = network(instance_data)
        heuristics = reshape_heuristic(heuristic_vector, instance_data)
    
    acoInstance = ACO(n_ants, distances, demands, heuristics=heuristics, capacity=cap)
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
        coords = instance_nodes[:, :-1]
        demands = instance_nodes[:, 2].unsqueeze(-1)
        distances = torch.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(2))
        distances[torch.arange(problem_size+1), torch.arange(problem_size+1)] = 1e9
        distances[0, 0] = 1e-10
        pyg_data = convert_to_pyg_format(coords, distances, demands)


        iteration_data = evaluate_iteration(network, pyg_data, distances, demands, n_ants)
        validation_data.append(iteration_data)
    validation_data = torch.stack(validation_data)
    validation_data = torch.mean(validation_data, dim=0)
    return validation_data

def validate_best_variable(network, min_problem_size, max_problem_size, n_ants, k_sparse=None, avg=True):
    # dataset = load_variable_dataset('test', min_problem_size, max_problem_size)
    dataset = test_dataset
    capacities = test_capacities
    validation_data = []
    for i, (coords, demands) in enumerate(dataset):
        # print(i)
        # print(i, capacities[i])
        pyg_data, distances = get_instance_data(coords, demands, k_sparse)
        iteration_data = evaluate_iteration_best(network, pyg_data, distances, n_ants, demands, capacities[i])
        # iteration_data = evaluate_iteration_best(network, pyg_data, distances, n_ants, demands, 50)
        validation_data.append(iteration_data)
    if avg:
        return sum(validation_data)/len(validation_data)
    else:
        return validation_data


def get_instance_data(nodes, demands, k_sparse=None):
    size = nodes.size()[0]
    distances = torch.sqrt(((nodes[:, None] - nodes[None, :]) ** 2).sum(2))
    distances[torch.arange(size), torch.arange(size)] = 1e9
    distances[0, 0] = 1e-10
    pyg_data = convert_to_pyg_format(nodes, distances, demands, k_sparse=k_sparse)
    return pyg_data, distances


def generate_path_costs(paths, distances):
    length = 0
    paths.append(paths[0])
    for i in range(len(paths)-1):
        u = paths[i]
        v = paths[i+1]
        length += distances[u, v]
    return length

def convert_to_pyg_format(nodes, distances, demands, k_sparse=None):
    x = demands
    # x = torch.cat((nodes, demands), dim=1)
    edge_data = distances.to_sparse()
    edge_index = edge_data.indices()
    edge_attr = edge_data.values().reshape(-1, 1)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


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
    

def train_iteration(network, optimiser, instance_data, distances, demands, n_ants, k_sparse=None, cap=30):
    network.train()
    heuristic_vector = network(instance_data)
    heuristics = reshape_heuristic(heuristic_vector, instance_data)
    
    acoInstance = ACO(n_ants, distances, demands, heuristics=heuristics, capacity=cap)
    # acoInstance.run(10, verbose=False)
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
            coords, demands = generate_problem_instance(problem_size)
            # Generate distance matrix
            distances = torch.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(2))
            distances[torch.arange(problem_size+1), torch.arange(problem_size+1)] = 1e9
            distances[0, 0] = 1e-10
            pyg_data = convert_to_pyg_format(coords, distances, demands)

            train_iteration(network, optimiser, pyg_data, distances, demands, n_ants, k_sparse=k_sparse)
            pbar.set_description(f'Epoch {epoch+1}')
        # validation_data.append(validate(network, problem_size, n_ants, k_sparse))
    # validation_data = torch.stack(validation_data)
    # plot_validation_data(validation_data)

def train_variable(network, min_problem_size, max_problem_size, epochs, iterations_per_epoch, n_ants, k_sparse=None, lr=1e-4, min_cap=None, max_cap=None):
    optimiser = torch.optim.AdamW(network.parameters(), lr=lr)
    for epoch in range(epochs):
        for _ in range(iterations_per_epoch):
            problem_size = random.randint(min_problem_size, max_problem_size)
            coords, demands = generate_problem_instance(problem_size)
            pyg_data, distances = get_instance_data(coords, demands)
            # # Generate distance matrix
            # distances = torch.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(2))
            # distances[torch.arange(problem_size+1), torch.arange(problem_size+1)] = 1e9
            # distances[0, 0] = 1e-10
            # pyg_data = convert_to_pyg_format(coords, distances, demands)

            cap = 30
            if min_cap is not None and max_cap is not None:
                cap = problem_size = random.randint(min_cap, max_cap)

            train_iteration(network, optimiser, pyg_data, distances, demands, n_ants, k_sparse=k_sparse, cap=cap)


# heuristic_network = neuralnetwork.GNN(32, 12)
# # heuristic_network = neuralnetwork.GNN(40, 20)
# problem_size = 25
# k_sparse = None
# epochs = 10
# iterations_per_epoch = 100
# # iterations_per_epoch = 200
# n_ants = 15
# train(heuristic_network, problem_size, epochs, iterations_per_epoch, n_ants, k_sparse=k_sparse)

# heuristic_network.eval()
# costs_base = []
# costs_heu = []

costs = []
SIGNIFICANCE_RUNS = 15
min_problem_size = 20
max_problem_size = 50
epochs = 10
iterations_per_epoch = 1500
n_ants = 15

min_cap = 10
max_cap = 50

if False:
    test_dataset = generate_variable_dataset('test', min_problem_size, max_problem_size, 45)
    order = torch.randperm(len(test_dataset))
    test_dataset = [test_dataset[i] for i in order]
    test_capacities = [[x for _ in range(35)] for x in range(min_cap, max_cap + 5, 5)]
    test_capacities = torch.tensor(test_capacities).flatten().tolist()
    print('starting')

    heuristic_network = neuralnetwork.GNN(32, 12)
    train_variable(heuristic_network, min_problem_size, max_problem_size, 1, iterations_per_epoch, n_ants)
    print('trained 30')
    heuristic_network.eval()
    data_30 = validate_best_variable(heuristic_network, min_problem_size, max_problem_size, n_ants, avg=False)
    print('evaluated 30')
    torch.save(torch.tensor(data_30), f'results/cvrp/caps-30.pt')

    heuristic_network = neuralnetwork.GNN(32, 12)
    train_variable(heuristic_network, min_problem_size, max_problem_size, 1, iterations_per_epoch, n_ants, min_cap=10, max_cap=50)
    print('trained 10-50')
    heuristic_network.eval()
    data_10_50 = validate_best_variable(heuristic_network, min_problem_size, max_problem_size, n_ants, avg=False)
    print('evaluated 10-50')
    torch.save(torch.tensor(data_10_50), f'results/cvrp/caps-10-50.pt')


    data_base = validate_best_variable(None, min_problem_size, max_problem_size, n_ants, avg=False)
    print('evaluated base')
    torch.save(torch.tensor(data_base), f'results/cvrp/caps-base.pt')


data_30 = torch.load('results/cvrp/caps-30.pt')
data_10_100 = torch.load('results/cvrp/caps-10-50.pt')
data_base = torch.load('results/cvrp/caps-base.pt')

fig, ax = plt.subplots()
x = [i for i in range(min_cap, max_cap+5, 5)]

for data, name in [(data_30, 'Model trained with capacity 30'), (data_10_100, 'Model trained with capacity 10-50'), (data_base, 'Expert heuristic')]:
    data = data.reshape((-1, 35))
    mean = data.mean(dim=1)
    std = data.std(dim=1)
    delta = 2.021 * std / (35 ** 0.5)
    ax.plot(x, mean, label=f'{name}')
    ax.fill_between(x, (mean-delta), (mean+delta), alpha=.2)


plt.xlabel('Vehicle Capacity')
plt.ylabel('Objective value')
plt.legend()
plt.title(f'Objective value against vehicle capacities')
plt.show()
