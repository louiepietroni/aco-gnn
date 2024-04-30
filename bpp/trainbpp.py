import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import numpy as np
from aco import ACO
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import graphnn.neuralnetwork as graphnn
from tqdm import trange
from utils import generate_dataset, visualiseWeights, load_dataset, generate_problem_instance, convert_to_pyg_format, get_distances


def evaluate_iteration(network, instance_data, distances, sizes, n_ants, k_sparse=None):
    network.eval()
    heuristic_vector = network(instance_data)
    heuristics = reshape_heuristic(heuristic_vector, instance_data)
    
    acoInstance = ACO(n_ants, distances, sizes, heuristics=heuristics)
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

def evaluate_iteration_best(network, instance_data, distances, sizes, n_ants, k_sparse=None):
    heuristics=None
    if network is not None:
        network.eval()
        heuristic_vector = network(instance_data)
        heuristics = reshape_heuristic(heuristic_vector, instance_data)
    
    acoInstance = ACO(n_ants, distances, sizes, heuristics=heuristics)
    best_costs = []
    for _ in range(150):
        acoInstance.run(1, verbose=False)
        best_costs.append(acoInstance.best_cost)
    return best_costs


def validate(network, problem_size, n_ants, k_sparse=None):
    dataset = load_dataset('val', problem_size)
    # dataset_size = dataset.size()[0]
    validation_data = []
    for instance_nodes in dataset:
        distances = get_distances(instance_nodes)
        pyg_data = convert_to_pyg_format(instance_nodes, distances)

        iteration_data = evaluate_iteration(network, pyg_data, distances, instance_nodes, n_ants)
        validation_data.append(iteration_data)
    validation_data = torch.stack(validation_data)
    validation_data = torch.mean(validation_data, dim=0)
    return validation_data

def validate_each_iteration(network, n_ants, avg=True, k_sparse=None):
    dataset = test_dataset
    validation_data = []
    for instance_nodes in dataset:
        distances = get_distances(instance_nodes)
        pyg_data = convert_to_pyg_format(instance_nodes, distances)
    
        iteration_data = evaluate_iteration_best(network, pyg_data, distances, instance_nodes, n_ants)
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
    

def train_iteration(network, optimiser, instance_data, distances, sizes, n_ants, k_sparse=None):
    network.train()
    heuristic_vector = network(instance_data)
    heuristics = reshape_heuristic(heuristic_vector, instance_data)
    
    acoInstance = ACO(n_ants, distances, sizes, heuristics=heuristics)
    # acoInstance.run(10, verbose=False)
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
        for _ in (pbar := trange(iterations_per_epoch)):
            sizes = generate_problem_instance(problem_size)
            distances = get_distances(sizes)
            pyg_data = convert_to_pyg_format(sizes, distances)

            train_iteration(network, optimiser, pyg_data, distances, sizes, n_ants, k_sparse=k_sparse)
            pbar.set_description(f'Epoch {epoch+1}')
    #     validation_data.append(validate(network, problem_size, n_ants, k_sparse))
    # validation_data = torch.stack(validation_data)
    # plot_validation_data(validation_data)


test_dataset = load_dataset('test', 25)
SIGNIFICANCE_RUNS=15

heuristic_network = graphnn.GNN(32, 12, node_features=1)
problem_size = 100
k_sparse = None
epochs = 5
iterations_per_epoch = 100
n_ants = 15

heuristic_network.eval()
costs_base = []
costs_model = []

costs = []
test_dataset = load_dataset('test', 100)
for _ in range(SIGNIFICANCE_RUNS):
    continue
    heuristic_network = graphnn.GNN(32, 12, node_features=1)
    train(heuristic_network, problem_size, epochs, iterations_per_epoch, n_ants, k_sparse=k_sparse)

    run_costs_base = []
    run_costs_model = []
    print('trained')

    for sizes in test_dataset:
        distances = get_distances(sizes)
        pyg_data = convert_to_pyg_format(sizes, distances)

        sim = ACO(n_ants, distances, sizes)
        sim.run(100)
        run_costs_base.append(sim.costs)
        print('eval base')


        heuristic_vector = heuristic_network(pyg_data)
        heuristics = reshape_heuristic(heuristic_vector, pyg_data)
        sim_heu = ACO(n_ants, distances, sizes, heuristics=heuristics)
        sim_heu.run(100)
        run_costs_model.append(sim_heu.costs)
        print('eval model')

    costs_base.append(torch.tensor(run_costs_base).mean(dim=0).tolist())
    costs_model.append(torch.tensor(run_costs_model).mean(dim=0).tolist())
    # costs_base = run_costs_base
    # costs_model = run_costs_model

# torch.save(torch.tensor(costs_model), 'results/bpp/run-model.pt')
# torch.save(torch.tensor(costs_base), 'results/bpp/run-base.pt')

data_model = torch.load('results/bpp/run-model.pt')
data_base = torch.load('results/bpp/run-base.pt')

print(data_model.size())


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
