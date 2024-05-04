import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import numpy as np
from aco import ACO
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import graphnn.graphnn as graphnn
from tqdm import trange
from utils import generate_dataset, visualiseWeights, load_dataset, generate_problem_instance, convert_to_pyg_format, get_distances


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

def evaluate_iteration(network, instance_data, distances, n_ants):
    heuristics = None
    if network is not None:
        network.eval()
        heuristic_vector = network(instance_data)
        heuristics = reshape_heuristic(heuristic_vector, instance_data)
    
    acoInstance = ACO(n_ants, distances, heuristics=heuristics)
    acoInstance.run(100, verbose=False)
    return acoInstance.best_cost


def validate_dataset(network, n_ants, dataset, avg=True):
    validation_data = []
    for instance_nodes in dataset:
        distances = get_distances(instance_nodes)
        pyg_data = convert_to_pyg_format(instance_nodes, distances)

        iteration_data = evaluate_iteration(network, pyg_data, distances, instance_nodes, n_ants)
        validation_data.append(iteration_data)
    if avg:
        return sum(validation_data)/len(validation_data)
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
    ax.plot(validation_data, label='Best objective cost')

    plt.xlabel('Epoch')
    plt.ylabel('Objective cost')
    plt.legend()
    plt.title(f'Objective cost per Epoch')
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
    _, tour_costs, tour_log_probs = acoInstance.generate_paths_and_costs(gen_probs=True) # Ignore actual paths

    loss = generateLoss(tour_costs, tour_log_probs)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()


def train(network, problem_size, epochs, iterations_per_epoch, n_ants, k_sparse=None, lr=1e-4):
    optimiser = torch.optim.AdamW(network.parameters(), lr=lr)
    # validation_data = []
    val_dataset = load_dataset('val', problem_size)
    # validation_data.append(validate_dataset(network, n_ants, val_dataset))
    for epoch in range(epochs):
        for _ in (pbar := trange(iterations_per_epoch)):
            sizes = generate_problem_instance(problem_size)
            distances = get_distances(sizes)
            pyg_data = convert_to_pyg_format(sizes, distances)

            train_iteration(network, optimiser, pyg_data, distances, sizes, n_ants, k_sparse=k_sparse)
            pbar.set_description(f'Epoch {epoch+1}')
    #     validation_data.append(validate_dataset(network, n_ants, val_dataset))
    # plot_validation_data(validation_data)
