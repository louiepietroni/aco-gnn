import torch
import numpy as np
from aco import ACO
from torch_geometric.data import Data


def convert_to_pyg_format(nodes, distances):
    x = torch.from_numpy(nodes)
    edge_index = torch.stack()
    edge_attr = torch.from_numpy(distances).reshape(-1, 1) # Infer first size, requiring that other dimension is 1



def generate_problem_instance(size):
    nodes = np.random.random(size=(size, 2))
    distances = np.sqrt(((nodes[:, None] - nodes[None, :]) ** 2).sum(2))
    pyg_data = convert_to_pyg_format(nodes, distances)


    


def generateLoss(tour_costs, tour_log_probs):
    simulation_size = tour_costs.size()
    mean = tour_costs.mean()
    loss = torch.sum((tour_costs - mean) * tour_log_probs) / simulation_size
    return loss
    

def train_iteration(network, optimiser, instance):
    heuristics = network(instance)
    acoInstance = ACO(25, distances, heuristics=heuristics)
    tour_costs, tour_log_probs = acoInstance.generate_paths_and_costs(gen_probs=True)
    iteration_loss = generateLoss(tour_costs, tour_log_probs)
    optimiser.zero_grad()
    iteration_loss.backward()
    optimiser.step()


def train(network, problem_size, epochs, iterations_per_epoch):
    optimiser = torch.optim.AdamW(network.parameters(), lr=1e-4)


    for epoch in range(epochs):
        for _ in range(iterations_per_epoch):
            network.train()
            instance = np.random.random(size=(problem_size, 2))
            train_iteration(network, optimiser, instance)


heuristicNetwork = 8 # This will store the network which we will be trained
problem_size = 10
epochs = 5
iterations_per_epoch = 100
train(heuristicNetwork, problem_size, epochs, iterations_per_epoch)