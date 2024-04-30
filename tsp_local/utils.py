import matplotlib.pyplot as plt
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.datasets import GNNBenchmarkDataset
from pathlib import Path


def visualiseWeights(nodes, weights, path=None, f=False):
    # Visualise the weights between edges in a graph
    nodes = nodes
    weights = weights.detach()

    n_nodes = nodes.size()[0]
    
    G = nx.from_numpy_array(weights.numpy(), create_using=nx.Graph())

    # Set the positions of nodes in the graph
    pos = {i: (nodes[i, 0], nodes[i, 1]) for i in range(n_nodes)}

    # List of edges in format (from_node, to_node, weight)
    weighted_edges = [(i, j, weights[i, j]) for i in range(n_nodes) for j in range(n_nodes) if i != j]

    # Normalize the weights to the range [0, weight_multiplier], so [0, 1] for alpha
    weight_multiplier = 5 
    max_weight = torch.max(weights)
    min_weight = torch.min(weights)
    normaliser = weight_multiplier / (max_weight - min_weight)

    # List of weight values, but normalised
    weight_values = [normaliser * (weight[2] - min_weight) for weight in weighted_edges]
    if f:
        weight_values = [z[2] for z in weighted_edges]

    # Draw the graph
    plt.figure(figsize=(7, 7))
    nx.draw_networkx_nodes(G, pos, node_size=25, node_color='black')
    ## nx.draw_networkx_edges(G, pos, edgelist=weight_values, edge_color='gray', width=2, alpha=[weight[2] for weight in weight_values])
    # nx.draw_networkx_edges(G, pos, edgelist=weighted_edges, edge_color='gray', width=weight_values)

    # nx.draw_networkx_labels(G, pos, {i: i for i in range(n_nodes)})
    
    if path is not None:
        path = path.detach().tolist()
        path_edges = list(zip(path, path[1:] + [path[0]]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red')

    plt.axis('equal')
    plt.show()

def generate_problem_instance(size):
    nodes = torch.rand(size=(size, 2))
    return nodes

def generate_dataset(dataset_type, problem_size, dataset_size):
    instances = [generate_problem_instance(problem_size) for _ in range(dataset_size)]
    dataset = torch.stack(instances)
    Path(f'datasets/{problem_size}').mkdir(parents=True, exist_ok=True)
    torch.save(dataset, f'datasets/{problem_size}/{dataset_type}.pt')
    print(f'Generated {dataset_size} instances in: datasets/{problem_size}/{dataset_type}.pt')

def load_dataset(dataset_type, problem_size):
    dataset_path = f'datasets/{problem_size}/{dataset_type}.pt'
    assert Path(dataset_path).is_file(), 'No matching dataset exists, you can create one with generate_dataset()'
    dataset = torch.load(dataset_path)
    return dataset
