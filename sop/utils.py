import matplotlib.pyplot as plt
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.datasets import GNNBenchmarkDataset
from pathlib import Path
import random

def visualiseWeights(nodes, weights, precedences=None, path=None):
    # Visualise the weights between edges in a graph
    nodes = nodes
    weights = weights.detach()

    n_nodes = nodes.size()[0]
    weights[torch.arange(n_nodes), torch.arange(n_nodes)] = 0
    
    G = nx.from_numpy_array(weights.numpy(), create_using=nx.DiGraph()) # DiGraph for sop so can draw precendence arrows

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

    # Draw the graph
    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_size=25, node_color='lightblue')
    # nx.draw_networkx_edges(G, pos, edgelist=weight_values, edge_color='gray', width=2, alpha=[weight[2] for weight in weight_values])
    nx.draw_networkx_edges(G, pos, edgelist=weighted_edges, edge_color='gray', width=weight_values, arrows=False)
    nx.draw_networkx_labels(G, pos, {x: x for x in range(n_nodes)})

    if precedences is not None:
        nx.draw_networkx_edges(G, pos, edgelist=precedences, style='--', edge_color='red')#, min_source_margin=30, min_target_margin=30)
    
    if path is not None:
        path = path.detach().tolist()
        path_edges = list(zip(path, path[1:] + [path[0]]))[:-1]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='green')

    plt.axis('equal')
    plt.show()

def generate_constraints(problem_size, probability):
    all_possible_pairs = torch.combinations(torch.arange(problem_size)).tolist()
    random_pairs = sparsify(all_possible_pairs, probability)
    necessary_pairs = set()
    # Add all selected pairs, and their transitive depenedencis
    for pair in random_pairs:
        pair_tuple = tuple(pair)
        pair_from, pair_to = pair_tuple
        necessary_pairs.add(pair_tuple)

        transitive = [(p1, pair_to) for p1, p2 in necessary_pairs if p2 == pair_from]
        necessary_pairs.update(transitive)

    return list(necessary_pairs)

def sparsify(items, p):
    return [item for item in items if random.random() < p]


def generate_problem_instance(size, precendence_probability=0.1):
    nodes = torch.rand(size=(size, 2))
    constraints = generate_constraints(size, precendence_probability)
    return nodes, constraints

def convert_to_pyg_format(nodes, distances, precedences, k_sparse=None):
    x = nodes

    if len(precedences) != 0:
        distances = distances.clone()
        precedences = torch.tensor(precedences).t()
        distances[precedences[1], precedences[0]] = 0 # Where we have to go the other way, say no edge
    edge_data = distances.to_sparse()
    edge_index = edge_data.indices()
    edge_attr = edge_data.values().reshape(-1, 1)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def generate_dataset(dataset_type, problem_size, dataset_size):
    instances = [generate_problem_instance(problem_size) for _ in range(dataset_size)]
    dataset = torch.stack(instances)
    Path(f'datasets/sop/{problem_size}').mkdir(parents=True, exist_ok=True)
    torch.save(dataset, f'datasets/sop/{problem_size}/{dataset_type}.pt')
    print(f'Generated {dataset_size} instances in: datasets/sop/{problem_size}/{dataset_type}.pt')

def generate_variable_dataset(dataset_type, min_problem_size, max_problem_size, dataset_size_per, step=5):
    instances = []
    for problem_size in range(min_problem_size, max_problem_size + step, step):
        current_sized_instances = [generate_problem_instance(problem_size) for _ in range(dataset_size_per)]
        instances += current_sized_instances
    return instances
    dataset = torch.vstack(instances)
    print(dataset.size())
    Path(f'datasets/sop/{min_problem_size}-{max_problem_size}').mkdir(parents=True, exist_ok=True)
    torch.save(dataset, f'datasets/sop/{min_problem_size}-{max_problem_size}/{dataset_type}.pt')
    print(f'Generated {len(instances)} instances in: datasets/sop/{min_problem_size}-{max_problem_size}/{dataset_type}.pt')

def load_dataset(dataset_type, problem_size):
    dataset_path = f'datasets/sop/{problem_size}/{dataset_type}.pt'
    assert Path(dataset_path).is_file(), 'No matching dataset exists, you can create one with generate_dataset()'
    dataset = torch.load(dataset_path)
    return dataset

def load_variable_dataset(dataset_type, min_problem_size, max_problem_size, step=5, quantity=5):
    dataset_path = f'datasets/sop/{min_problem_size}-{max_problem_size}/{dataset_type}.pt'
    assert Path(dataset_path).is_file(), 'No matching dataset exists, you can create one with generate_dataset()'
    dataset = torch.load(dataset_path)
    data = []
    low = 0
    for problem_size in range(min_problem_size, max_problem_size + step, step):
        for _ in range(quantity):
            data.append(dataset[low:low+problem_size])
            low += problem_size

    return data


generate_variable_dataset('test', 20, 50, 5)

# size = 5
# nodes, precedences = generate_problem_instance(size, 1)
# distances = torch.sqrt(((nodes[:, None] - nodes[None, :]) ** 2).sum(2))
# distances[torch.arange(size), torch.arange(size)] = 1e9
# print(precedences)
# visualiseWeights(nodes, 1/distances, precedences)
