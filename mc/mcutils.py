import matplotlib.pyplot as plt
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.datasets import GNNBenchmarkDataset
from pathlib import Path


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def visualiseWeights(nodes, weights, path=None):
    # Visualise the weights between edges in a graph
    nodes = nodes

    weights = weights.detach()

    n_nodes = nodes.size()[0]

    weights[torch.arange(n_nodes), torch.arange(n_nodes)] = 0
    
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

    # Draw the graph
    plt.figure(figsize=(7, 7))
    nx.draw_networkx_nodes(G, pos, range(1, n_nodes), node_size=25, node_color='lightblue')
    nx.draw_networkx_nodes(G, pos, [0], node_size=50, node_color='green')
    # nx.draw_networkx_edges(G, pos, edgelist=weight_values, edge_color='gray', width=2, alpha=[weight[2] for weight in weight_values])
    # nx.draw_networkx_edges(G, pos, edgelist=weighted_edges, edge_color='gray', width=weight_values)
    nx.draw_networkx_labels(G, pos, {x: x for x in range(n_nodes)})
    
    if path is not None:
        path = path.detach()
        splits = (path == 0).nonzero().flatten().tolist()
        # print(path)
        # print(splits)
        # print(splits.nonzero().flatten().tolist())
        sub_paths = [path[splits[cur]:splits[cur+1]].tolist() for cur in range(len(splits)-1)]
        sub_paths = [path for path in sub_paths if len(path) > 1]
        # print(sub_paths)
        for path in sub_paths:
            color = "#{:06x}".format(torch.randint(0, 0xFFFFFF, (1,)).item())
            path_edges = list(zip(path, path[1:] + [path[0]]))
            # nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=cmap(i))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=color)
            # nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red')

    plt.axis('equal')
    plt.show()

def generate_problem_instance(size):
    # If we have n matrices, we need a sequence of n + 1 dimensions
    matrix_dimensions = torch.randint(1, 10, size=(size+3, 1)) # 2 of these will become dummy nodes
    # Then add a dummy node on either end
    matrix_dimensions[0] = matrix_dimensions[-1] = 1e-10
    return matrix_dimensions


def get_distances(matrix_dimensions):
    size = costs.size()[0]
    weights = torch.zeros(size=(size, size)) # (1+#agents+#tasks) x (1+#agents+#tasks) #-1 as costs already includes the +1
    values = torch.zeros_like(weights)
    weigh_values = costs[1:, 0]
    value_values = costs[1:, 1]
    weights[0, 1:] = weigh_values
    weights[1:, 0] = weigh_values
    values[0, 1:] = value_values
    values[1:, 0] = value_values
    
    return weights, values



def convert_to_pyg_format(distances, demands, k_sparse=None):
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

def generate_dataset(dataset_type, problem_size, dataset_size):
    instances = [generate_problem_instance(problem_size) for _ in range(dataset_size)]
    instances = [torch.cat((coords, demands), dim=1) for coords, demands in instances] # n+1 x 3
    dataset = torch.stack(instances)
    Path(f'datasets/cvrp/{problem_size}').mkdir(parents=True, exist_ok=True)
    torch.save(dataset, f'datasets/cvrp/{problem_size}/{dataset_type}.pt')
    print(f'Generated {dataset_size} instances in: datasets/cvrp/{problem_size}/{dataset_type}.pt')

def load_dataset(dataset_type, problem_size):
    dataset_path = f'datasets/cvrp/{problem_size}/{dataset_type}.pt'
    assert Path(dataset_path).is_file(), 'No matching dataset exists, you can create one with generate_dataset()'
    dataset = torch.load(dataset_path)

    return dataset

