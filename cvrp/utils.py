import matplotlib.pyplot as plt
import torch
import networkx as nx
from torch_geometric.data import Data
from pathlib import Path


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

def visualiseWeights(nodes, weights, path=None, l=None):
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
    # nx.draw_networkx_labels(G, pos, {x: x for x in range(n_nodes)})
    nx.draw_networkx_labels(G, pos, {x: int(l[x].item()) for x in range(n_nodes)})
    
    if path is not None:
        path = path.detach()
        splits = (path == 0).nonzero().flatten().tolist()

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
    # Creates n random node positions, preprended with the depot coords, with their demands
    depot_coords = torch.tensor([0.5, 0.5]).unsqueeze(dim=0)
    customer_coords = torch.rand(size=(size, 2))
    all_coords = torch.cat((depot_coords, customer_coords)) # n+1 x 2

    depot_demands = torch.zeros(size=(1, 1))
    customer_demands = torch.randint(1, 9, size=(size, 1))
    all_demands = torch.cat((depot_demands, customer_demands)) # n+1 x 1

    # instance_data = torch.cat((all_coords, all_demands), dim=1) # n+1 x 3
    return all_coords, all_demands


def convert_to_pyg_format(distances, demands, k_sparse=None):
    if k_sparse:
        # Update distances by setting non k closest connections to 0
        nearest_k_values, nearest_k_inds = torch.topk(distances, k_sparse, -1, largest=False)
        row_indices = torch.arange(distances.size(0)).unsqueeze(1).expand_as(nearest_k_inds)

        distances = torch.zeros_like(distances)
        distances[row_indices, nearest_k_inds] = nearest_k_values

    x = demands
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

def generate_variable_dataset(dataset_type, min_problem_size, max_problem_size, dataset_size_per, step=5):
    instances = []
    for problem_size in range(min_problem_size, max_problem_size + step, step):
        current_sized_instances = [generate_problem_instance(problem_size) for _ in range(dataset_size_per)]
        instances += current_sized_instances
    return instances

def load_dataset(dataset_type, problem_size):
    dataset_path = f'datasets/cvrp/{problem_size}/{dataset_type}.pt'
    assert Path(dataset_path).is_file(), 'No matching dataset exists, you can create one with generate_dataset()'
    dataset = torch.load(dataset_path)

    return dataset

