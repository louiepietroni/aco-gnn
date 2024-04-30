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

def get_distances(edges, size):
    distances = torch.zeros(size=(size+1, size+1))
    # distances[:, :] = 1e9 # Set most distances to max
    distances[edges[:, 0], edges[:, 1]] = 1 # Distances where we have an edge are 1
    distances[edges[:, 1], edges[:, 0]] = 1
    # distances[0, :] = 1e-9 # Distances to/from dummy node is very small
    # distances[:, 0] = 1e-9
    return distances

def sparsify(items, p):
    # Returns each item in items with probability of p
    return [item for item in items if torch.rand(size=(1,)).item() < p]

def generate_problem_instance(size, p=0.25):
    # Creates n nodes, with edges between them
    all_possible_pairs = torch.combinations(torch.arange(size)).tolist()
    random_pairs = torch.tensor(sparsify(all_possible_pairs, p)) + 1 # +1 as now node 0 will be a dummy node
    return random_pairs

# p = generate_problem_instance(4, p=0.5)
# print(p)
# from_nodes = p[:, 0].expand(3, -1)
# to_nodes = p[:, 1].expand(3, -1)
# both = torch.stack((from_nodes, to_nodes))
# print(both)

# pos = torch.tensor([[1], [3], [2]])
# pos = pos.expand(-1, len(p))
# print(pos)

# print(torch.any(torch.eq(both, pos), dim=0))


def convert_to_pyg_format(distances):
    x = torch.sum(distances, dim=1).unsqueeze(dim=1) # # Number of edges from this node
    # edge_data = distances.to_sparse()
    adjacency_matix = torch.ones_like(distances)
    adjacency_matix[distances!=1] = 1e9 # Locations with no edge edge = 1, no edge = -1
    adjacency_matix[distances!=1] = -1 # Locations with no edge edge = 1, no edge = -1
    # NOTE: Above with -1, 1e-9, 1e9 try out!
    # adjacency_matix[distances==1] = -1 # Locations with edge
    # print(adjacency_matix)
    edge_data = adjacency_matix.to_sparse()
    edge_index = edge_data.indices()
    edge_attr = edge_data.values().reshape(-1, 1)
    # print(edge_attr)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

# convert_to_pyg_format(get_distances(generate_problem_instance(6), 6))

def generate_dataset(dataset_type, problem_size, dataset_size):
    instances = [generate_problem_instance(problem_size) for _ in range(dataset_size)]
    # dataset = torch.stack(instances)
    dataset = instances
    Path(f'datasets/vcp/{problem_size}').mkdir(parents=True, exist_ok=True)
    torch.save(dataset, f'datasets/vcp/{problem_size}/{dataset_type}.pt')
    print(f'Generated {dataset_size} instances in: datasets/vcp/{problem_size}/{dataset_type}.pt')

def load_dataset(dataset_type, problem_size):
    dataset_path = f'datasets/vcp/{problem_size}/{dataset_type}.pt'
    assert Path(dataset_path).is_file(), 'No matching dataset exists, you can create one with generate_dataset()'
    dataset = torch.load(dataset_path)

    return dataset


# generate_dataset('test', 30, 25)
