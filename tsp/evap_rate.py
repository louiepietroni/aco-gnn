import torch
import numpy as np
from aco import ACO
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import neuralnetwork
import random
from tqdm import trange
from utils import generate_dataset, visualiseWeights, load_dataset, load_variable_dataset

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

def evaluate_iteration_best_vary_evap(network, instance_data, distances, n_ants, evap, k_sparse=None):
    heuristics=None
    if network is not None:
        network.eval()
        heuristic_vector = network(instance_data)
        heuristics = reshape_heuristic(heuristic_vector, instance_data)
    
    acoInstance = ACO(n_ants, distances, heuristics=heuristics, evaportation_rate=evap)
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
        iteration_data = evaluate_iteration_best_vary_evap(network, pyg_data, distances, n_ants, 0.1)
        validation_data.append(iteration_data)
    return sum(validation_data)/len(validation_data)

def validate_best_variable_vary_evap(network, min_problem_size, max_problem_size, n_ants, evap, avg=True, k_sparse=None):
    # dataset = load_variable_dataset('test_50_2', min_problem_size, max_problem_size, quantity=50)
    dataset = load_variable_dataset('test', min_problem_size, max_problem_size, quantity=5)
    validation_data = []
    # print(dataset.size())
    # n = 0
    # for instance_nodes in dataset:
    #     n += 1
    #     # print(n)
    for instance_nodes in dataset:
        pyg_data, distances = get_instance_data(instance_nodes, k_sparse)
        iteration_data = evaluate_iteration_best_vary_evap(network, pyg_data, distances, n_ants, evap)
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

def convert_to_pyg_format(nodes, distances,k_sparse=None):
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

"""
tensor([[15.7525, 17.5157, 15.0365, 16.7608],
    [15.1880, 16.9642, 14.2720, 15.9504],
    [13.7001, 15.3524, 12.3574, 13.8757],
    [11.3397, 12.6523, 10.4144, 11.6986],
    [10.2122, 11.2819,  9.5979, 10.7184],
    [ 9.6173, 10.5592,  9.1564, 10.1280]])
"""
"""
        min pre   avg pre  min sim  avg sim
tensor([[15.7191, 17.4983, 15.0790, 16.7559],
        [15.2736, 17.0475, 14.3904, 16.0485],
        [13.5999, 15.2285, 12.2889, 13.7958],
        [11.0577, 12.3771, 10.2710, 11.5307],
        [ 9.9931, 11.0201,  9.4241, 10.4844],
        [ 9.5815, 10.5188,  9.1261, 10.0947],
        [ 9.3532, 10.2976,  9.0105,  9.9514],
        [ 9.3938, 10.1942,  8.9914,  9.7827],
        [ 9.3137, 10.1337,  8.9498,  9.7843],
        [ 9.2288, 10.1006,  8.9011,  9.7537],
        [ 9.2623, 10.1800,  8.9721,  9.8889]])

tensor([[15.7287, 17.4833, 15.0820, 16.7928],
        [15.5731, 17.3570, 14.8781, 16.5685],
        [14.2058, 15.8529, 13.1884, 14.7236],
        [11.3166, 12.5908, 10.2802, 11.5297],
        [ 9.6678, 10.7066,  9.1609, 10.2248],
        [ 9.4503, 10.3570,  9.0182,  9.9199],
        [ 9.3390, 10.2120,  8.9806,  9.8432],
        [ 9.3131, 10.1639,  8.9748,  9.8304],
        [ 9.2590, 10.1343,  8.9578,  9.7959],
        [ 9.1937, 10.0679,  8.8841,  9.7442],
        [ 9.1878, 10.0524,  8.9038,  9.7471]])
"""


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

if False:
    full_data = []
    for i in range(SIGNIFICANCE_RUNS):
        # train_variable(heuristic_network, 20, 50, 1, iterations_per_epoch, n_ants, k_sparse=None)
        # heuristic_network.eval()
        heuristic_network = None

        data = []
        for evap in np.linspace(0, 1, 21):
            val_data = validate_best_variable_vary_evap(heuristic_network, min_problem_size, max_problem_size, n_ants, evap)
            data.append(val_data)
        print(f'Completed run {i}/{SIGNIFICANCE_RUNS}')
        full_data.append(data)
    full_data = torch.tensor(full_data)
    print(full_data)
    torch.save(full_data, f'results/evap_base.pt')

    full_data = []
    for i in range(SIGNIFICANCE_RUNS):
        heuristic_network = neuralnetwork.GNN(32, 12)
        train_variable(heuristic_network, 20, 50, 1, iterations_per_epoch, n_ants, k_sparse=None)
        heuristic_network.eval()
        # heuristic_network = None

        data = []
        for evap in np.linspace(0, 1, 21):
            val_data = validate_best_variable_vary_evap(heuristic_network, min_problem_size, max_problem_size, n_ants, evap)
            data.append(val_data)
        print(f'Completed run {i}/{SIGNIFICANCE_RUNS}')
        full_data.append(data)
    full_data = torch.tensor(full_data)
    print(full_data)
    torch.save(full_data, f'results/evap_model.pt')

fig, ax = plt.subplots()
data_model = torch.load(f'results/evap_model.pt')
data_base = torch.load(f'results/evap_base.pt')
x = torch.linspace(0, 1, 21)

mean_model = data_model.mean(dim=0)
std_model = data_model.std(dim=0)
delta_model = 2.131 *  std_model / (SIGNIFICANCE_RUNS ** 0.5)

mean_base = data_base.mean(dim=0)
std_base = data_base.std(dim=0)
delta_base = 2.131 *  std_base / (SIGNIFICANCE_RUNS ** 0.5)

a = ax.plot(x, mean_model, label=f'GNN + ACO')
ax.scatter(mean_model.tolist().index(min(mean_model))/20, min(mean_model))

ax.fill_between(x, (mean_model-delta_model), (mean_model+delta_model), alpha=.2)


a = ax.plot(x, mean_base, label=f'Pure ACO')
ax.scatter(mean_base.tolist().index(min(mean_base))/20, min(mean_base), color='orange')

ax.fill_between(x, (mean_base-delta_base), (mean_base+delta_base), alpha=.2)



plt.xlabel('Evaporation rate')
plt.ylabel('Objective value')
plt.legend()
plt.title(f'Objective value as evarporation rate varies')
plt.show()
exit
# plt.subplot(1, 3, 1)
data_model = torch.load(f'results/abheatmap_model.pt')
# plt.imshow(data_model, cmap='OrRd')
# locs = np.arange(0, len(variable_values))
# plt.gca().set(xticks=locs, xticklabels=variable_values, yticks=locs, yticklabels=variable_values)

# plt.subplot(1, 3, 2)
data_base = torch.load(f'results/abheatmap_base.pt')
# plt.imshow(data_base, cmap='OrRd')
# locs = np.arange(0, len(variable_values))
# plt.gca().set(xticks=locs, xticklabels=variable_values, yticks=locs, yticklabels=variable_values)

# plt.colorbar()
# plt.show()
print(data_base)
print(data_model)
    
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8.5, 5))

for ax, data in zip(axes.flat, [data_base, data_model]):
    # ax.set_axis_off()
    im = ax.imshow(data, cmap='PuBu', vmin=5, vmax=17)
    # im = ax.imshow(data, cmap='Reds', vmin=5, vmax=17)
    locs = np.arange(0, len(variable_values))
    ax.set(xticks=locs, xticklabels=variable_values, yticks=locs, yticklabels=variable_values)

fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.2, hspace=0.02)

# add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8

cb_ax = fig.add_axes([0.83, 0.2, 0.02, 0.6])
cbar = fig.colorbar(im, cax=cb_ax)

# cbar.set_ticks(np.arange(0, 1.1, 0.5))
# cbar.set_ticklabels(['low', 'medium', 'high'])

plt.show()

