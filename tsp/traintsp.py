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

def evaluate_iteration_best(network, instance_data, distances, n_ants, k_sparse=None):
    network.eval()
    heuristic_vector = network(instance_data)
    heuristics = reshape_heuristic(heuristic_vector, instance_data)
    
    acoInstance = ACO(n_ants, distances, heuristics=heuristics)
    acoInstance.run(100, verbose=False)
    return acoInstance.best_cost

def evaluate_iteration_best_vary_evap(network, instance_data, distances, n_ants, evap, k_sparse=None):
    heuristics=None
    if network is not None:
        network.eval()
        heuristic_vector = network(instance_data)
        heuristics = reshape_heuristic(heuristic_vector, instance_data)
    
    acoInstance = ACO(n_ants, distances, heuristics=heuristics, evaportation_rate=evap)
    acoInstance.run(100, verbose=False)
    return acoInstance.best_cost

def validate_best_variable_vary_evap(network, min_problem_size, max_problem_size, n_ants, evap, avg=True, k_sparse=None):
    # dataset = load_variable_dataset('test_50_2', min_problem_size, max_problem_size, quantity=50)
    dataset = load_variable_dataset('test', min_problem_size, max_problem_size, quantity=5)
    validation_data = []

    for instance_nodes in dataset:
        pyg_data, distances = get_instance_data(instance_nodes, k_sparse)
        iteration_data = evaluate_iteration_best_vary_evap(network, pyg_data, distances, n_ants, evap)
        validation_data.append(iteration_data)
    if avg:
        return sum(validation_data)/len(validation_data)
    else:
        return validation_data

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
        iteration_data = evaluate_iteration_best(network, pyg_data, distances, n_ants)
        validation_data.append(iteration_data)
    return sum(validation_data)/len(validation_data)

def validate_best_variable(network, min_problem_size, max_problem_size, n_ants, k_sparse=None):
    dataset = load_variable_dataset('test', min_problem_size, max_problem_size)
    validation_data = []
    # print(dataset.size())
    # n = 0
    # for instance_nodes in dataset:
    #     n += 1
    #     # print(n)
    for instance_nodes in dataset:
        pyg_data, distances = get_instance_data(instance_nodes, k_sparse)
        iteration_data = evaluate_iteration_best(network, pyg_data, distances, n_ants)
        validation_data.append(iteration_data)
    return sum(validation_data)/len(validation_data)


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
iterations_per_epoch = 100
n_ants = 15

min_problem_size = 10
max_problem_size = 100

# Train up to 2000 total iterations to see how many are enough

all_data = {}
SIGNIFICANCE_RUNS=15
STEPS=20
for problem_size in [10]:
    continue
    k_sparse = problem_size
    data_for_model = []
    # Number of runs of each model for statistical significance
    
    for model_number in range(SIGNIFICANCE_RUNS):
        heuristic_network = neuralnetwork.GNN(32, 12)
        data_for_run = []

        for iterations in range(STEPS):
            # print(iterations)
            train(heuristic_network, problem_size, 1, iterations_per_epoch, n_ants, k_sparse=k_sparse)
            # train_variable(heuristic_network, min_problem_size, max_problem_size, 1, iterations_per_epoch, n_ants)
            heuristic_network.eval()
            avg_over_val_data = validate_best(heuristic_network, problem_size, n_ants, k_sparse)
            # avg_over_val_data = validate_best_variable(heuristic_network, min_problem_size, max_problem_size, n_ants)
            data_for_run.append(avg_over_val_data)
        data_for_model.append(data_for_run)
        print(f'Completed run {model_number}/{SIGNIFICANCE_RUNS} for problem size {problem_size}')
    tensor_data_for_model = torch.tensor(data_for_model)
    print(f'DATA FOR {problem_size}, {tensor_data_for_model.size()}')
    mean = tensor_data_for_model.mean(dim=0)
    std = tensor_data_for_model.std(dim=0)
    print(mean)
    print(std)

    comp = {}
    comp['all'] = tensor_data_for_model
    comp['m'] = mean
    comp['s'] = std
    all_data[problem_size] = comp
    
    # print(mean.size())
print(all_data)

# fig, ax = plt.subplots()
# x = [(i+1)*iterations_per_epoch for i in range(STEPS)]
# for s in all_data:
#     for_s = all_data[s]
#     d = for_s['m']
#     std = for_s['s']
#     delta = 2.131 * std / (SIGNIFICANCE_RUNS ** 0.5)
#     a = ax.plot(x, d, label=f'TSP{s}')
#     ax.fill_between(x, (d-delta), (d+delta), alpha=.1)

# plt.xlabel('Training iterations')
# plt.ylabel('Objective value')
# plt.legend()
# plt.title(f'Objective value during training')
# plt.show()



# 50iters x 20 times
tsp10 = {
    'm' : torch.tensor([3.2073, 3.1806, 3.1621, 3.1448, 3.1176, 3.0638, 3.0300, 2.9935, 2.9541,
        2.9270, 2.9117, 2.9028, 2.8989, 2.8980, 2.8980, 2.8979, 2.8979, 2.8979,
        2.8980, 2.8982]),
    's' : torch.tensor([2.4646e-02, 1.7057e-02, 2.0453e-02, 3.1193e-02, 3.4884e-02, 3.1288e-02,
        4.2405e-02, 4.3975e-02, 4.2721e-02, 2.9153e-02, 1.9896e-02, 9.0998e-03,
        2.2271e-03, 4.0447e-04, 3.0467e-04, 8.1003e-05, 3.6197e-05, 1.5375e-04,
        2.0354e-04, 3.4508e-04])
}

# 50 iters x 20 times
tsp20 = {
    'm' : torch.tensor([6.9121, 6.8570, 6.7296, 6.5739, 6.3778, 6.1276, 5.8549, 5.5044, 5.1798,
        4.8512, 4.5476, 4.3119, 4.1367, 3.9961, 3.9009, 3.8503, 3.8252, 3.8110,
        3.8054, 3.8024]),
    's' : torch.tensor([0.0604, 0.0716, 0.1042, 0.1406, 0.2036, 0.2476, 0.3364, 0.3438, 0.3589,
        0.3305, 0.3065, 0.2788, 0.2286, 0.1675, 0.1105, 0.0714, 0.0431, 0.0233,
        0.0122, 0.0057])
}
# 50 iters x 20 times
tsp50 = {
    'm' : torch.tensor([20.3598, 20.1006, 19.7501, 19.1896, 18.4337, 17.4801, 16.2191, 14.7615,
        13.2882, 11.8767, 10.6344,  9.6074,  8.7567,  8.0715,  7.5734,  7.1566,
         6.7957,  6.5345,  6.3383,  6.1676]),
    's' : torch.tensor([0.1431, 0.2701, 0.3679, 0.5894, 0.8928, 1.1798, 1.5351, 1.7253, 1.7466,
        1.6534, 1.4295, 1.2275, 1.0129, 0.8117, 0.6727, 0.5702, 0.4446, 0.3553,
        0.2939, 0.2189])
}

# 100 iters x 20 times
tsp100 = {
    'm' : torch.tensor([43.3061, 40.7958, 35.5042, 27.4608, 19.9409, 15.3021, 12.7611, 11.2257,
        10.2103,  9.5418,  9.0447,  8.7448,  8.5480,  8.4435,  8.3803,  8.3534,
         8.3427,  8.3395,  8.3330,  8.3338]),
    's' : torch.tensor([0.3366, 0.8733, 1.8780, 2.5962, 2.3068, 1.5280, 1.0443, 0.6576, 0.4471,
        0.3222, 0.2163, 0.1421, 0.0997, 0.0602, 0.0407, 0.0261, 0.0192, 0.0127,
        0.0124, 0.0115])
}

# 100 iters x 20 times
tsp50_long = {
    'm' : torch.tensor([20.0211, 18.8590, 16.7284, 13.7438, 10.8359,  8.8283,  7.5357,  6.7878,
         6.3227,  6.0492,  5.9023,  5.8243,  5.8041,  5.7859,  5.7839,  5.7778,
         5.7836,  5.7853,  5.7858,  5.7941]),
    's' : torch.tensor([0.2858, 0.8409, 1.5170, 1.9030, 1.5582, 1.1162, 0.7354, 0.4805, 0.3018,
        0.1795, 0.0921, 0.0522, 0.0237, 0.0094, 0.0083, 0.0069, 0.0097, 0.0093,
        0.0136, 0.0097])
}

# 100 iters x 20 times
tsp20_50 = {
    'm' : torch.tensor([12.7760, 12.3415, 11.4209,  9.9156,  8.2885,  6.9278,  5.9967,  5.4368,
         5.1175,  4.9500,  4.8688,  4.8290,  4.8099,  4.8043,  4.8079,  4.8120,
         4.8155,  4.8258,  4.8288,  4.8341]),
    's' : torch.tensor([0.1183, 0.1899, 0.3865, 0.6260, 0.6668, 0.5605, 0.3819, 0.2296, 0.1297,
        0.0727, 0.0379, 0.0195, 0.0121, 0.0087, 0.0076, 0.0084, 0.0093, 0.0063,
        0.0063, 0.0103])
}

# 100 iters x 20 times
tsp10_100 = {
    'm' : torch.tensor([22.9416, 21.7405, 19.5404, 16.3361, 13.0374, 10.4617,  8.8911,  7.9101,
         7.2511,  6.8435,  6.5691,  6.3520,  6.2133,  6.1269,  6.0693,  6.0380,
         6.0232,  6.0128,  5.9996,  5.9943]),
    's' : torch.tensor([0.3299, 0.8235, 1.6888, 2.1303, 2.1416, 1.7118, 1.1466, 0.8202, 0.5245,
        0.4139, 0.3520, 0.2359, 0.1605, 0.0997, 0.0745, 0.0514, 0.0516, 0.0446,
        0.0217, 0.0154])
}

tsp20_long = {
    'm' : torch.tensor([6.8545, 6.5684, 6.0812, 5.4164, 4.6983, 4.1866, 3.9187, 3.8255, 3.8045,
        3.8021, 3.8022, 3.8027, 3.8048, 3.8057, 3.8072, 3.8098, 3.8109, 3.8133,
        3.8161, 3.8154]),
    's' : torch.tensor([0.0585, 0.1515, 0.2242, 0.3070, 0.2991, 0.1919, 0.0787, 0.0244, 0.0048,
        0.0016, 0.0018, 0.0019, 0.0027, 0.0025, 0.0046, 0.0039, 0.0033, 0.0046,
        0.0037, 0.0041])
}

tsp10_long  = {
    'm' : torch.tensor([3.1835, 3.1450, 3.0692, 2.9798, 2.9189, 2.9009, 2.8979, 2.8979, 2.8981,
        2.8980, 2.8985, 2.8985, 2.8992, 2.9000, 2.9004, 2.9013, 2.9030, 2.9036,
        2.9042, 2.9053]),
    's' : torch.tensor([0.0259, 0.0354, 0.0404, 0.0346, 0.0194, 0.0048, 0.0002, 0.0001, 0.0004,
        0.0002, 0.0007, 0.0005, 0.0008, 0.0013, 0.0022, 0.0019, 0.0020, 0.0024,
        0.0022, 0.0031])
}

fig, ax = plt.subplots()
x = [(i+1)*iterations_per_epoch for i in range(STEPS)]
# all_data = {tsp50_long, tsp100, tsp20_50, tsp10_100}
for for_s, name in zip([tsp10_long, tsp20_long, tsp50_long, tsp100, tsp20_50, tsp10_100], ['10', '20', '50', '100', '20-50', '10-100']):
    d = for_s['m']
    std = for_s['s']
    delta = 2.131 * std / (SIGNIFICANCE_RUNS ** 0.5)
    # delta = 2.947 * std / (SIGNIFICANCE_RUNS ** 0.5)
    a = ax.plot(x, d, label=f'TSP{name}')
    ax.fill_between(x, (d-delta), (d+delta), alpha=.4)

plt.xlabel('Training iterations')
plt.ylabel('Objective value')
plt.legend()
plt.title(f'Objective value during training')
plt.show()



