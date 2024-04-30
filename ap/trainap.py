import torch
import numpy as np
from aco import ACO
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import neuralnetwork
from tqdm import trange
import random
from utils import generate_dataset, visualiseWeights, load_dataset, generate_problem_instance, get_distances, convert_to_pyg_format, generate_variable_dataset, solve_dataset


def evaluate_iteration(network, instance_data, distances, n_ants, k_sparse=None):
    network.eval()
    heuristic_vector = network(instance_data)
    heuristics = reshape_heuristic(heuristic_vector, instance_data)
    
    acoInstance = ACO(n_ants, distances, heuristics=heuristics)
    _, initial_tour_costs, _ = acoInstance.generate_paths_and_costs() # Ignore paths and probs

    acoInstance.run(10, verbose=False)
    _, simulated_tour_costs, _ = acoInstance.generate_paths_and_costs() # Ignore paths and probs

    initial_best_tour = torch.min(initial_tour_costs)
    initial_mean_tour = torch.mean(initial_tour_costs)

    simulated_best_tour = torch.min(simulated_tour_costs)
    simulated_mean_tour = torch.mean(simulated_tour_costs)

    iteration_validation_data = torch.tensor([initial_best_tour, initial_mean_tour, simulated_best_tour, simulated_mean_tour])

    return iteration_validation_data


def evaluate_iteration_best(network, instance_data, distances, n_ants, k_sparse=None):
    heuristics = None
    if network is not None:
        network.eval()
        heuristic_vector = network(instance_data)
        heuristics = reshape_heuristic(heuristic_vector, instance_data)
    
    acoInstance = ACO(n_ants, distances, heuristics=heuristics)
    acoInstance.run(100, verbose=False)
    return acoInstance.best_cost

def get_instance_data(nodes, k_sparse=None):
    size = nodes.size()[0]
    distances = torch.sqrt(((nodes[:, None] - nodes[None, :]) ** 2).sum(2))
    distances[torch.arange(size), torch.arange(size)] = 1e9
    pyg_data = convert_to_pyg_format(nodes, distances, k_sparse=k_sparse)
    return pyg_data, distances



def validate(network, problem_size, n_ants, k_sparse=None):
    print('Running Validation')
    dataset = load_dataset('val', problem_size)
    # dataset_size = dataset.size()[0]
    validation_data = []
    for instance_costs in dataset:
        distances = get_distances(instance_costs)
        pyg_data = convert_to_pyg_format(distances)

        iteration_data = evaluate_iteration(network, pyg_data, distances, n_ants)
        validation_data.append(iteration_data)
    validation_data = torch.stack(validation_data)
    validation_data = torch.mean(validation_data, dim=0)
    return validation_data

def validate_best_variable(network, min_problem_size, max_problem_size, n_ants, k_sparse=None, avg=True):
    # dataset = load_variable_dataset('test', min_problem_size, max_problem_size)
    dataset = test_dataset
    dataset_sols = sols
    validation_data = []
    for i, instance_costs in enumerate(dataset):
        distances = get_distances(instance_costs)
        pyg_data = convert_to_pyg_format(distances)

        iteration_data = evaluate_iteration_best(network, pyg_data, distances, n_ants)
        # iteration_data = evaluate_iteration_best(network, pyg_data, distances, n_ants, demands, 50)
        validation_data.append(iteration_data / dataset_sols[i])
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
    

def train_iteration(network, optimiser, instance_data, distances, n_ants, k_sparse=None):
    network.train()
    heuristic_vector = network(instance_data)
    heuristics = reshape_heuristic(heuristic_vector, instance_data)
    
    acoInstance = ACO(n_ants, distances, heuristics=heuristics)
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
    validation_data = []
    # validation_data.append(validate(network, problem_size, n_ants, k_sparse))
    for epoch in range(epochs):
        for _ in range(iterations_per_epoch):
            task_costs = generate_problem_instance(problem_size)
            distances = get_distances(task_costs)
            pyg_data = convert_to_pyg_format(distances)

            train_iteration(network, optimiser, pyg_data, distances, n_ants, k_sparse=k_sparse)
    #     validation_data.append(validate(network, problem_size, n_ants, k_sparse))
    # validation_data = torch.stack(validation_data)
    # plot_validation_data(validation_data)
            
def train_variable(network, min_problem_size, max_problem_size, epochs, iterations_per_epoch, n_ants, k_sparse=None, lr=1e-4):
    optimiser = torch.optim.AdamW(network.parameters(), lr=lr)
    for epoch in range(epochs):
        for _ in range(iterations_per_epoch):
            problem_size = random.randint(min_problem_size, max_problem_size)
            task_costs = generate_problem_instance(problem_size)
            distances = get_distances(task_costs)
            pyg_data = convert_to_pyg_format(distances)

            train_iteration(network, optimiser, pyg_data, distances, n_ants, k_sparse=k_sparse)


costs = []
min_problem_size = 10
max_problem_size = 100
epochs = 10
iterations_per_epoch = 1000
n_ants = 15

min_cap = 10
max_cap = 50
if False:
    test_dataset = generate_variable_dataset('test', min_problem_size, max_problem_size, 50)
    sols = solve_dataset(test_dataset)

    for size in [20, 50, 100]:
        heuristic_network = neuralnetwork.GNN(32, 12)
        train(heuristic_network, size, epochs, iterations_per_epoch, n_ants)
        heuristic_network.eval()
        print(f'trained {size}')
        data = validate_best_variable(heuristic_network, min_problem_size, max_problem_size, n_ants, avg=False)
        torch.save(torch.tensor(data), f'results/ap/sizes-{size}.pt')
        print(f'Completed {size}')

    heuristic_network = neuralnetwork.GNN(32, 12)
    train_variable(heuristic_network, 20, 50, epochs, iterations_per_epoch, n_ants)
    heuristic_network.eval()
    print(f'trained 20-50')
    data_20_50 = validate_best_variable(heuristic_network, min_problem_size, max_problem_size, n_ants, avg=False)
    torch.save(torch.tensor(data_20_50), 'results/ap/sizes-20-50.pt')
    print(f'Completed 20-50')

    data_base = validate_best_variable(None, min_problem_size, max_problem_size, n_ants, avg=False)
    torch.save(torch.tensor(data_base), 'results/ap/sizes-base.pt')
    print(f'Completed base')


data_20 = torch.load('results/ap/sizes-20.pt')
data_50 = torch.load('results/ap/sizes-50.pt')
data_100 = torch.load('results/ap/sizes-100.pt')
data_20_50 = torch.load('results/ap/sizes-20-50.pt')
data_base = torch.load('results/ap/sizes-base.pt')

fig, ax = plt.subplots()
x = [i for i in range(min_problem_size, max_problem_size+5, 5)]

for data, name in [(data_20, 'AP20'), (data_50, 'AP50'), (data_100, 'AP100'), (data_20_50, 'AP20-50'), (data_base, 'Expert heuristic')]:
    data = (data - 1) * 100
    data = data.reshape((-1, 50))
    mean = data.mean(dim=1)
    std = data.std(dim=1)
    delta = 2.021 * std / (50 ** 0.5)
    ax.plot(x, mean, label=f'{name}')
    ax.fill_between(x, (mean-delta), (mean+delta), alpha=.2)

plt.xlabel('Problem size')
plt.ylabel('Optimality gap %')
plt.legend()
plt.title(f'Optimality gap against problem sizes')
plt.show()

print('this')









heuristic_network = neuralnetwork.GNN(32, 12)
# heuristic_network = neuralnetwork.GNN(40, 20)
problem_size = 100
k_sparse = None
epochs = 10
iterations_per_epoch = 100
# iterations_per_epoch = 200
n_ants = 15
train(heuristic_network, problem_size, epochs, iterations_per_epoch, n_ants, k_sparse=k_sparse)

heuristic_network.eval()
costs_base = []
costs_heu = []

costs_base_local = []
costs_heu_local = []

# costs_base_ls = []
# costs_heu_ls = []

# costs_base_ls_pre = []
# costs_heu_ls_pre = []

costs = []
for _ in range(5):
    task_costs = generate_problem_instance(problem_size)
    distances = get_distances(task_costs)
    pyg_data = convert_to_pyg_format(distances)


    sim = ACO(n_ants, distances)
    sim.run(100)
    costs_base.append(sim.costs)
    costs_base_local.append(sim.local_costs)

    # costs_base_ls.append(sim.generate_path_costs(sim.two_opt(sim.generate_best_path())))

    # visualiseWeights(data.x, sim.heuristics)
    # visualiseWeights(data.x, sim.pheromones * sim.heuristics)
    # visualiseWeights(distances, sim.pheromones * sim.heuristics, sim.generate_best_path())

    heuristic_vector = heuristic_network(pyg_data)
    heuristics = reshape_heuristic(heuristic_vector, pyg_data)
    sim_heu = ACO(n_ants, distances, heuristics=heuristics)
    sim_heu.run(100)
    costs_heu.append(sim_heu.costs)
    costs_heu_local.append(sim_heu.local_costs)

    # path = sim_heu.generate_best_path()
    # updated_path = sim_heu.two_opt(path)
    # print('Path, updated path')
    # print(path)
    # print(updated_path)
    # input()

    # costs_heu_ls.append(sim_heu.generate_path_costs(sim_heu.two_opt(sim_heu.generate_best_path())))



    # visualiseWeights(distances, sim_heu.pheromones * sim_heu.heuristics, sim_heu.generate_best_path())



costs_base = np.column_stack(tuple(costs_base))
costs_heu = np.column_stack(tuple(costs_heu))

costs_base_local = np.column_stack(tuple(costs_base_local))
costs_heu_local = np.column_stack(tuple(costs_heu_local))

fig, ax = plt.subplots()
ax.plot(np.mean(costs_base, axis=1), label='Base')
ax.plot(np.mean(costs_heu, axis=1), label='Heu')

ax.plot(np.mean(costs_base_local, axis=1), label='Base+LS')
ax.plot(np.mean(costs_heu_local, axis=1), label='Heu+LS')

# ax.axhline(y=np.mean(costs_base_ls), label='Base + LS', color='red')
# ax.axhline(y=np.mean(costs_heu_ls), label='Heu + LS', color='green')
# ax.axhline(y=np.mean(costs_base_ls_pre), label='base PRE', color='brown')
# ax.axhline(y=np.mean(costs_heu_ls_pre), label='Heu PRE', color='black')

plt.xlabel('No. Iterations')
plt.ylabel('Assignment Cost')
plt.legend()
plt.title(f'AP {problem_size}')
plt.show()

#  # Extrapolate
# costs_base = []
# costs_heu = []

# for _ in range(10):
#     data, distances = generate_problem_instance(problem_size * 2)
#     sim = ACO(15, distances)
#     sim.run(50)
#     costs_base.append(sim.costs)

#     heuristic_vector = heuristic_network(data)
#     heuristics = reshape_heuristic(heuristic_vector, data)
#     sim_heu = ACO(15, distances, heuristics=heuristics)
#     sim_heu.run(50)
#     costs_heu.append(sim_heu.costs)

# costs_base = np.column_stack(tuple(costs_base))
# costs_heu = np.column_stack(tuple(costs_heu))

# fig, ax = plt.subplots()
# ax.plot(np.mean(costs_base, axis=1), label='Base')
# ax.plot(np.mean(costs_heu, axis=1), label='Heu')

# plt.xlabel('No. Iterations')
# plt.ylabel('Path Length')
# plt.legend()
# plt.title(f'TSP Extrapolated')
# plt.show()
