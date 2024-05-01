import matplotlib.pyplot as plt
from torch_geometric.data import Data
from tqdm import trange
import torch
import numpy as np
from utils import visualiseWeights

class ACO:
    def __init__(self, n_ants, distances, demands, alpha=1, beta=1, evaportation_rate = 0.1, heuristics=None, capacity=25) -> None:
        self.n_ants = n_ants
        self.n_nodes = len(distances)
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaportation_rate
        self.distances = distances
        self.demands = demands
        self.heuristics = heuristics if heuristics is not None else 1/distances
        self.pheromones = torch.ones_like(distances)
        self.costs = []
        self.capacity = capacity
        self.best_cost = torch.inf
    
    @torch.no_grad()
    def run(self, n_iterations, verbose=True):
        if verbose:
            for _ in (pbar := trange(n_iterations)):
                self.run_iteration()
                pbar.set_description(f'{round(self.costs[-1], 2)}')
        else:
            for _ in range(n_iterations):
                self.run_iteration()

    @torch.no_grad()
    def run_iteration(self):
        paths, path_costs, _ = self.generate_paths_and_costs() # We disregard the probs here, not needed
        self.update_pheromones(paths, path_costs)
        self.costs.append(torch.mean(path_costs).item())

        best_iteration_cost = torch.min(path_costs).item()
        self.best_cost = min(best_iteration_cost, self.best_cost)
    
    @torch.no_grad()
    def update_pheromones(self, paths, path_costs):
        self.pheromones *= (1-self.evaporation_rate)
        for i in range(self.n_ants):
            ant_path_starts = paths[i]
            ant_path_ends = torch.roll(ant_path_starts, -1)
            ant_path_cost = path_costs[i]

            # Deposit pheromones proportional to the cost of the path
            self.pheromones[ant_path_starts[:-1], ant_path_ends[:-1]] += 1./ant_path_cost
    

    def generate_paths_and_costs(self, gen_probs=False):
        paths, path_log_probs = self.generate_paths(gen_probs)
        costs = self.generate_path_costs(paths)
        return paths, costs, path_log_probs

    def generate_best_path(self):
        paths, costs, _ = self.generate_paths_and_costs()
        # print(paths.shape, costs.shape)
        min_index = torch.argmin(costs).item()
        best_path = paths[min_index]
        # print(best_path.shape)
        return best_path


    @torch.no_grad()
    def generate_path_costs(self, paths):
        hop_starts = paths
        hop_ends = torch.roll(hop_starts, -1, dims=1)
        return torch.sum(self.distances[hop_starts[:, :-1], hop_ends[:, :-1]], dim=1)

    def update_mask(self, mask, current_positions):
        # Locations just visited no longer valid
        mask[torch.arange(self.n_ants), current_positions] = 0
        # Completed if can't go to any other node and are at dummy node
        completed_agents = (mask[:, 1:]==0).all(dim=1) * (current_positions==0)
        valid_to_dummy = torch.logical_or(completed_agents, (current_positions != 0))
        # Completed agents and those not at dummy node now can always visit the dummy node
        mask[valid_to_dummy, 0] = 1
        return mask
    
    def done(self, valid_mask, current_positions):
        # Want to verify we've visited everywhere and we're currently at the depot
        visited_all_locations = (valid_mask[:, 1:] == 0).all()
        all_at_depot = (current_positions[:] == 0).all()
        return visited_all_locations and all_at_depot
    
    def get_remaining_capacities(self, current_positions, used_capacity):
        # Reset used capacities when at depot
        used_capacity[current_positions==0] = 0
        # We've used up what's required at the current position
        used_capacity = used_capacity + self.sizes[current_positions]

        # Initialise to all 0ss
        within_capacity = torch.zeros(size=(self.n_ants, self.n_nodes)) # #n_ants x #n_nodes
        
        available_capacity = self.capacity - used_capacity # #n_ants x 1

        # Where the sizes are <= available, we can visit that node
        within_capacity[self.sizes.t().repeat(self.n_ants, 1) <= available_capacity.repeat(1, self.n_nodes)] = 1
        
        return used_capacity, within_capacity
    
    
    def generate_paths(self, gen_probs=False):
        current_positions = torch.randint(low=0, high=1, size=(self.n_ants,))
        valid_mask = torch.ones(size=(self.n_ants, self.n_nodes))
        used_capacity = torch.zeros((self.n_ants, 1))


        paths = current_positions.reshape(self.n_ants, 1) # #ants x 1
        path_log_probs = torch.zeros_like(paths) # #ants x 1
        path_log_probs = []

        while not self.done(valid_mask, current_positions):
            valid_mask = valid_mask.clone()
            valid_mask = self.update_mask(valid_mask, current_positions)
            used_capacity, capacity_mask = self.get_remaining_capacities(current_positions, used_capacity)

            next_positions, next_log_probs = self.move(current_positions, valid_mask, capacity_mask, gen_probs)
            current_positions = next_positions
            paths = torch.hstack((paths, current_positions.reshape(self.n_ants, 1))) # #ants x (2, 3, ..., #nodes)
            path_log_probs.append(next_log_probs)
        # print(paths)
        if gen_probs:
            path_log_probs = torch.stack(path_log_probs)

        return paths, path_log_probs
    
        
    def move(self, current_positions, valid_mask, capacity_mask, gen_probs=False):
        # Get the heuristics and pheromones from each of the current positions
        move_heuristics = self.heuristics[current_positions]
        move_pheromones = self.pheromones[current_positions]

        # Build the probabilities for each of these positions 
        move_probabilities = move_heuristics ** self.alpha * move_pheromones ** self.beta * valid_mask * capacity_mask # #ants x #nodes

        # Generate random indices (moves) based on the probabilities
        moves = torch.multinomial(move_probabilities, 1).squeeze()

        log_probabilites = None
        if gen_probs:
            probabilites = move_probabilities[torch.arange(len(move_probabilities)), moves] / torch.sum(move_probabilities, axis=1)
            log_probabilites = torch.log(probabilites)

        return moves, log_probabilites


def generate_problem_instance(size):
    # Creates n random node positions, preprended with the depot coords, with their demands
    depot_coords = torch.tensor([0.5, 0.5]).unsqueeze(dim=0)
    customer_coords = torch.rand(size=(size, 2))
    all_coords = torch.cat((depot_coords, customer_coords)) # n+1 x 2

    depot_demands = torch.zeros(size=(1, 1))
    customer_demands = torch.randint(1, 10, size=(size, 1))
    all_demands = torch.cat((depot_demands, customer_demands)) # n+1 x 1

    return all_coords, all_demands


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


def example_run():
    size = 10
    nodes, demands = generate_problem_instance(size)
    distances = torch.sqrt(((nodes[:, None] - nodes[None, :]) ** 2).sum(2))
    distances[torch.arange(size+1), torch.arange(size+1)] = 1e9
    distances[0, 0] = 1e-10
    costs = []
    for _ in range(1):
        sim = ACO(15, distances, demands)
        sim.run(500)
        costs.append(sim.costs)
    costs = np.column_stack(tuple(costs))
    fig, ax = plt.subplots()
    ax.plot(np.mean(costs, axis=1), label='average')
    plt.legend()
    plt.xlabel('No. Iterations')
    plt.ylabel('Path Length')
    plt.title(f'TSP{size}')
    plt.show()

    visualiseWeights(nodes, sim.heuristics * sim.pheromones, path=sim.generate_best_path(), l=demands)

# example_run()