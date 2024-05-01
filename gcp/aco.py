import matplotlib.pyplot as plt
from torch_geometric.data import Data
from tqdm import trange
import torch
import numpy as np
from utils import visualiseWeights, generate_problem_instance, get_distances

class ACO:
    def __init__(self, n_ants, distances, alpha=1, beta=1, evaportation_rate = 0.1, heuristics=None) -> None:
        self.n_ants = n_ants
        self.n_nodes = len(distances)
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaportation_rate
        self.distances = distances
        # Our basic heuristic for a node is the #edges that node has
        # Heuristics don't matter which edge you come from, so u->x = v->x
        self.heuristics = heuristics + 1e-9 if heuristics is not None else torch.sum(distances, dim=1).repeat(self.n_nodes, 1)+0.5
        if heuristics is None:
            self.heuristics[:, 0] = 1e-9
        self.pheromones = torch.ones_like(distances)
        self.costs = []
        self.constraints = 1 - distances # 1 for all valid nodes to visit from i
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
        # Cost is the number of colours used
        colours_used = torch.zeros(size=(self.n_ants,))
        for i in range(self.n_ants):
            ant_tour_length = len(torch.unique_consecutive(paths[i]))
            ant_bins_used = ant_tour_length - self.n_nodes
            colours_used[i] = ant_bins_used

        return colours_used
    

    def update_mask(self, mask, current_positions):
        # Locations just visited no longer valid
        mask[torch.arange(self.n_ants), current_positions] = 0
        # Completed if can't go to any other node and are at dummy node
        completed_agents = (mask[:, 1:]==0).all(dim=1) * (current_positions==0)
        valid_to_dummy = torch.logical_or(completed_agents, (current_positions != 0))
        # Completed agents and those not at dummy node now can always visit the dummy node
        mask[valid_to_dummy, 0] = 1

        return mask
    
    def update_valid_colour_mask(self, colour_mask, current_positions):
        # If we're at the dummy node, reset to allow all nodes
        colour_mask[current_positions == 0, :] = 1
        # Any nodes which have edges to where we are now can't be visited
        for i in range(self.n_ants):
            colour_mask[i] = colour_mask[i] * self.constraints[current_positions[i]]
        
        return colour_mask

    
    def done(self, valid_mask, current_positions):
        # Want to verify we've visited everywhere and we're currently at the depot
        visited_all_locations = (valid_mask[:, 1:] == 0).all()
        all_at_depot = (current_positions[:] == 0).all()
        return visited_all_locations and all_at_depot

    
    def generate_paths(self, gen_probs=False):
        current_positions = torch.randint(low=0, high=1, size=(self.n_ants,))
        valid_mask = torch.ones(size=(self.n_ants, self.n_nodes))
        valid_colour_mask = torch.ones(size=(self.n_ants, self.n_nodes))


        paths = current_positions.reshape(self.n_ants, 1) # #ants x 1
        path_log_probs = torch.zeros_like(paths) # #ants x 1
        path_log_probs = []

        while not self.done(valid_mask, current_positions):
            # print('Starting move')
            valid_mask = valid_mask.clone()
            valid_mask = self.update_mask(valid_mask, current_positions)
            valid_colour_mask = valid_colour_mask.clone()
            valid_colour_mask = self.update_valid_colour_mask(valid_colour_mask, current_positions)

            next_positions, next_log_probs = self.move(current_positions, valid_mask, valid_colour_mask, gen_probs)
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



def example_run():
    size = 50
    edges = generate_problem_instance(size, p=0.8)
    distances = get_distances(edges, size)
    
    costs = []
    for _ in range(5):
        sim = ACO(15, distances)
        sim.run(300)
        costs.append(sim.costs)

    costs = np.column_stack(tuple(costs))
    fig, ax = plt.subplots()
    ax.plot(np.mean(costs, axis=1), label='average')
    plt.legend()
    plt.xlabel('No. Iterations')
    plt.ylabel('Path Length')
    plt.title(f'TSP{size}')
    plt.show()

    # visualiseWeights(nodes, sim.heuristics * sim.pheromones, path=sim.generate_best_path())

# example_run()
