import matplotlib.pyplot as plt
from tqdm import trange
import torch
import numpy as np

class ACO:
    def __init__(self, n_ants, distances, alpha=1, beta=1, evaportation_rate = 0.1, heuristics=None) -> None:
        self.n_ants = n_ants
        self.n_nodes = len(distances)
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaportation_rate
        self.distances = distances
        self.heuristics = heuristics if heuristics is not None else torch.stack((1/distances, 1/distances))
        self.pheromones = torch.ones_like(distances)
        self.costs = []
    
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
    
    @torch.no_grad()
    def update_pheromones(self, paths, path_costs):
        self.pheromones *= (1-self.evaporation_rate)
        for i in range(self.n_ants):
            ant_path_starts = paths[i]
            ant_path_ends = torch.roll(ant_path_starts, -1)
            ant_path_cost = path_costs[i]

            # Deposit pheromones proportional to the cost of the path
            self.pheromones[ant_path_starts, ant_path_ends] += 1./ant_path_cost
            self.pheromones[ant_path_ends, ant_path_starts] += 1./ant_path_cost
    

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
        costs = torch.sum(self.distances[hop_starts, hop_ends], dim=1) # #ants x 1
        return costs
    
    
    def generate_paths(self, gen_probs=False):
        current_positions = torch.randint(low=0, high=self.n_nodes, size=(self.n_ants,))
        valid_mask = torch.ones(size=(self.n_ants, self.n_nodes))

        paths = current_positions.reshape(self.n_ants, 1) # #ants x 1
        path_log_probs = torch.zeros_like(paths) # #ants x 1
        path_log_probs = []
        second = 0
        for _ in range(self.n_nodes-1):
            if _ > (self.n_nodes-1)/2:
                second = 1
            valid_mask = valid_mask.clone()
            valid_mask[torch.arange(self.n_ants), current_positions] = 0
            next_positions, next_log_probs = self.move(current_positions, valid_mask, second, gen_probs)
            current_positions = next_positions
            paths = torch.hstack((paths, current_positions.reshape(self.n_ants, 1))) # #ants x (2, 3, ..., #nodes)
            path_log_probs.append(next_log_probs)
        
        if gen_probs:
            path_log_probs = torch.stack(path_log_probs)

        return paths, path_log_probs
    
        
    def move(self, current_positions, valid_mask, second, gen_probs=False):
        # Get the heuristics and pheromones from each of the current positions
        # print(self.heuristics.size())
        # print(self.heuristics[:, :, 0].size())
        move_heuristics = self.heuristics[second, current_positions]
        # move_heuristics = move_heuristics[:, :, 0]
        # print(move_heuristics.size())
        move_pheromones = self.pheromones[current_positions]
        # print(move_pheromones.size())

        # Build the probabilities for each of these positions 
        move_probabilities = move_heuristics ** self.alpha * move_pheromones ** self.beta * valid_mask # #ants x #nodes

        # Generate random indices (moves) based on the probabilities
        moves = torch.multinomial(move_probabilities, 1).squeeze()

        log_probabilites = None
        if gen_probs:
            probabilites = move_probabilities[torch.arange(len(move_probabilities)), moves] / torch.sum(move_probabilities, axis=1)
            log_probabilites = torch.log(probabilites)

        return moves, log_probabilites

def example_run():
    size = 20
    nodes = torch.rand(size=(size, 2))
    distances = torch.sqrt(((nodes[:, None] - nodes[None, :]) ** 2).sum(2))
    distances[torch.arange(size), torch.arange(size)] = 1e9


    costs = []
    for _ in range(3):
        sim = ACO(50, distances)
        sim.run(50)
        costs.append(sim.costs)
    costs = np.column_stack(tuple(costs))
    fig, ax = plt.subplots()
    ax.plot(np.mean(costs, axis=1), label='average')
    plt.legend()
    plt.xlabel('No. Iterations')
    plt.ylabel('Path Length')
    plt.title(f'TSP{size}')
    plt.show()

# example_run()
