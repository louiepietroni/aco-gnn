import matplotlib.pyplot as plt
from tqdm import trange
import torch
import numpy as np
from utils import generate_problem_instance, visualiseWeights

class ACO:
    def __init__(self, n_ants, distances, precedences, alpha=1, beta=1, evaportation_rate = 0.1, heuristics=None) -> None:
        self.n_ants = n_ants
        self.n_nodes = len(distances)
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaportation_rate
        self.distances = distances
        self.precedences = self.generate_precendence_matrix(precedences)
        self.heuristics = heuristics if heuristics is not None else 1/distances
        self.pheromones = torch.ones_like(distances)
        self.costs = []
        self.best_cost = torch.inf
    
    def generate_precendence_matrix(self, precedences):
        # #nodes x #nodes, what nodes we 'unlock' by going to this node
        precedence_matrix = torch.zeros(size=(self.n_nodes, self.n_nodes))
        if len(precedences) != 0:
            precedences = torch.tensor(precedences).t()
            precedence_matrix[precedences[0], precedences[1]] = 1
        return precedence_matrix

    
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
        costs = torch.sum(self.distances[hop_starts[:, :-1], hop_ends[:, :-1]], dim=1) # #ants x 1
        return costs
    
    def update_precedence_mask(self, positions, precendence_mask, precedence_helper):

        precedence_helper += self.precedences[positions].int() # We have made one more contribution to precedences in cur. positions

        a = torch.sum(self.precedences, dim=0).repeat(self.n_ants, 1)

        inds = precedence_helper ==  a

        precendence_mask[inds] = 1

        return precendence_mask, precedence_helper



    
 
    def generate_paths(self, gen_probs=False):
        current_positions = torch.randint(low=0, high=1, size=(self.n_ants,))
        valid_mask = torch.ones(size=(self.n_ants, self.n_nodes))

        precendence_mask = ~torch.any(self.precedences, dim=0)
        precendence_mask = precendence_mask.repeat(self.n_ants, 1).int()
        precendence_helper = torch.zeros_like(precendence_mask) # Will help

        paths = current_positions.reshape(self.n_ants, 1) # #ants x 1
        path_log_probs = torch.zeros_like(paths) # #ants x 1
        path_log_probs = []

        for _ in range(self.n_nodes-1):
            valid_mask = valid_mask.clone()
            valid_mask[torch.arange(self.n_ants), current_positions] = 0
            precendence_mask = precendence_mask.clone()
            precendence_mask, precendence_helper = self.update_precedence_mask(current_positions, precendence_mask, precendence_helper) # It's important we do this before the first iteration

            next_positions, next_log_probs = self.move(current_positions, valid_mask, precendence_mask, gen_probs)
            current_positions = next_positions
            paths = torch.hstack((paths, current_positions.reshape(self.n_ants, 1))) # #ants x (2, 3, ..., #nodes)
            path_log_probs.append(next_log_probs)
        
        if gen_probs:
            path_log_probs = torch.stack(path_log_probs)

        return paths, path_log_probs
    
        
    def move(self, current_positions, valid_mask, precedence_mask, gen_probs=False):
        # Get the heuristics and pheromones from each of the current positions
        move_heuristics = self.heuristics[current_positions]
        move_pheromones = self.pheromones[current_positions]

        # Build the probabilities for each of these positions 
        move_probabilities = move_heuristics ** self.alpha * move_pheromones ** self.beta * valid_mask * precedence_mask# #ants x #nodes

        # Generate random indices (moves) based on the probabilities
        moves = torch.multinomial(move_probabilities, 1).squeeze()

        log_probabilites = None
        if gen_probs:
            probabilites = move_probabilities[torch.arange(len(move_probabilities)), moves] / torch.sum(move_probabilities, axis=1)
            log_probabilites = torch.log(probabilites)

        return moves, log_probabilites

def example_run():
    size = 20
    nodes, precedences = generate_problem_instance(size, 0.01)
    distances = torch.sqrt(((nodes[:, None] - nodes[None, :]) ** 2).sum(2))
    distances[torch.arange(size), torch.arange(size)] = 1e9
    print(precedences)
    sim = ACO(3, distances, precedences)
    sim.run(20)
    visualiseWeights(nodes, sim.heuristics*sim.pheromones, precedences=precedences, path=sim.generate_best_path())

# example_run()
