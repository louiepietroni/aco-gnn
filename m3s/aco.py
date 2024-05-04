import matplotlib.pyplot as plt
from torch_geometric.data import Data
from tqdm import trange
import torch
import numpy as np
from utils import visualiseWeights, generate_problem_instance, get_distances

class ACO:
    def __init__(self, n_ants, distances, clauses, alpha=1, beta=1, evaportation_rate = 0.1, heuristics=None) -> None:
        self.n_ants = n_ants
        self.n_nodes = len(distances)
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaportation_rate
        self.distances = distances
        self.clauses = clauses
        # heuristic for nodes which DON'T share a clause with this variable
        self.heuristics = heuristics+1e-7 if heuristics is not None else 1-distances + 0.1
        self.pheromones = torch.ones_like(distances)
        self.costs = []
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
            self.pheromones[ant_path_starts, ant_path_ends] += 1./(1+ant_path_cost)
    

    def generate_paths_and_costs(self, gen_probs=False):
        paths, path_log_probs = self.generate_paths(gen_probs)
        costs = self.generate_path_costs(paths)
        return paths, costs, path_log_probs

    def generate_best_path(self):
        paths, costs, _ = self.generate_paths_and_costs()
        min_index = torch.argmin(costs).item()
        best_path = paths[min_index]
        return best_path


    @torch.no_grad()
    def generate_path_costs(self, paths):
        # The cost is defined as the number of unsatisfied clauses
        unsatisfied_clauses = torch.zeros(size=(self.n_ants,))
        for i in range(self.n_ants):
            ant_satisfied_clauses = torch.any(torch.isin(self.clauses, paths[i]), dim=1)
            ant_number_unsat_clauses = torch.sum(ant_satisfied_clauses == False)
            unsatisfied_clauses[i] = ant_number_unsat_clauses
        return unsatisfied_clauses


    def update_mask(self, mask, current_positions):
        mask[torch.arange(self.n_ants), current_positions] = 0 # Places just visited now not valid
        if 0 in current_positions:
            # Just left the dummy node, no negation to this variable
            return mask
        negated_positions = (current_positions + 3) % (self.n_nodes-1) # Get the equvialent negated variables too
        negated_positions[negated_positions == 0] = self.n_nodes-1 # This negated n will be slightly wrong
        mask[torch.arange(self.n_ants), negated_positions] = 0 # Also can't visit negation of variable
        
        return mask
    
    def done(self, valid_mask, current_positions):
        # Want to verify we've visited everywhere and we're currently at the depot
        visited_all_locations = (valid_mask[:, 1:] == 0).all()
        all_at_depot = (current_positions[:] == 0).all()
        return visited_all_locations and all_at_depot
    
    
    def generate_paths(self, gen_probs=False):
        current_positions = torch.randint(low=0, high=1, size=(self.n_ants,))
        valid_mask = torch.ones(size=(self.n_ants, self.n_nodes))


        paths = current_positions.reshape(self.n_ants, 1) # #ants x 1
        path_log_probs = torch.zeros_like(paths) # #ants x 1
        path_log_probs = []

        for _ in range(self.n_nodes//2):
            # print(_)
            valid_mask = valid_mask.clone()
            valid_mask = self.update_mask(valid_mask, current_positions)

            next_positions, next_log_probs = self.move(current_positions, valid_mask, gen_probs)
            current_positions = next_positions
            paths = torch.hstack((paths, current_positions.reshape(self.n_ants, 1))) # #ants x (2, 3, ..., #nodes)
            path_log_probs.append(next_log_probs)

        if gen_probs:
            path_log_probs = torch.stack(path_log_probs)

        return paths, path_log_probs
    
        
    def move(self, current_positions, valid_mask, gen_probs=False):
        # Get the heuristics and pheromones from each of the current positions
        move_heuristics = self.heuristics[current_positions]
        move_pheromones = self.pheromones[current_positions]

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
    clauses = generate_problem_instance(size)
    distances = get_distances(clauses, size)


    costs = []
    for _ in range(5):
        sim = ACO(15, distances, clauses)
        sim.run(1000)
        costs.append(sim.costs)

    costs = np.column_stack(tuple(costs))
    fig, ax = plt.subplots()
    ax.plot(np.mean(costs, axis=1), label='average')
    plt.legend()
    plt.xlabel('No. Iterations')
    plt.ylabel('Unsatisfied Clauses')
    plt.title(f'M3S{size}')
    plt.show()

    # visualiseWeights(nodes, sim.heuristics * sim.pheromones, path=sim.generate_best_path())

# example_run()
