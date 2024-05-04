import matplotlib.pyplot as plt
from torch_geometric.data import Data
from tqdm import trange
import torch
import numpy as np
from utils import visualiseWeights, generate_problem_instance, get_distances

class ACO:
    def __init__(self, n_ants, distances, edges, alpha=1, beta=1, evaportation_rate = 0.1, heuristics=None) -> None:
        self.n_ants = n_ants
        self.n_nodes = len(distances)
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaportation_rate
        self.distances = distances
        # Our basic heuristic for a node is the #edges that node has
        # Heuristics don't matter which edge you come from, so u->x = v->x
        self.heuristics = torch.clamp(heuristics.clone(), min=1e-20) if heuristics is not None else torch.sum(distances, dim=1).repeat(self.n_nodes, 1)+0.1
        if heuristics is not None:
            heuristics[:, 0] = 1 # Heuristic to return to depot
        self.pheromones = torch.ones_like(distances)
        self.costs = []
        self.constraints = 1 - distances # 1 for all valid nodes to visit from i
        # Will create a 2 x #ants x #edges matrix. Values in each plane are corresponing vertices of an edge
        self.edge_nodes = torch.stack((edges[:, 0].expand(n_ants, -1), edges[:, 1].expand(n_ants, -1)))
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
        # Cost is the number vertices used to cover (eg. vertices visited)
        vertices_used = torch.zeros(size=(self.n_ants,))
        for i in range(self.n_ants):
            ant_tour_length = len(torch.unique_consecutive(paths[i]))
            ant_vertices_used = ant_tour_length - 2 # Ignore the dummy node on either end
            vertices_used[i] = ant_vertices_used

        return vertices_used
    

    def update_mask(self, mask, current_positions, covered_edges):
        mask[torch.arange(self.n_ants), current_positions] = 0 # Places just visited now not valid
        valid_cover = torch.all(covered_edges, dim=1)
        mask[valid_cover, :] = 0 # Ants which have covered every edge can't continue
        mask[valid_cover, 0] = 1 # Ants which have covered every edge can visit the depot
        return mask
    
    
    def done(self, covered_edges, current_positions):
        # Want to verify we've covered every edge
        covered_all_edges = (covered_edges == True).all()
        at_depot = (current_positions == 0).all()
        return covered_all_edges and at_depot
    
    def update_covered_edges(self, covered_edges, current_positions):
        positions_expanded = current_positions.unsqueeze(0).t().expand(-1, self.edge_nodes.size()[2]) # Expand to #ants x #edges
        newly_covered_edges = torch.any(torch.eq(self.edge_nodes, positions_expanded), dim=0) #ants x #edges
        covered_edges = torch.logical_or(covered_edges, newly_covered_edges)
        return covered_edges
    
    
    def generate_paths(self, gen_probs=False):
        current_positions = torch.randint(low=0, high=1, size=(self.n_ants,))
        # current_positions = torch.zeros((self.n_ants,))
        valid_mask = torch.ones(size=(self.n_ants, self.n_nodes))
        covered_edges = torch.zeros(size=(self.n_ants, self.edge_nodes.size()[2]))


        paths = current_positions.reshape(self.n_ants, 1) # #ants x 1
        path_log_probs = torch.zeros_like(paths) # #ants x 1
        path_log_probs = []

        while not self.done(covered_edges, current_positions):
            valid_mask = valid_mask.clone()
            valid_mask = self.update_mask(valid_mask, current_positions, covered_edges)

            next_positions, next_log_probs = self.move(current_positions, valid_mask, gen_probs)
            current_positions = next_positions
            paths = torch.hstack((paths, current_positions.reshape(self.n_ants, 1))) # #ants x (2, 3, ..., #nodes)
            path_log_probs.append(next_log_probs)
            covered_edges = self.update_covered_edges(covered_edges, current_positions)

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
        try:
            moves = torch.multinomial(move_probabilities, 1).squeeze()
        except RuntimeError:
            print('Error encountered')
            move_probabilities = move_pheromones ** self.beta * valid_mask # #ants x #nodes
            moves = torch.multinomial(move_probabilities, 1).squeeze()

        log_probabilites = None
        if gen_probs:
            probabilites = move_probabilities[torch.arange(len(move_probabilities)), moves] / torch.sum(move_probabilities, axis=1)
            log_probabilites = torch.log(probabilites)

        return moves, log_probabilites



def example_run():
    size = 20
    edges = generate_problem_instance(size, p=0.05)
    distances = get_distances(edges, size)
    print('EDGES')
    print(edges)

    costs = []
    for _ in range(5):
        sim = ACO(15, distances, edges)
        sim.run(2000)
        costs.append(sim.costs)

    costs = np.column_stack(tuple(costs))
    fig, ax = plt.subplots()
    ax.plot(np.mean(costs, axis=1), label='average')
    plt.legend()
    plt.xlabel('No. Iterations')
    plt.ylabel('Path Length')
    plt.title(f'VCP{size}')
    plt.show()

    # visualiseWeights(nodes, sim.heuristics * sim.pheromones, path=sim.generate_best_path())

# example_run()
