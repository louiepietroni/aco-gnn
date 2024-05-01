import matplotlib.pyplot as plt
from tqdm import trange
import torch
import numpy as np

class BeamSearch:
    def __init__(self, n_beams, beam_width, distances, heuristics=None) -> None:
        self.n_beams = n_beams
        self.beam_width = beam_width
        self.n_nodes = len(distances)
        self.distances = distances
        self.heuristics = heuristics if heuristics is not None else 1/distances
        self.costs = []
    
    @torch.no_grad()
    def run(self):
        for _ in (pbar := trange(self.n_nodes)):
            self.step()


    @torch.no_grad()
    def step(self):
        paths, path_costs, _ = self.generate_paths_and_costs() # We disregard the probs here, not needed
        self.update_pheromones(paths, path_costs)
        self.costs.append(torch.mean(path_costs).item())
    

    def generate_paths_and_costs(self, gen_probs=False):
        paths, path_log_probs = self.generate_paths(gen_probs)
        costs = self.generate_path_costs(paths)
        return paths, costs, path_log_probs


    @torch.no_grad()
    def generate_path_costs(self, paths):
        hop_starts = paths
        hop_ends = torch.roll(hop_starts, -1, dims=1)
        costs = torch.sum(self.distances[hop_starts, hop_ends], dim=1) # #ants x 1
        return costs
    
    
    def generate_paths(self, gen_probs=False):
        current_positions = torch.randint(low=0, high=self.n_nodes, size=(self.n_beams,))
        valid_mask = torch.ones(size=(self.n_beams, self.n_nodes))

        paths = current_positions.reshape(self.n_beams, 1) # #ants x 1
        path_log_probs = torch.zeros_like(paths) # #ants x 1
        path_log_probs = []

        for _ in range(self.n_nodes-1):
            valid_mask = valid_mask.clone()
            valid_mask[torch.arange(self.n_beams), current_positions] = 0
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
