import matplotlib.pyplot as plt
from tqdm import trange
import torch
import numpy as np
from utils import visualiseWeights

class ACO:
    def __init__(self, n_ants, distances, alpha=1, beta=1, evaportation_rate = 0.1, heuristics=None) -> None:
        self.n_ants = n_ants
        self.n_nodes = len(distances)
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaportation_rate
        self.distances = distances
        self.heuristics = heuristics if heuristics is not None else 1/distances
        self.pheromones = torch.ones_like(distances)
        self.costs = []
        self.local_costs = []
    
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

        # local_paths = self.two_opt(paths)
        # local_path_costs = self.generate_path_costs(local_paths)
        # self.local_costs.append(torch.mean(local_path_costs).item())
    
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

        for _ in range(self.n_nodes-1):
            valid_mask = valid_mask.clone()
            valid_mask[torch.arange(self.n_ants), current_positions] = 0
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

    def local_search(self, paths):
        if paths.dim() != 2:
            paths = paths.unsqueeze(0) # Now we have #ants x #nodes paths
        # We now do a very simple pass through, possibly swapping consecutive nodes
        current_costs = self.generate_path_costs(paths)
        for index in range(self.n_nodes):
            # Try swapping indices (index) and (index+1)
            from_index = index
            to_index = (index+1)%self.n_nodes
            modified_paths = paths.clone()
            modified_paths[:, [from_index, to_index]] = modified_paths[:, [to_index, from_index]]
            updated_costs = self.generate_path_costs(modified_paths)
            improved_paths = updated_costs < current_costs

            paths[improved_paths] = modified_paths[improved_paths]
            current_costs = torch.min(current_costs, updated_costs)
        return paths
    
    def two_opt_original(self, paths):
        if paths.dim() != 2:
            paths = paths.unsqueeze(0) # Now we have #ants x #nodes paths
        # For every pair of edges, we check whether swapping them would improve the solution
        for first_index in range(self.n_nodes-2):
            for second_index in range(first_index+2, self.n_nodes):
                first_from = first_index
                first_to = first_index+1 # Can never wrap around
                second_from = second_index
                second_to = (second_index+1)%self.n_nodes
                # We have a situation like ... (1st) (1st+1) X ... Z (2nd)   (2nd+1) ... 
                # We now test the change   ... (1st) (2nd)   Z ... X (1st+1) (2nd+1) ...

                # Cost change = + 2 new edge costs - 2 old edge costs
                cost_change = (self.distances[paths[:, first_from], paths[:, second_from]] 
                               + self.distances[paths[:, first_to], paths[:, second_to]]
                               - self.distances[paths[:, first_from], paths[:, first_to]]
                               - self.distances[paths[:, second_from], paths[:, second_to]])

                improved_tours = cost_change < 0
                paths[improved_tours, first_to:second_to] = paths[improved_tours, first_to:second_to].flip(1)
                
        return paths
    
    def two_opt(self, paths, rounds=1):
        if paths.dim() != 2:
            paths = paths.unsqueeze(0) # Now we have #some n x #nodes paths
        num_paths = paths.size()[0]
        for _ in range(rounds):
            # store a matrix, for each ant the best improvement, 1st and 2nd #ants x 3
            optimisation_scores = torch.zeros(size=(num_paths,))
            optimisation_data = torch.zeros(size=(num_paths, 2))
            # For every pair of edges, we check whether swapping them would improve the solution
            for first_index in range(self.n_nodes-2):
                for second_index in range(first_index+2, self.n_nodes):
                    first_from = first_index
                    first_to = first_index+1 # Can never wrap around
                    second_from = second_index
                    second_to = (second_index+1)%self.n_nodes
                    # We have a situation like ... (1st) (1st+1) X ... Z (2nd)   (2nd+1) ... 
                    # We now test the change   ... (1st) (2nd)   Z ... X (1st+1) (2nd+1) ...
                    # So reverse the order of            \_____________________/

                    # Cost change = + 2 new edge costs - 2 old edge costs
                    cost_change = (self.distances[paths[:, first_from], paths[:, second_from]] 
                                + self.distances[paths[:, first_to], paths[:, second_to]]
                                - self.distances[paths[:, first_from], paths[:, first_to]]
                                - self.distances[paths[:, second_from], paths[:, second_to]])
                    

                    improved_tours = cost_change < optimisation_scores
                    optimisation_scores[improved_tours] = cost_change[improved_tours]
                    optimisation_data[improved_tours, 0] = first_index
                    optimisation_data[improved_tours, 1] = second_index

            improved_tours = optimisation_scores < 0 # The ants whose tours we've been able to improve
            first_froms = optimisation_data[:, 0].long()
            second_froms = optimisation_data[:, 1].long()
            first_tos = first_froms+1
            second_tos = (second_froms+1)%self.n_nodes

            for ant_index in range(num_paths):
                if improved_tours[ant_index]:
                    # Apply the two opt change for all ants whose tours were improved
                    paths[ant_index, first_tos[ant_index]:second_tos[ant_index]] = paths[ant_index, first_tos[ant_index]:second_tos[ant_index]].flip(0)

        return paths


def example_run():
    size = 25
    nodes = torch.rand(size=(size, 2))
    # torch.save(nodes, 'temp.pt')
    nodes = torch.load('temp.pt')
    distances = torch.sqrt(((nodes[:, None] - nodes[None, :]) ** 2).sum(2))
    distances[torch.arange(size), torch.arange(size)] = 1e9


    # costs = []
    # for _ in range(1):
    #     sim = ACO(5, distances)
    #     sim.run(150)
    #     costs.append(sim.costs)
    sim = ACO(15, distances)
    sim.run(100)
    
    paths, costs, _ = sim.generate_paths_and_costs()
    paths = paths[0]
    visualiseWeights(nodes, sim.pheromones, paths)
    for _ in range(5):
        paths = sim.two_opt(paths)[0]
        visualiseWeights(nodes, sim.pheromones, paths)

        
        
    # visualiseWeights(nodes, sim.pheromones * sim.heuristics)
    # visualiseWeights(nodes, sim.pheromones, f=True)
    # for _ in range(10):
    #     sim.run(10)
    #     # visualiseWeights(nodes, sim.pheromones * sim.heuristics)
    #     visualiseWeights(nodes, sim.pheromones)

    # paths, costs, _ = sim.generate_paths_and_costs()
    # print('Costs before:', torch.mean(costs), torch.mean(sim.generate_path_costs(paths)))
    # paths = sim.two_opt(paths)
    # # print(paths, 'new paths')
    # print('Costs after:', torch.mean(sim.generate_path_costs(paths)))
    # exit()

    # path = sim.generate_best_path()
    # visualiseWeights(nodes, sim.pheromones * sim.heuristics, path)
    # path = sim.two_opt(path, 1)[0]
    # visualiseWeights(nodes, sim.pheromones * sim.heuristics, path)
    
    # costs = np.column_stack(tuple(costs))
    # fig, ax = plt.subplots()
    # ax.plot(np.mean(costs, axis=1), label='average')
    # plt.legend()
    # plt.xlabel('No. Iterations')
    # plt.ylabel('Path Length')
    # plt.title(f'TSP{size}')
    # plt.show()


example_run()
