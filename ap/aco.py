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
        self.heuristics = heuristics + 1e-10 if heuristics is not None else 1/(distances+1e-10)
        self.pheromones = torch.ones_like(distances)
        self.costs = []
        self.local_costs = []
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

        # local_paths = self.two_opt(paths)
        # local_path_costs = self.generate_path_costs(local_paths)
        # self.local_costs.append(torch.mean(local_path_costs).item())

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
            # self.pheromones[ant_path_ends, ant_path_starts] += 1./ant_path_cost
    

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
        # print(paths)
        hop_starts = paths
        hop_ends = torch.roll(hop_starts, -1, dims=1)
        # costs = torch.sum(self.distances[hop_starts, hop_ends], dim=1) # #ants x 1
        # return costs
        # print(hop_starts)
        # print(hop_ends)
        # print(self.distances[hop_starts[:, :-1], hop_ends[:, :-1]])
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
    
    def update_agent_task_mask(self, current_positions, agent_task_mask):
        size = self.distances.size()[0]//2
        agent_task_mask[:, :] = 0

        ants_at_dummy = current_positions == 0
        agent_task_mask[ants_at_dummy, 1:size+1] = 1

        ants_at_agent = torch.logical_and(current_positions > 0, current_positions < size+1)
        agent_task_mask[ants_at_agent, size+1:] = 1

        ants_at_task = current_positions > size
        agent_task_mask[ants_at_task, 0] = 1

        return agent_task_mask




    
    def update_capacity_mask(self, current_positions, used_capacity):
        capacity_mask = torch.ones(size=(self.n_ants, self.n_nodes))
        # update capacity
        used_capacity[current_positions==0] = 0
        used_capacity = used_capacity + self.demands[current_positions]
        # update capacity_mask
        remaining_capacity = self.capacity - used_capacity # (n_ants,)
        remaining_capacity_repeat = remaining_capacity.repeat(1, self.n_nodes) # (n_ants, p_size)
        demand_repeat = self.demands.t().repeat(self.n_ants, 1) # (n_ants, p_size)
        # print(self.demands.size())
        # print(remaining_capacity.size())
        # print(remaining_capacity_repeat.size())
        # print(demand_repeat.size())
        capacity_mask[demand_repeat > remaining_capacity_repeat] = 0
        
        return used_capacity, capacity_mask
    
    
    def generate_paths(self, gen_probs=False):
        current_positions = torch.randint(low=0, high=1, size=(self.n_ants,))
        # current_positions = torch.zeros((self.n_ants,))
        valid_mask = torch.ones(size=(self.n_ants, self.n_nodes))
        used_capacity = torch.zeros((self.n_ants, 1))
        # used_capacity, capacity_mask = self.update_capacity_mask(current_positions, used_capacity)
        agent_task_mask = torch.zeros_like(valid_mask)


        paths = current_positions.reshape(self.n_ants, 1) # #ants x 1
        path_log_probs = torch.zeros_like(paths) # #ants x 1
        path_log_probs = []

        # while not self.done(valid_mask, current_positions):
        for _ in range((self.n_nodes//2)*3):
            valid_mask = valid_mask.clone()
            valid_mask = self.update_mask(valid_mask, current_positions)
            # used_capacity, capacity_mask = self.update_capacity_mask(current_positions, used_capacity)
            agent_task_mask = agent_task_mask.clone()
            agent_task_mask = self.update_agent_task_mask(current_positions, agent_task_mask)

            # print('round', _)
            # print(valid_mask)
            # print(agent_task_mask)

            next_positions, next_log_probs = self.move(current_positions, valid_mask, agent_task_mask, gen_probs)
            current_positions = next_positions
            paths = torch.hstack((paths, current_positions.reshape(self.n_ants, 1))) # #ants x (2, 3, ..., #nodes)
            path_log_probs.append(next_log_probs)
        # print(paths)
        if gen_probs:
            path_log_probs = torch.stack(path_log_probs)

        return paths, path_log_probs
    
        
    def move(self, current_positions, valid_mask, agent_task_mask, gen_probs=False):
        # Get the heuristics and pheromones from each of the current positions
        move_heuristics = self.heuristics[current_positions]
        move_pheromones = self.pheromones[current_positions]

        # Build the probabilities for each of these positions 
        move_probabilities = move_heuristics ** self.alpha * move_pheromones ** self.beta * valid_mask * agent_task_mask # #ants x #nodes
        # print('NAN', move_heuristics.isnan().any())
        # exit()
        # print(move_heuristics)
        # print(move_pheromones)
        # print(valid_mask)
        # print(capacity_mask)
        # print(move_probabilities)

        # Generate random indices (moves) based on the probabilities
        # print(move_probabilities)
        moves = torch.multinomial(move_probabilities, 1).squeeze()

        log_probabilites = None
        if gen_probs:
            probabilites = move_probabilities[torch.arange(len(move_probabilities)), moves] / torch.sum(move_probabilities, axis=1)
            log_probabilites = torch.log(probabilites)

        return moves, log_probabilites

    def two_opt(self, paths):
        # print(paths)
        if paths.dim() != 2:
            paths = paths.unsqueeze(0) # Now we have #some n x #nodes paths
        num_paths = paths.size()[0]
        paths_length = paths.size()[1]
        for _ in range(3):
            # print('starting')
            # store a matrix, for each ant the best improvement, 1st agent swap, 2nd agent swap #ants x 3
            optimisation_scores = torch.zeros(size=(num_paths,))
            optimisation_data = torch.zeros(size=(num_paths, 2))
            # For every pair of agents, check if swapping their tasks would result in a better solution
            for first_index in range(1, paths_length-3, 3):
                for second_index in range(first_index+3, paths_length, 3):
                    # print(f'First index {first_index}, second index {second_index}')
                    first_agent = paths[:, first_index]
                    second_agent = paths[:, second_index]
                    first_task = paths[:, first_index+1]
                    second_task = paths[:, second_index+1]
                    # We have a situation like ... (1st) (1st+1) X ... Z (2nd)   (2nd+1) ... 
                    # We now test the change   ... (1st) (2nd)   Z ... X (1st+1) (2nd+1) ...

                    # Cost change = + 2 new edge costs - 2 old edge costs
                    cost_change = (self.distances[first_agent, second_task] 
                                + self.distances[second_agent, first_task] 
                                - self.distances[first_agent, first_task] 
                                - self.distances[second_agent, second_task])
                    
                    # print(cost_change)
                    # print(optimisation_scores)
                    improved_tours = cost_change < optimisation_scores
                    # print('imp, cost, optscores')
                    # print(improved_tours)
                    
                    optimisation_scores[improved_tours] = cost_change[improved_tours]
                    optimisation_data[improved_tours, 0] = first_index
                    optimisation_data[improved_tours, 1] = second_index

            improved_tours = optimisation_scores < 0 # The ants whose tours we've been able to improve
            first_indices = optimisation_data[:, 0].long()
            second_indices = optimisation_data[:, 1].long()

            # print('first, second')
            # print(improved_tours)
            # print(first_indices)
            # print(second_indices)

            # paths[:, [first_indices, second_indices]] = paths[:, [second_indices, first_indices]]

            for ant_index in range(num_paths):
                if improved_tours[ant_index]:
                    # print(f'Improvement made: {optimisation_scores[ant_index]}')
                    paths[ant_index, [first_indices[ant_index], second_indices[ant_index]]] = paths[ant_index, [second_indices[ant_index], first_indices[ant_index]]]

        return paths


def example_run():
    size = 5
    # task_costs = generate_problem_instance(size)
    # distances = get_distances(task_costs)

    costs = []
    for _ in range(1):
        task_costs = generate_problem_instance(size)
        distances = get_distances(task_costs)
        sim = ACO(25, distances)
        sim.run(150)
        costs.append(sim.costs)
        path = sim.generate_best_path()
        print(path)
        path = sim.two_opt(path)
        print(path)
        cost = sim.generate_path_costs(path).item()
        print()
        # visualiseWeights(distances, sim.heuristics * sim.pheromones, path=sim.generate_best_path())

    costs = np.column_stack(tuple(costs))
    fig, ax = plt.subplots()
    ax.plot(np.mean(costs, axis=1), label='average')
    ax.axhline(y=cost, label='Perfect')
    plt.legend()
    plt.xlabel('No. Iterations')
    plt.ylabel('Path Length')
    plt.title(f'TSP{size}')
    plt.show()

    # visualiseWeights(nodes, sim.heuristics * sim.pheromones, path=sim.generate_best_path())

# example_run()
