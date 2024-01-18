import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

class ACO:
    def __init__(self, n_ants, distances, alpha=1, beta=1, evaportation_rate = 0.1, heuristics=None) -> None:
        self.n_ants = n_ants
        self.n_nodes = len(distances)
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaportation_rate
        self.distances = distances
        self.heuristics = heuristics if heuristics is not None else 1/distances
        self.pheromones = np.ones(shape=distances.shape)

        self.costs = []
    
    def run(self, n_iterations):
        for _ in (pbar := trange(n_iterations)):
            self.run_iteration()
            pbar.set_description(f'{round(self.costs[-1], 2)}')

    def run_iteration(self):
        paths, path_costs, _ = self.generate_paths_and_costs() # We disregard the probs here, not needed
        self.update_pheromones(paths, path_costs)
        self.costs.append(np.average(path_costs))
    
    def update_pheromones(self, paths, path_costs):
        self.pheromones *= (1-self.evaporation_rate)
        for i in range(self.n_ants):
            ant_path_starts = paths[i]
            ant_path_ends = np.roll(ant_path_starts, -1)
            ant_path_cost = path_costs[i]

            # Deposit pheromones proportional to the cost of the path
            self.pheromones[ant_path_starts, ant_path_ends] += 1./ant_path_cost
            self.pheromones[ant_path_ends, ant_path_starts] += 1./ant_path_cost
    
    def generate_paths_and_costs(self, gen_probs=False):
        paths, path_log_probs = self.generate_paths(gen_probs)
        costs = self.generate_path_costs(paths)
        return paths, costs, path_log_probs

    def generate_path_costs(self, paths):
        hop_starts = paths
        hop_ends = np.roll(hop_starts, -1, axis=1)
        costs = np.sum(self.distances[hop_starts, hop_ends], axis=1) # #ants x 1
        return costs
    
    def generate_paths(self, gen_probs=False):
        current_positions = np.random.randint(low=0, high=self.n_nodes, size=(self.n_ants,))
        valid_mask = np.ones(shape=(self.n_ants, self.n_nodes))
        valid_mask[np.arange(self.n_ants), current_positions] = 0

        paths = current_positions.reshape(self.n_ants, 1) # #ants x 1
        path_log_probs = np.zeros_like(paths) # #ants x 1

        for _ in range(self.n_nodes-1):
            next_positions, next_log_probs = self.move(current_positions, valid_mask, gen_probs)
            current_positions = next_positions
            valid_mask[np.arange(self.n_ants), current_positions] = 0
            paths = np.hstack((paths, current_positions.reshape(self.n_ants, 1))) # #ants x (2, 3, ..., #nodes)
            if gen_probs:
                path_log_probs += next_log_probs

        return paths, path_log_probs
        

    def move(self, current_positions, valid_mask, gen_probs=False):
        # Get the heuristics and pheromones from each of the current positions
        move_heuristics = self.heuristics[current_positions]
        move_pheromones = self.pheromones[current_positions]

        # Build the probabilities for each of these positions 
        move_probabilities = move_heuristics ** self.alpha * move_pheromones ** self.beta * valid_mask # #ants x #nodes

        # Generate random indices (moves) based on the probabilities
        possible_indices = np.arange(self.n_nodes)
        moves = np.apply_along_axis(lambda x: np.random.choice(possible_indices, p=x/x.sum()), axis=1, arr=move_probabilities) #ants x 1
        log_probabilites = None
        if gen_probs:
            probabilites = move_probabilities[moves] / move_probabilities.sum(axis=1)
            log_probabilites = np.log(probabilites)

        return moves, log_probabilites

size = 100

nodes = np.random.random(size=(size, 2))

distances = np.sqrt(((nodes[:, None] - nodes[None, :]) ** 2).sum(2))


# x2 = np.sum(nodes**2, axis=1)
# y2 = np.sum(nodes**2, axis=1)
# xy = np.matmul(nodes, nodes.T)
# x2 = x2.reshape(-1, 1)
# distances = np.sqrt(x2 - 2*xy + y2) 


distances[np.arange(size), np.arange(size)] = 1e9

# a = ACO(50, distances)
# iters = 100

# a.run(iters)

# b = ACO(10, distances)
# b.run(iters)

# fig, ax = plt.subplots()
# ax.plot(a.costs, label='a')
# ax.plot(b.costs, label='b')
# plt.legend()
# plt.show()

costs = []
for _ in range(3):
    sim = ACO(20, distances)
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



# size = 20
# a = ACO(50, np.random.random(size=(size, size)))
# iters = 50
# a.run(iters)

# fig, ax = plt.subplots()
# ax.plot(a.costs)

# plt.show()
