import numpy as np

HEURISTIC_WEIGHTING = 1
PHEROMONE_WEIGHTING = 1


class Ant:
    def __init__(self, starting_node, use=False) -> None:
        self.visited = [starting_node]
        self.current = starting_node
        self.valid_moves = []
        self.valid_moves_probabilities = []
        self.use = use
    
    def generate_valid_moves(self, nodes):
        self.valid_moves = []
        self.valid_moves_probabilities = []
        for node in nodes:
            if node not in self.visited:
                self.valid_moves.append(node)

                distance_to_other_node = np.linalg.norm(self.current.position - node.position)
                heuristic_value = (1/distance_to_other_node) ** 3

                pheromone_value = self.current.edge_pheromones[id(node)] ** HEURISTIC_WEIGHTING

                self.valid_moves_probabilities.append(pheromone_value * heuristic_value)
    
    def move(self):
        if self.use:
            index = self.valid_moves_probabilities.index(max(self.valid_moves_probabilities))
            node = self.valid_moves[index]
            self.current = node
            self.visited.append(node)
            return

        limit = sum(self.valid_moves_probabilities)
        selected_point = np.random.random() * limit
        for node, prob in zip(self.valid_moves, self.valid_moves_probabilities):
            if selected_point <= prob:
                self.current = node
                self.visited.append(node)
                return
            else:
                selected_point -= prob
        raise ValueError('We seem to have failed to choose a random next vertex')

            
            
