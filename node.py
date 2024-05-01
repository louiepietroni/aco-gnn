class Node:
    def __init__(self, position) -> None:
        self.position = position
        self.edge_pheromones = {}
    
    def initialise_pheromones(self, nodes):
        for node in nodes:
            if node is not self:
                self.edge_pheromones[id(node)] = 1
    
    def deplete_pheromones(self, rate):
        for key in self.edge_pheromones.keys():
            self.edge_pheromones[key] = self.edge_pheromones[key] * (1 - rate)
    
    def add_to_pheromones(self, node, value):
        self.edge_pheromones[id(node)] = self.edge_pheromones[id(node)] + value

    def x(self):
        return self.position[0]
    
    def y(self):
        return self.position[1]
