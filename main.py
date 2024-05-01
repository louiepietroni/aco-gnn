import numpy as np
import matplotlib.pyplot as plt

from node import Node
from ant import Ant

# HYPER PARAMETERS
INSTANCE_SIZE = 15

# PHEROMONE_WEIGHTING = 1
# HEURISTIC_WEIGHTING = 1

PHEROMONE_EVAPORATION_CONSTANT = 0.2 # From 0 to 1, proportion of pheromone which evaporates each iteration
TSP_PHEROMONE_AMOUNT = 50

def get_x(node_list):
    return [node.x() for node in node_list]

def get_y(node_list):
    return [node.y() for node in node_list]


instance_points = [Node(np.random.random(size=(2))) for _ in range(INSTANCE_SIZE)]
# instance_points = [Node(np.array([0.31875722, 0.95289267])), Node(np.array([0.06788429, 0.97840346])), Node(np.array([0.97669318, 0.79148225])), Node(np.array([0.98294801, 0.58757233]))]
for point in instance_points:
    point.initialise_pheromones(instance_points)

# for point in instance_points:
#     print(point.position)

plt.figure(figsize=(8, 4))

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal')
plt.ion()
plt.show()
plt.pause(1)


def draw_pheromones():
    lines = []
    for node1 in instance_points:
        for node2 in instance_points:
            if node1 is not node2:
                # l = plt.plot([node1.x(), node2.x()], [node1.y(), node2.y()], color='red', linewidth=0.1 * node1.edge_pheromones[id(node2)])
                l = plt.plot([node1.x(), node2.x()], [node1.y(), node2.y()], color='red', alpha=min(1, node1.edge_pheromones[id(node2)] / 1000))
                # lines.append(plt.plot([node1.x(), node2.x()], [node1.y(), node2.y()], color='red', alpha=min(1, node1.edge_pheromones[id(node2)] / 10000)))
                lines.append(l)
    return lines

def draw_nodes():
    return plt.scatter(get_x(instance_points), get_y(instance_points), s=5)

cn = draw_nodes()
cl = draw_pheromones()
plt.draw()

for x in range(45):
    ants = [Ant(instance_points[0]) for _ in range(100)]

    for _ in range(INSTANCE_SIZE-1):
        for ant in ants:
            c = ant.current
            ant.generate_valid_moves(instance_points)
            ant.move()
            n = ant.current
        # plt.plot([c.x(), n.x()], [c.y(), n.y()])
        # plt.draw()
        # plt.pause(0.01)


    # plt.plot([n.x(), ant.visited[0].x()], [n.y(), ant.visited[0].y()])
    # plt.draw()
    plt.pause(0.001)

    for node in instance_points:
        node.deplete_pheromones(PHEROMONE_EVAPORATION_CONSTANT)
    for ant in ants:
        path = ant.visited
        path_distance = 0
        for index, node in enumerate(path):
            if index == len(path) - 1:
                other = path[0]
            else:
                other = path[index+1]
            distance = np.linalg.norm(node.position - other.position)
            path_distance += distance

        pheromone_to_add = TSP_PHEROMONE_AMOUNT / path_distance
        for index, node in enumerate(path):
            if index == len(path) - 1:
                other = path[0]
            else:
                other = path[index+1]
            node.add_to_pheromones(other, pheromone_to_add)

    for artist in plt.gca().lines:
        artist.remove()
    
    cn = draw_nodes()
    cl = draw_pheromones()

    ant = Ant(instance_points[0], use=True)
    for _ in range(INSTANCE_SIZE-1):
        c = ant.current
        ant.generate_valid_moves(instance_points)
        ant.move()
        n = ant.current
        plt.plot([c.x(), n.x()], [c.y(), n.y()], color='teal', alpha=0.99)

    plt.plot([n.x(), ant.visited[0].x()], [n.y(), ant.visited[0].y()], color='teal', alpha=0.99)
    plt.pause(0.01)

ant = Ant(instance_points[0], use=True)
for _ in range(INSTANCE_SIZE-1):
    c = ant.current
    ant.generate_valid_moves(instance_points)
    ant.move()
    n = ant.current
    plt.plot([c.x(), n.x()], [c.y(), n.y()], color='teal')
    # plt.draw()
    plt.pause(0.01)


plt.plot([n.x(), ant.visited[0].x()], [n.y(), ant.visited[0].y()], color='teal')
plt.draw()
plt.pause(0.1)





