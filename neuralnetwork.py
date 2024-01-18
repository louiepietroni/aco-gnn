import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

class GNNLayer(nn.Module):
    def __init__(self, dimension, aggregation_function, activation_function='relu'):
        super().__init__()
        self.dimension = dimension
        self.aggregation_function = getattr(gnn, aggregation_function)
        self.activation_function = getattr(F, self.activation_function)
        self.U = nn.Linear(dimension, dimension)
        self.V = nn.Linear(dimension, dimension)
        self.A = nn.Linear(dimension, dimension)
        self.B = nn.Linear(dimension, dimension)
        self.C = nn.Linear(dimension, dimension)

        self.h_batchNorm = gnn.batchNorm(dimension)
        self.e_batchNorm = gnn.batchNorm(dimension)

    def forward(self, h_in, e_in, edge_index):
        h_initial = h_in
        e_initial = e_in

        Uh = self.U(h)
        Vh = self.V(h)
        
        Ae = self.A(e)
        Bh = self.B(h)
        Ch = self.C(h)

        sigmoid_e = torch.sigmoid(e_initial)

        aggregated_h = self.aggregation_function(sigmoid_e * Vh[edge_index[1]], edge_index[0])
        h = h_initial + self.activation_function(self.h_batchNorm(Uh + aggregated_h))

        Bhi = Bh[edge_index[0]] # From vertex features
        Chi = Ch[edge_index[1]] # To vertex features
        e = e_initial + self.activation_function(self.e_batchNorm(Ae + Bhi + Chi))

        return h, e


class GNN(nn.Module):
    def __init__(self, hidden_layer_size, depth, activation_function='relu', aggregation_function='global_mean_pool'):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([GNNLayer(hidden_layer_size, aggregation_function, activation_function)] for _ in range(depth))
        self.h_projection = nn.Linear(2, hidden_layer_size)
        self.e_projection = nn.Linear(1, hidden_layer_size)
        self.activation_function = getattr(F, self.activation_function)

        self.feature_extractor = BasicMLP([hidden_layer_size, hidden_layer_size, hidden_layer_size, 1])
    
    def forward(self, x, edge_attr, edge_index):
        h = self.activation_function(self.h_projection(x))
        e = self.activation_function(self.e_projection(edge_attr))

        for gnnLayer in self.layers:
            h, e = gnnLayer(h, e, edge_index)
        
        output_edge_embeddings = e
        heuristic_vector = self.feature_extractor(output_edge_embeddings)
        return heuristic_vector
    


class BasicMLP(nn.Module):
    def __init__(self, layerSizes, sigmoidLastLayer=True):
        super().__init__()
        self.layerSizes = layerSizes
        self.layers = nn.ModuleList([nn.Linear(nodesIn, nodesOut) for nodesIn, nodesOut in zip(layerSizes[:-1], layerSizes[1:])])
        self.activationFunctions = [torch.silu] * len(self.layers)
        # self.activationFunctions = [getattr(nn, activationFunction)] * len(self.layers)
        self.sigmoidLastLayer = sigmoidLastLayer
        if sigmoidLastLayer:
            self.activationFunctions[-1] = torch.sigmoid

    def forward(self, x):
        for layer, activationFunction in zip(self.layers, self.activationFunctions):
            x = layer(x)
            x = activationFunction(x)
        return x