import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

class GNN(nn.Module):
    def __init__(self, hidden_layer_size, depth, activation_function='silu', aggregation_function='global_mean_pool', node_features=2, edge_features=1, edge_gates=True, updated_layers=False):
        super().__init__()
        self.depth = depth
        if updated_layers:
            self.layers = nn.ModuleList([UpdatedGNNLayer(hidden_layer_size, activation_function, aggregation_function, edge_gates) for _ in range(depth)])
        else:
            self.layers = nn.ModuleList([GNNLayer(hidden_layer_size, activation_function, aggregation_function, edge_gates) for _ in range(depth)])
        
        self.h_projection = BasicMLP([node_features, hidden_layer_size], sigmoidLastLayer=False)
        self.e_projection = BasicMLP([edge_features, hidden_layer_size], sigmoidLastLayer=False)
        
        self.activation_function = getattr(F, activation_function)

        self.feature_extractor = BasicMLP([hidden_layer_size, hidden_layer_size, hidden_layer_size, 1])
    
    def forward(self, instance_data):
        x = instance_data.x
        edge_attr = instance_data.edge_attr
        edge_index = instance_data.edge_index

        h = self.h_projection(x)
        e = self.e_projection(edge_attr)

        for gnnLayer in self.layers:
            h, e = gnnLayer(h, e, edge_index)
        
        output_edge_embeddings = e
    
        heuristic_vector = self.feature_extractor(output_edge_embeddings)
        return heuristic_vector

class GNNLayer(nn.Module):
    def __init__(self, dimension, activation_function, aggregation_function, edge_gates):
        super().__init__()
        self.dimension = dimension
        self.edge_gates = edge_gates
        self.aggregation_function = getattr(gnn, aggregation_function)
        self.activation_function = getattr(F, activation_function)
        self.U = nn.Linear(dimension, dimension)
        self.V = nn.Linear(dimension, dimension)
        self.A = nn.Linear(dimension, dimension)
        self.B = nn.Linear(dimension, dimension)
        self.C = nn.Linear(dimension, dimension)

        self.h_batchNorm = gnn.BatchNorm(dimension)
        self.e_batchNorm = gnn.BatchNorm(dimension)

    def forward(self, h_in, e_in, edge_index):
        h_initial = h_in
        e_initial = e_in

        Uh = self.U(h_initial)
        Vh = self.V(h_initial)
        
        Ae = self.A(e_initial)
        Bh = self.B(h_initial)
        Ch = self.C(h_initial)
        
        if self.edge_gates:
            sigmoid_e = torch.sigmoid(e_initial)
        else:
            sigmoid_e = e_initial

        aggregated_h = self.aggregation_function(sigmoid_e * Vh[edge_index[1]], edge_index[0])
        h = h_initial + self.activation_function(self.h_batchNorm(Uh + aggregated_h))

        Bhi = Bh[edge_index[0]] # From vertex features
        Chi = Ch[edge_index[1]] # To vertex features
        e = e_initial + self.activation_function(self.e_batchNorm(Ae + Bhi + Chi))

        return h, e
    
class UpdatedGNNLayer(nn.Module):
    def __init__(self, dimension, activation_function, aggregation_function, edge_gates):
        super().__init__()
        self.dimension = dimension
        self.edge_gates = edge_gates
        self.aggregation_function = getattr(gnn, aggregation_function)
        self.activation_function = getattr(F, activation_function)
        self.U = nn.Linear(dimension, dimension)
        self.V = nn.Linear(dimension, dimension)
        self.A = nn.Linear(dimension, dimension)
        self.B = nn.Linear(dimension, dimension)
        self.C = nn.Linear(dimension, dimension)

        self.h_batchNorm = gnn.BatchNorm(dimension)
        self.e_batchNorm = gnn.BatchNorm(dimension)

        self.message_generator = nn.Linear(dimension*2, dimension)

    def forward(self, h_in, e_in, edge_index):
        h_initial = h_in
        e_initial = e_in

        Uh = self.U(h_initial)
        Vh = self.V(h_initial)
        
        Ae = self.A(e_initial)
        Bh = self.B(h_initial)
        Ch = self.C(h_initial)
        
        if self.edge_gates:
            sigmoid_e = torch.sigmoid(e_initial)
        else:
            sigmoid_e = e_initial

        aggregated_h = self.aggregation_function(sigmoid_e * Vh[edge_index[1]], edge_index[0])
        h = h_initial + self.activation_function(self.h_batchNorm(Uh + aggregated_h))

        Bhi = Bh[edge_index[0]] # From vertex features
        Chi = Ch[edge_index[1]] # To vertex features

        partial_message = self.message_generator(torch.cat((Bhi, Chi), dim=1))
        e = e_initial + self.activation_function(self.e_batchNorm(Ae + partial_message))

        return h, e
    

class BasicMLP(nn.Module):
    def __init__(self, layerSizes, sigmoidLastLayer=True, activation_function='silu'):
        super().__init__()
        self.layerSizes = layerSizes
        self.layers = nn.ModuleList([nn.Linear(nodesIn, nodesOut) for nodesIn, nodesOut in zip(layerSizes[:-1], layerSizes[1:])])
        self.activationFunctions = [getattr(F, activation_function)] * len(self.layers)
        self.sigmoidLastLayer = sigmoidLastLayer
        if sigmoidLastLayer:
            self.activationFunctions[-1] = torch.sigmoid

    def forward(self, x):
        for layer, activationFunction in zip(self.layers, self.activationFunctions):
            x = layer(x)
            x = activationFunction(x)
        return x