import torch

def two_opt(paths, distances, rounds=1):
        n_nodes = distances.size(0)
        if paths.dim() != 2:
            paths = paths.unsqueeze(0) # Now we have #some n x #nodes paths
        num_paths = paths.size()[0]
        for _ in range(rounds):
            # store a matrix, for each ant the best improvement, 1st and 2nd #ants x 3
            optimisation_scores = torch.zeros(size=(num_paths,))
            optimisation_data = torch.zeros(size=(num_paths, 2))
            # For every pair of edges, we check whether swapping them would improve the solution
            for first_index in range(n_nodes-2):
                for second_index in range(first_index+2, n_nodes):
                    first_from = first_index
                    first_to = first_index+1 # Can never wrap around
                    second_from = second_index
                    second_to = (second_index+1)%n_nodes
                    # We have a situation like ... (1st) (1st+1) X ... Z (2nd)   (2nd+1) ... 
                    # We now test the change   ... (1st) (2nd)   Z ... X (1st+1) (2nd+1) ...
                    # So reverse the order of            \_____________________/

                    # Cost change = + 2 new edge costs - 2 old edge costs
                    cost_change = (distances[paths[:, first_from], paths[:, second_from]] 
                                + distances[paths[:, first_to], paths[:, second_to]]
                                - distances[paths[:, first_from], paths[:, first_to]]
                                - distances[paths[:, second_from], paths[:, second_to]])
                    

                    improved_tours = cost_change < optimisation_scores
                    optimisation_scores[improved_tours] = cost_change[improved_tours]
                    optimisation_data[improved_tours, 0] = first_index
                    optimisation_data[improved_tours, 1] = second_index

            improved_tours = optimisation_scores < 0 # The ants whose tours we've been able to improve
            first_froms = optimisation_data[:, 0].long()
            second_froms = optimisation_data[:, 1].long()
            first_tos = first_froms+1
            second_tos = (second_froms+1)%n_nodes

            for ant_index in range(num_paths):
                if improved_tours[ant_index]:
                    # Apply the two opt change for all ants whose tours were improved
                    paths[ant_index, first_tos[ant_index]:second_tos[ant_index]] = paths[ant_index, first_tos[ant_index]:second_tos[ant_index]].flip(0)

        return paths