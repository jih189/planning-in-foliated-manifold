import networkx as nx
import numpy as np
class LeadPlanner:
    def __init__(self):
        self.graph = nx.Graph()

        self.manifolds_in_foliation = {} # {foliation_name: [manifold1, manifold2, ...]}
        self.transition_maps = {} # {(foliation1_name, foliation2_name): [(manifold1, manifold2), ...]}
        self.similarity_matrix_in_foliations = {} # {foliation_name: similarity_matrix}

    def load_config(self, foliation_config):
        '''
        Load the foliated problem configuration.
        '''
        for foliation in foliation_config.foliation_set:
            self.add_foliation(foliation)
        for intersection in foliation_config.foliated_intersection_set:
            self.add_intersection(intersection)

    def add_foliation(self, foliation):
        '''
        Each manifolds in the foliation is represented by a node in the graph.
        '''

        manifolds = []
        for i in range(len(foliation["co-parameter-set"])):
            self.graph.add_node(foliation["name"] + "_" + str(i), foliation_name=foliation["name"], manifold_index=i)
            manifolds.append(foliation["name"] + "_" + str(i))
        self.manifolds_in_foliation[foliation["name"]] = manifolds
        self.similarity_matrix_in_foliations[foliation["name"]] = foliation["similarity-matrix"]
        
    def add_intersection(self, intersection):
        '''
        Add edges between manifolds from different foliations.
        '''

        first_foliation_name = intersection["foliation1"]["name"]
        second_foliation_name = intersection["foliation2"]["name"]

        manifolds_from_first_foliation = self.manifolds_in_foliation[first_foliation_name]
        manifolds_from_second_foliation = self.manifolds_in_foliation[second_foliation_name]
        
        # if two foliation share the same co-parameter type,
        # then they should have a parallel manifold structure
        if intersection["foliation1"]["co-parameter-type"] == intersection["foliation2"]["co-parameter-type"]:
            # check if they have the same co-parameter set
            if len(intersection["foliation1"]["co-parameter-set"]) != len(intersection["foliation2"]["co-parameter-set"]):
                raise ValueError("Foliations must have the same number of co-parameter sets")
            
            is_same_co_parameter_set = True
            for i in range(len(intersection["foliation1"]["co-parameter-set"])):
                if not np.allclose(intersection["foliation1"]["co-parameter-set"][i], intersection["foliation2"]["co-parameter-set"][i]):
                    raise ValueError("Foliations must have the same co-parameter sets")
            
            self.transition_maps[(first_foliation_name, second_foliation_name)] = []
            # add edge from each manifold from foliation 1 to the corresponding manifold from foliation 2
            for manifold1, manifold2 in zip(manifolds_from_first_foliation, manifolds_from_second_foliation):
                self.graph.add_edge(manifold1, manifold2, weight=0.0)
                self.transition_maps[(first_foliation_name, second_foliation_name)].append((manifold1, manifold2))
        else:
            self.transition_maps[(first_foliation_name, second_foliation_name)] = []
            # add edge from each manifold from foliation 1 to each manifold from foliation 2
            for manifold1 in manifolds_from_first_foliation:
                for manifold2 in manifolds_from_second_foliation:
                    self.graph.add_edge(manifold1, manifold2, weight=0.0)
                    self.transition_maps[(first_foliation_name, second_foliation_name)].append((manifold1, manifold2))

    def get_lead(self, start_foliation, start_manifold_index, goal_foliation, goal_manifold_index):
        '''
        Return the lead graph.
        '''
        return nx.shortest_path(self.graph, source=start_foliation + "_" + str(start_manifold_index), target=goal_foliation + "_" + str(goal_manifold_index))

    def add_penalty(self, foliation_1, manifold_1_index, foliation_2, manifold_2_index, penalty):
        '''
        Add penalty to the edge between two manifolds.
        '''

        manifolds_from_first_foliation = self.manifolds_in_foliation[foliation_1]
        manifolds_from_second_foliation = self.manifolds_in_foliation[foliation_2]

        for i in range(len(manifolds_from_first_foliation)):
            for j in range(len(manifolds_from_second_foliation)):
                self.graph[manifolds_from_first_foliation[i]][manifolds_from_second_foliation[j]]["weight"] += \
                    penalty * self.similarity_matrix_in_foliations[foliation_1][manifold_1_index][i] * self.similarity_matrix_in_foliations[foliation_2][manifold_2_index][j]
