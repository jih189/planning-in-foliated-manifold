import networkx as nx
import numpy as np
class LeadGeneration:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_foliation(self, foliation):
        self.graph.add_node(foliation["name"], foliation=foliation)

    def add_intersection(self, intersection):
        self.graph.add_node(intersection["name"], intersection=intersection)
        if len(intersection["foliations"]) != 2:
            raise ValueError("Intersection must have exactly 2 foliations")

        first_foliation_name = intersection["foliations"][0]["name"]
        second_foliation_name = intersection["foliations"][1]["name"]
        
        # if two foliation share the same co-parameter type,
        # then they should have a parallel manifold structure
        if intersection["foliations"][0]["co-parameter-type"] == intersection["foliations"][1]["co-parameter-type"]:
            # check if they have the same co-parameter set
            if len(intersection["foliations"][0]["co-parameter-set"]) != len(intersection["foliations"][1]["co-parameter-set"]):
                raise ValueError("Foliations must have the same number of co-parameter sets")
            self.graph.add_edge(
                first_foliation_name, second_foliation_name, transition_map=np.zeros((len(intersection["foliations"][0]["co-parameter-set"]), 1))
            )
        else:
            self.graph.add_edge(
                first_foliation_name, second_foliation_name, transition_map=np.zeros((len(intersection["foliations"][0]["co-parameter-set"]), len(intersection["foliations"][1]["co-parameter-set"])))
            )

    def get_lead(self):
        pass

    def add_penalty(self, intersection_name, foliation1_index, foliation2_index, penalty):
        pass