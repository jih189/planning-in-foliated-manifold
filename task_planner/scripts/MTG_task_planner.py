import numpy as np
import networkx as nx
from foliation_planning.foliated_base_class import (
    BaseTaskPlanner,
    BaseIntersection,
    Task,
)

class MTGTaskPlanner(BaseTaskPlanner):
    def __init__(self, planner_name_="MTGTaskPlanner", parameter_dict_={}):
        # Constructor
        super(BaseTaskPlanner, self).__init__()  # python 2
        # super().__init__() # python 3
        self.planner_name = planner_name_
        self.parameter_dict = parameter_dict_

    # MTGTaskPlanner
    def reset_task_planner(self):
        self.mode_transition_graph = nx.Graph()
        self.manifolds_in_foliation = {} # {foliation_name: [manifold1, manifold2, ...]}
        self.transition_maps = {} # {(foliation1_name, foliation2_name): [(manifold1, manifold2), ...]}
        
        self.start_foliation_name = None
        self.start_co_parameter_index = None
        self.start_configuration = None
        self.goal_foliation_name = None
        self.goal_co_parameter_index = None
        self.goal_configuration = None

    # # MTGTaskPlanner
    def add_manifold(self, foliation_name, co_parameter_index):
        self.mode_transition_graph.add_node((foliation_name, co_parameter_index))
        # if foliation_name not in self.manifolds_in_foliation then add it to the dictionary
        if foliation_name not in self.manifolds_in_foliation:
            self.manifolds_in_foliation[foliation_name] = []
        self.manifolds_in_foliation[foliation_name].append((foliation_name, co_parameter_index))

    # MTGTaskPlanner
    def add_foliated_intersection(self, foliation1_name, foliation2_name, intersection_detail):
        transition_pairs = self.intersection_rule.find_connected_co_parameters(self.foliations_set[foliation1_name], self.foliations_set[foliation2_name])

        self.transition_maps[(foliation1_name, foliation2_name)] = []
        for i, j in transition_pairs:
            self.mode_transition_graph.add_edge((foliation1_name, i), (foliation2_name, j), intersection_detail=intersection_detail)
            self.transition_maps[(foliation1_name, foliation2_name)].append(((foliation1_name, i), (foliation2_name, j)))

    # MTGTaskPlanner
    def set_start_and_goal(
        self,
        start_foliation_name_,
        start_co_parameter_index_,
        start_configuration_,
        goal_foliation_name_,
        goal_co_parameter_index_,
        goal_configuration_,
    ):
        self.start_foliation_name = start_foliation_name_
        self.start_co_parameter_index = start_co_parameter_index_
        self.start_configuration = start_configuration_
        self.goal_foliation_name = goal_foliation_name_
        self.goal_co_parameter_index = goal_co_parameter_index_
        self.goal_configuration = goal_configuration_

    # MTGTaskPlanner
    def generate_lead_sequence(self):
        path = nx.shortest_path(
            self.mode_transition_graph, 
            source=(self.start_foliation_name, self.start_co_parameter_index),
            target=(self.goal_foliation_name, self.goal_co_parameter_index)
            )

    # class Task:
    # def __init__(
    #     self, foliation_name_, co_parameter_index_, start_configuration_, goal_configurations_, next_motion_, use_atlas_
    # ):
    #     # Constructor
    #     self.foliation_name = foliation_name_
    #     self.co_parameter_index = co_parameter_index_
    #     self.goal_configurations = goal_configurations_ # this goal can be a region instead of only one configuration.
    #     self.next_motion = next_motion_  # the robot motion after the task is completed
    #     self.related_experience = []
    #     self.use_atlas = use_atlas_

        result = []
        # return the edges of the shortest path
        for i in range(len(path) - 1):
            sampled_intersections = self.intersection_sampler.generate_configurations_on_intersection(
                self.foliations_set[path[i][0]],
                path[i][1],
                self.foliations_set[path[i+1][0]],
                path[i+1][1],
                self.mode_transition_graph.get_edge_data(path[i], path[i+1])["intersection_detail"]
            )
            
            result.append(Task(
                path[i][0], 
                path[i][1],
                sampled_intersections,
                False
            ))

        return result

    # MTGTaskPlanner
    def update(self, task_graph_info_, plan_, manifold_constraint_):
        # if current task is faled to solve, then we can increate the weight of the edge which is similar to the current task.
        # the similarity is defined as the product of the similarity of the previous manifold, the next manifold, and the current similarity.
        pass