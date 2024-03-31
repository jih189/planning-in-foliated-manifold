import numpy as np
import networkx as nx
from foliation_planning.foliated_base_class import (
    BaseTaskPlanner,
    BaseIntersection,
    Task,
)

class FoliatedRepMapTaskPlanner(BaseTaskPlanner):
    def __init__(self, gmm, planner_name_="FoliatedRepMapTaskPlanner", parameter_dict_={}):
        # Constructor
        super(BaseTaskPlanner, self).__init__()  # python 2
        # super().__init__() # python 3
        self.gmm_ = gmm
        self.planner_name = planner_name_
        self.parameter_dict = parameter_dict_

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

        # reset the foliated Repetition Roadmap
        self.explored_manifolds_in_foliation = {}
        self.FoliatedRepMap = nx.DiGraph()

    def add_manifold(self, foliation_name, co_parameter_index):
        self.mode_transition_graph.add_node((foliation_name, co_parameter_index))
        # if foliation_name not in self.manifolds_in_foliation then add it to the dictionary
        if foliation_name not in self.manifolds_in_foliation:
            self.manifolds_in_foliation[foliation_name] = []
        self.manifolds_in_foliation[foliation_name].append((foliation_name, co_parameter_index))

    def add_foliated_intersection(self, foliation1_name, foliation2_name, intersection_detail):
        transition_pairs = self.intersection_rule.find_connected_co_parameters(self.foliations_set[foliation1_name], self.foliations_set[foliation2_name])

        self.transition_maps[(foliation1_name, foliation2_name)] = []
        for i, j in transition_pairs:
            self.mode_transition_graph.add_edge((foliation1_name, i), (foliation2_name, j), intersection_detail=intersection_detail, weight=0.0)
            self.transition_maps[(foliation1_name, foliation2_name)].append(((foliation1_name, i), (foliation2_name, j)))

    def add_penalty(self, foliation_1, manifold_1_index, foliation_2, manifold_2_index, penalty):
        '''
        Add penalty to the edge between two manifolds.
        '''

        manifolds_from_first_foliation = self.manifolds_in_foliation[foliation_1]
        manifolds_from_second_foliation = self.manifolds_in_foliation[foliation_2]

        for i in range(len(manifolds_from_first_foliation)):
            for j in range(len(manifolds_from_second_foliation)):
                # check if the edge exists
                if self.mode_transition_graph.has_edge(manifolds_from_first_foliation[i], manifolds_from_second_foliation[j]):
                    self.mode_transition_graph[manifolds_from_first_foliation[i]][manifolds_from_second_foliation[j]]["weight"] += \
                        penalty * self.total_similiarity_table[foliation_1][manifold_1_index][i] * self.total_similiarity_table[foliation_2][manifold_2_index][j]

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

    def generate_lead_sequence(self, current_start_configuration, current_foliation_name, current_co_parameter_index):

        found_lead = True
        sampled_intersections = []
        for step in range(100):
            found_lead = True

            # check if there is a path between the start and the goal
            if not nx.has_path(self.mode_transition_graph, (current_foliation_name, current_co_parameter_index), (self.goal_foliation_name, self.goal_co_parameter_index)):
                return []

            # seach for lead sequence
            path = nx.shortest_path(
                self.mode_transition_graph, 
                source=(current_foliation_name, current_co_parameter_index),
                target=(self.goal_foliation_name, self.goal_co_parameter_index),
                weight="weight"
                )

            sampled_intersections = []

            # return the edges of the shortest path
            for i in range(len(path) - 1):

                current_sampled_intersections = self.intersection_sampler.generate_configurations_on_intersection(
                    self.foliations_set[path[i][0]],
                    path[i][1],
                    self.foliations_set[path[i+1][0]],
                    path[i+1][1],
                    self.mode_transition_graph.get_edge_data(path[i], path[i+1])["intersection_detail"]
                )

                if len(current_sampled_intersections) == 0:
                    self.add_penalty(
                        path[i][0],
                        path[i][1],
                        path[i+1][0],
                        path[i+1][1],
                        10.0
                    )
                    found_lead = False
                    break

                # collect the sampled intersections
                sampled_intersections += current_sampled_intersections

                # add penalty to the edge
                self.add_penalty(
                    path[i][0],
                    path[i][1],
                    path[i+1][0],
                    path[i+1][1],
                    0.1
                )

            if found_lead:
                break        
        if not found_lead:
            return []

        print "number of sampled intersections: ", len(sampled_intersections)
        for i in sampled_intersections:
            print "intersection between ", i.foliation1_name, i.co_parameter1_index, i.foliation2_name, i.co_parameter2_index
            print "start configuration ", i.intersection_motion[0]
            print "goal configuration ", i.intersection_motion[-1]

            # if it is one of manifold is new, then construct a local FoliatedRepMap for it.
            if (i.foliation1_name, i.co_parameter1_index) not in self.explored_manifolds_in_foliation:
                self.explored_manifolds_in_foliation[(i.foliation1_name, i.co_parameter1_index)] = nx.DiGraph()

            if (i.foliation2_name, i.co_parameter2_index) not in self.explored_manifolds_in_foliation:
                self.explored_manifolds_in_foliation[(i.foliation2_name, i.co_parameter2_index)] = nx.DiGraph()

            # add the intersection between two local foliated RepMap.

        # search for a solution lead based on the foliated repetition roadmap

        return []

    def update(self, mode_transition, success_flag, motion_plan_result, experience, manifold_constraint):
        #TODO: Implement this function
        pass