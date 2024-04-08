import numpy as np
import networkx as nx
from foliation_planning.foliated_base_class import (
    BaseTaskPlanner,
    BaseIntersection,
    Task,
)
import copy

class FoliatedRepMapTaskPlanner(BaseTaskPlanner):
    def __init__(self, gmm, planner_name_="FoliatedRepMapTaskPlanner", parameter_dict_={}):
        # Constructor
        super(BaseTaskPlanner, self).__init__()  # python 2
        # super().__init__() # python 3
        self.planner_name = planner_name_
        self.parameter_dict = parameter_dict_

        # need to generate the local foliated repetition roadmap based on the gmm. This 
        # genereated roadmap will be used to clone the foliated repetition roadmap later.
        self.local_foliated_rep_map_template = nx.DiGraph()
        self.gmm_ = gmm
        self.prepare_gmm(gmm)

    def prepare_gmm(self, gmm):
        for i in range(len(gmm.distributions)):
            self.local_foliated_rep_map_template.add_node(i, foliation_name = "", co_parameter_index = -1, weight = 0.0)

        for edge in gmm.edge_of_distribution:
            # this graph is directed, so we need to add two edges
            self.local_foliated_rep_map_template.add_edge(edge[0], edge[1], is_intersection = False, intersection = None, weight = 0.0)
            self.local_foliated_rep_map_template.add_edge(edge[1], edge[0], is_intersection = False, intersection = None, weight = 0.0)

    def generate_local_foliated_rep_map(self, foliation_name, co_parameter_index):
        '''
        Generate a local foliated repetition roadmap based on the foliation name and the co-parameter index.
        '''
        # clone the local foliated repetition roadmap
        local_foliated_rep_map = copy.deepcopy(self.local_foliated_rep_map_template)

        # relabel the nodes
        mapping = {i: (foliation_name, co_parameter_index, i) for i in local_foliated_rep_map.nodes()}
        nx.relabel_nodes(local_foliated_rep_map, mapping, copy=False)
        
        nx.set_node_attributes(local_foliated_rep_map, foliation_name, "foliation_name")
        nx.set_node_attributes(local_foliated_rep_map, co_parameter_index, "co_parameter_index")

        return local_foliated_rep_map

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
        self.explored_manifolds_in_foliation = set()
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

    def update_foliated_rep_map(self, sampled_intersections):
        '''
        Update the foliated repetition roadmap based on the sampled intersections.
        '''
        start_distribution_id_goal_distribution_id_intersection = []
        intersection_edge_configurations = []
        for i in sampled_intersections:
            intersection_edge_configurations.append(i.intersection_motion[0])
            intersection_edge_configurations.append(i.intersection_motion[-1])

        intersection_edge_distribution_ids = self.gmm_.get_distribution_indexs(intersection_edge_configurations)

        for i in range(len(sampled_intersections)):
            intersection_start_foliation_name = sampled_intersections[i].foliation1_name
            intersection_start_co_parameter_index = sampled_intersections[i].co_parameter1_index
            intersection_start_configuration = sampled_intersections[i].intersection_motion[0]
            intersection_start_distribution_id = intersection_edge_distribution_ids[2*i]
            intersection_goal_foliation_name = sampled_intersections[i].foliation2_name
            intersection_goal_co_parameter_index = sampled_intersections[i].co_parameter2_index
            intersection_goal_configuration = sampled_intersections[i].intersection_motion[-1]
            intersection_goal_distribution_id = intersection_edge_distribution_ids[2*i+1]

            # print "intersection between ", intersection_start_foliation_name, intersection_start_co_parameter_index, " and ", intersection_goal_foliation_name, intersection_goal_co_parameter_index
            # print "start configuration ", intersection_start_configuration
            # print "start distribution id ", intersection_start_distribution_id
            # print "goal configuration ", intersection_goal_configuration
            # print "goal distribution id ", intersection_goal_distribution_id

            # if it is one of manifold is new, then construct a local FoliatedRepMap for it.
            if (intersection_start_foliation_name, intersection_start_co_parameter_index) not in self.explored_manifolds_in_foliation:
                self.explored_manifolds_in_foliation.add((intersection_start_foliation_name, intersection_start_co_parameter_index))

                # combine the new graph with the current FoliatedRepMap
                self.FoliatedRepMap = nx.compose(
                    self.FoliatedRepMap, 
                    self.generate_local_foliated_rep_map(intersection_start_foliation_name, intersection_start_co_parameter_index)
                )

            if (intersection_goal_foliation_name, intersection_goal_co_parameter_index) not in self.explored_manifolds_in_foliation:
                self.explored_manifolds_in_foliation.add((intersection_goal_foliation_name, intersection_goal_co_parameter_index))

                # combine the new graph with the current FoliatedRepMap
                self.FoliatedRepMap = nx.compose(
                    self.FoliatedRepMap, 
                    self.generate_local_foliated_rep_map(intersection_goal_foliation_name, intersection_goal_co_parameter_index)
                )

            # add the intersection between two local foliated RepMap.
            self.FoliatedRepMap.add_edge(
                (intersection_start_foliation_name, intersection_start_co_parameter_index, intersection_start_distribution_id),
                (intersection_goal_foliation_name, intersection_goal_co_parameter_index, intersection_goal_distribution_id),
                is_intersection = True,
                intersection = sampled_intersections[i]
            )

    def generate_lead_sequence(self, current_start_configuration, current_foliation_name, current_co_parameter_index):
        '''
        Similar to the MTG task planner, this funciton first generates a lead sequence between the start and the goal
        based on the mode transition graph.
        '''

        sampled_intersections = []
        for step in range(100):
            found_lead = True

            # check if there is a path between the start and the goal
            if not nx.has_path(self.mode_transition_graph, (current_foliation_name, current_co_parameter_index), (self.goal_foliation_name, self.goal_co_parameter_index)):
                print "No path found in the mode transition graph!"
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
                '''
                If a lead sequence is found, we need to update the foliatedRepMap based on the sampled intersections.
                After updating the foliatedRepMap, we can find the start node and the goal node in the foliated repetition roadmap.
                However, if there is no path between the start and the goal in the foliated repetition roadmap, we need to
                run the rerun to generate more intersections for updating the foliated repetition roadmap.
                '''
                self.update_foliated_rep_map(sampled_intersections)

                # find the start node and the goal node in the foliated repetition roadmap
                start_distribution_id, goal_distribution_id = self.gmm_.get_distribution_indexs([self.start_configuration, self.goal_configuration])
                
                start_node = (self.start_foliation_name, self.start_co_parameter_index, start_distribution_id)
                goal_node = (self.goal_foliation_name, self.goal_co_parameter_index, goal_distribution_id)

                if nx.has_path(self.FoliatedRepMap, start_node, goal_node):
                    print "Path found in the foliated repetition roadmap"
                    solution_from_foliated_rep_map = nx.shortest_path(self.FoliatedRepMap, source=start_node, target=goal_node)

                    # generate the guidiance to motion planner.
                    return self.generate_task_sequence(solution_from_foliated_rep_map)
                else:
                    print "Not path found in the foliated repetition roadmap, and need to generate more intersections."

        return []

    def generate_task_sequence(self, path_from_foliated_rep_map):
        
        task_sequence = []

        task_start_configuration = self.start_configuration
        task_node_experience = [] # each element is a tuple (node_id, distribution, list_of_related_nodes) This list of related nodes is used for Atlas.

        for node1, node2 in zip(path_from_foliated_rep_map[:-1], path_from_foliated_rep_map[1:]):

            current_edge = self.FoliatedRepMap.get_edge_data(node1, node2)

            if current_edge["is_intersection"]:
                # current edge is a transition edge from one manifold to another manifold
                task = Task(
                    self.foliations_set[node1[0]].constraint_parameters, # constraint_parameters
                    self.foliations_set[node1[0]].co_parameters[node1[1]], # co_parameters
                    task_node_experience, # related experience
                    [current_edge["intersection"]], # intersection
                    False
                )
                task_sequence.append((task, (node1[0], node1[1], node2[0], node2[1])))
            else:
                # current edge is in the same manifold
                task_node_experience.append(
                    (node2, self.gmm_.distributions[node2[2]], [])
                )

        # add the last task
        task = Task(
            self.foliations_set[path_from_foliated_rep_map[-1][0]].constraint_parameters, # constraint_parameters
            self.foliations_set[path_from_foliated_rep_map[-1][0]].co_parameters[path_from_foliated_rep_map[-1][1]], # co_parameters
            task_node_experience, # related experience
            [], # intersection
            False
        )

        task_sequence.append((task, (path_from_foliated_rep_map[-1][0], path_from_foliated_rep_map[-1][1], None, None)))

        return task_sequence


    def update(self, mode_transition, success_flag, motion_plan_result, experience, manifold_constraint):
        #TODO: Implement this function
        pass