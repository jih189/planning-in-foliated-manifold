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
        self.foliation_name_map = {}

        self.success_penalty_for_foliated_rep_map = 0.01
        self.failure_penalty_for_foliated_rep_map = 1.0

        self.mode_transition_graph = None
        self.manifolds_in_foliation = None
        self.transition_maps = None
        self.explored_manifolds_in_foliation = None
        self.explored_intersections_in_foliation = None
        self.FoliatedRepMap = None

    def prepare_gmm(self, gmm):
        for i in range(len(gmm.distributions)):
            # valid count and invalid count are for valid configurations and invalid configurations due to constraints. While, invalid_count_for_robot_env is for invalid configurations in the robot environment.
            self.local_foliated_rep_map_template.add_node(i, foliation_name = "", co_parameter_index = -1, weight = 0.0, valid_count = 0, invalid_count = 0, invalid_count_for_robot_env = 0)

        for edge in gmm.edge_of_distribution:
            # this graph is directed, so we need to add two edges
            self.local_foliated_rep_map_template.add_edge(edge[0], edge[1], is_intersection = False, intersection = None, weight = 0.0)
            self.local_foliated_rep_map_template.add_edge(edge[1], edge[0], is_intersection = False, intersection = None, weight = 0.0)

    def generate_local_foliated_rep_map(self, foliation_name, co_parameter_index):
        '''
        Generate a local foliated repetition roadmap based on the foliation name and the co-parameter index.
        '''
        similar_manifolds = []
        # find the explored manifolds in the same foliation
        for manifold in self.explored_manifolds_in_foliation:
            if manifold[0] == foliation_name:
                similar_manifolds.append(manifold)

        # print "create new manifold ", foliation_name, co_parameter_index
        # print "with similar manifolds ", similar_manifolds

        # clone the local foliated repetition roadmap
        local_foliated_rep_map = copy.deepcopy(self.local_foliated_rep_map_template)

        # relabel the nodes
        mapping = {i: (foliation_name, co_parameter_index, i) for i in local_foliated_rep_map.nodes()}
        nx.relabel_nodes(local_foliated_rep_map, mapping, copy=False)
        
        nx.set_node_attributes(local_foliated_rep_map, foliation_name, "foliation_name")
        nx.set_node_attributes(local_foliated_rep_map, co_parameter_index, "co_parameter_index")

        # update the node's value based on the similar manifolds
        for _, explored_co_parameter_index in similar_manifolds:

            for f, c, distribution_id in local_foliated_rep_map.nodes():

                local_foliated_rep_map.nodes[(f, c, distribution_id)]["weight"] += \
                (
                    self.total_similiarity_table[foliation_name][co_parameter_index, explored_co_parameter_index] * 
                    (
                        0.1 * self.FoliatedRepMap.nodes[(foliation_name, explored_co_parameter_index, distribution_id)]["valid_count"] + 
                        1.0 * self.FoliatedRepMap.nodes[(foliation_name, explored_co_parameter_index, distribution_id)]["invalid_count"]
                    ) + 1.0 * self.FoliatedRepMap.nodes[(foliation_name, explored_co_parameter_index, distribution_id)]["invalid_count_for_robot_env"]
                )

        # update the edge's value by summing the weights of two nodes
        for u, v in local_foliated_rep_map.edges():
            local_foliated_rep_map[u][v]["weight"] = local_foliated_rep_map.nodes[u]["weight"] + local_foliated_rep_map.nodes[v]["weight"]

        return local_foliated_rep_map

    def reset_task_planner(self):

        # need to clear memory.
        if self.mode_transition_graph is not None:
            self.mode_transition_graph.clear()
            del self.mode_transition_graph
        if self.manifolds_in_foliation is not None:
            self.manifolds_in_foliation.clear()
            del self.manifolds_in_foliation
        if self.transition_maps is not None:
            self.transition_maps.clear()
            del self.transition_maps
        if self.explored_manifolds_in_foliation is not None:
            self.explored_manifolds_in_foliation.clear()
            del self.explored_manifolds_in_foliation
        if self.explored_intersections_in_foliation is not None:
            self.explored_intersections_in_foliation.clear()
            del self.explored_intersections_in_foliation
        if self.FoliatedRepMap is not None:
            self.FoliatedRepMap.clear()
            del self.FoliatedRepMap

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
        self.explored_intersections_in_foliation = set()
        self.FoliatedRepMap = nx.DiGraph()

    def add_manifold(self, foliation_name, co_parameter_index):
        self.mode_transition_graph.add_node((foliation_name, co_parameter_index))
        # if foliation_name not in self.manifolds_in_foliation then add it to the dictionary
        if foliation_name not in self.manifolds_in_foliation:
            self.manifolds_in_foliation[foliation_name] = []
        self.manifolds_in_foliation[foliation_name].append((foliation_name, co_parameter_index))

        if foliation_name not in self.foliation_name_map:
            self.foliation_name_map[foliation_name] = len(self.foliation_name_map)

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

        if len(sampled_intersections) == 0:
            return

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
                # combine the new graph with the current FoliatedRepMap
                self.FoliatedRepMap = nx.compose(
                    self.FoliatedRepMap, 
                    self.generate_local_foliated_rep_map(intersection_start_foliation_name, intersection_start_co_parameter_index)
                )

                self.explored_manifolds_in_foliation.add((intersection_start_foliation_name, intersection_start_co_parameter_index))

                # update the edge weight
                self.update_edge_weight(intersection_start_foliation_name)

            if (intersection_goal_foliation_name, intersection_goal_co_parameter_index) not in self.explored_manifolds_in_foliation:
                # combine the new graph with the current FoliatedRepMap
                self.FoliatedRepMap = nx.compose(
                    self.FoliatedRepMap, 
                    self.generate_local_foliated_rep_map(intersection_goal_foliation_name, intersection_goal_co_parameter_index)
                )

                self.explored_manifolds_in_foliation.add((intersection_goal_foliation_name, intersection_goal_co_parameter_index))

                self.update_edge_weight(intersection_goal_foliation_name)

            # add the intersection between two local foliated RepMap.
            self.FoliatedRepMap.add_edge(
                (intersection_start_foliation_name, intersection_start_co_parameter_index, intersection_start_distribution_id),
                (intersection_goal_foliation_name, intersection_goal_co_parameter_index, intersection_goal_distribution_id),
                is_intersection = True,
                intersection = sampled_intersections[i],
                weight = 0.1
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
                return [], None

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

                if (self.foliations_set[path[i][0]], path[i][1],
                    self.foliations_set[path[i+1][0]], path[i+1][1]) not in self.explored_intersections_in_foliation:

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

                    # mark intersection as explored only if there are intersections.
                    self.explored_intersections_in_foliation.add(
                        (self.foliations_set[path[i][0]], path[i][1],
                        self.foliations_set[path[i+1][0]], path[i+1][1])
                    )

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
                start_distribution_id, goal_distribution_id = self.gmm_.get_distribution_indexs([current_start_configuration, self.goal_configuration])
                
                start_node = (current_foliation_name, current_co_parameter_index, start_distribution_id)
                goal_node = (self.goal_foliation_name, self.goal_co_parameter_index, goal_distribution_id)

                if nx.has_path(self.FoliatedRepMap, start_node, goal_node):
                    # print "Path found in the foliated repetition roadmap"

                    return self.generate_task_with_multi_goals(start_node, goal_node), (start_node[0] == goal_node[0] and start_node[1] == goal_node[1])

                    # generate the guidiance to motion planner.
                    # return self.generate_task_sequence(start_node, goal_node), (start_node[0] == goal_node[0] and start_node[1] == goal_node[1])
                else:
                    print "start node: ", start_node, "goal node: ", goal_node
                    print "Not path found in the foliated repetition roadmap, and need to generate more intersections."

        return [], None

    def generate_task_with_multi_goals(self, start_node, goal_node):
        # print "start node: ", start_node
        # print "goal node: ", goal_node

        # if start node and goal node are in the same manifold
        if start_node[0] == goal_node[0] and start_node[1] == goal_node[1]:
            # you should have only one goal in this case
            nodes_to_goal = nx.shortest_path(self.FoliatedRepMap, source=start_node, target=goal_node, weight="weight")

            task_node_experience = []
            for exp_n in nodes_to_goal:
                task_node_experience.append(
                    ((self.foliation_name_map[exp_n[0]], exp_n[1], exp_n[2]), self.gmm_.distributions[exp_n[2]], [])
                )

            task = Task(
                self.foliations_set[start_node[0]].constraint_parameters, # constraint_parameters
                self.foliations_set[start_node[0]].co_parameters[start_node[1]], # co_parameters
                task_node_experience, # related experience
                self.intersection_sampler.generate_final_configuration(self.foliations_set[goal_node[0]], goal_node[1], self.goal_configuration), # intersection
                False
            )

            return [task]
        else:
            # compute the distance for all nodes to the goal node
            distance_to_goal_map = dict(nx.single_source_dijkstra_path_length(self.FoliatedRepMap.reverse(), source=goal_node, weight="weight"))

            next_intersections = []      
            related_nodes = set()

            # find all the intersection node in the same manifold of start node.
            for u, v, is_intersection in self.FoliatedRepMap.edges.data('is_intersection'):
                if is_intersection:

                    # find all intersection leaving from the manifold where start node is in.
                    if u[0] == start_node[0] and u[1] == start_node[1]:

                        # check if reachable
                        if u not in distance_to_goal_map or v not in distance_to_goal_map:
                            continue
                        
                        # only consider the intersection node leading to closer to the goal node.
                        if distance_to_goal_map[u] > distance_to_goal_map[v]:
                            next_intersections.append(self.FoliatedRepMap.edges[u, v]["intersection"])
                            nodes_to_next = nx.shortest_path(self.FoliatedRepMap, source=start_node, target=u, weight="weight")
                            for exp_n in nodes_to_next:
                                related_nodes.add(exp_n)

            task_node_experience = []
            for exp_n in list(related_nodes):
                task_node_experience.append(
                    ((self.foliation_name_map[exp_n[0]], exp_n[1], exp_n[2]), self.gmm_.distributions[exp_n[2]], [])
                )

            task = Task(
                self.foliations_set[start_node[0]].constraint_parameters, # constraint_parameters
                self.foliations_set[start_node[0]].co_parameters[start_node[1]], # co_parameters
                task_node_experience, # related experience
                next_intersections, # intersection
                False
            )

            # print "current mode transition ", 

            return [task]

    # def generate_task_sequence(self, start_node, goal_node):
    #     '''
    #     Generate the task sequence from start node to goal node based on the folaited repetation roadmap.
    #     '''

    #     path_from_foliated_rep_map = nx.shortest_path(self.FoliatedRepMap, source=start_node, target=goal_node)
        
    #     task_sequence = []

    #     task_start_configuration = self.start_configuration
    #     task_node_experience = [] # each element is a tuple (node_id, distribution, list_of_related_nodes) This list of related nodes is used for Atlas.

    #     for node1, node2 in zip(path_from_foliated_rep_map[:-1], path_from_foliated_rep_map[1:]):

    #         current_edge = self.FoliatedRepMap.get_edge_data(node1, node2)

    #         task_node_experience.append(
    #             ((self.foliation_name_map[node1[0]], node1[1], node1[2]), self.gmm_.distributions[node1[2]], [])
    #         )

    #         if current_edge["is_intersection"]:
                
    #             # current edge is a transition edge from one manifold to another manifold
    #             task = Task(
    #                 self.foliations_set[node1[0]].constraint_parameters, # constraint_parameters
    #                 self.foliations_set[node1[0]].co_parameters[node1[1]], # co_parameters
    #                 task_node_experience, # related experience
    #                 [current_edge["intersection"]], # intersection
    #                 False
    #             )
    #             task_node_experience = []
    #             task_sequence.append((task, (node1[0], node1[1], node2[0], node2[1])))

    #     task_node_experience.append(
    #         ((self.foliation_name_map[path_from_foliated_rep_map[-1][0]], path_from_foliated_rep_map[-1][1], path_from_foliated_rep_map[-1][2]), 
    #         self.gmm_.distributions[path_from_foliated_rep_map[-1][2]], 
    #         [])
    #     )

    #     # add the last task
    #     task = Task(
    #         self.foliations_set[path_from_foliated_rep_map[-1][0]].constraint_parameters, # constraint_parameters
    #         self.foliations_set[path_from_foliated_rep_map[-1][0]].co_parameters[path_from_foliated_rep_map[-1][1]], # co_parameters
    #         task_node_experience, # related experience
    #         self.intersection_sampler.generate_final_configuration(self.foliations_set[path_from_foliated_rep_map[-1][0]], path_from_foliated_rep_map[-1][1], self.goal_configuration), # goal configuration
    #         False
    #     )

    #     task_sequence.append((task, (path_from_foliated_rep_map[-1][0], path_from_foliated_rep_map[-1][1],None, None)))

    #     return task_sequence

    def generate_sampled_distribution_tag_table(self, plan):
        # if sampled data is empty, then skip it.
        if len(plan[4].verified_motions) == 0:
            print("sampled data is empty.")
            return None

        sampled_data_numpy = np.array(
            [sampled_data.sampled_state for sampled_data in plan[4].verified_motions]
        )

        # if sampled_data_numpy is empty, then skip it.
        sampled_data_distribution_id = self.gmm_._sklearn_gmm.predict(
            sampled_data_numpy
        ).tolist()

        # the task graph info here is the manifold id(foliatino id and co-parameter id) of the current task.

        # initialize a table with number of distributions in GMM times 4.
        # each row is a distribution in GMM, and each column is a tag of sampled data.
        # the value in the table is the number of sampled data with the same distribution id and tag.
        # tag in column 0: collision free
        # tag in column 1: arm-env collision or out of joint limit
        # tag in column 2: path constraint violation
        # tag in column 3: obj-env collision
        sampled_data_distribution_tag_table = np.zeros(
            (len(self.gmm_.distributions), 4)
        )

        # count the number of sampled data with the same distribution id and tag.
        for i in range(len(sampled_data_distribution_id)):
            sampled_data_gmm_id = sampled_data_distribution_id[i]
            sampled_data_tag = plan[4].verified_motions[i].sampled_state_tag

            if sampled_data_tag == 0 or sampled_data_tag == 5:
                sampled_data_distribution_tag_table[sampled_data_gmm_id][0] += 1
            elif sampled_data_tag == 1 or sampled_data_tag == 6:
                sampled_data_distribution_tag_table[sampled_data_gmm_id][1] += 1
            elif sampled_data_tag == 2 or sampled_data_tag == 7:
                sampled_data_distribution_tag_table[sampled_data_gmm_id][2] += 1
            elif sampled_data_tag == 4 or sampled_data_tag == 9:
                sampled_data_distribution_tag_table[sampled_data_gmm_id][3] += 1
        return sampled_data_distribution_tag_table

    def update_edge_weight(self, current_foliation_id):
        '''
        Update the roadmap's edge in the current folaition id. You may want to do this parallelly.
        '''

        # loop over all the edges in the roadmap
        for u, v, edge_attr in self.FoliatedRepMap.edges(data=True):
            # if this edge is an intersection edge, then skip it.
            if edge_attr["is_intersection"]:
                continue
            
            # if the edge is not in the current foliation, then skip it.
            if u[0] != current_foliation_id or v[0] != current_foliation_id:
                continue

            # update the edge weight by summing the weight of the nodes.
            # edge_attr["weight"] = self.FoliatedRepMap.nodes[u]["weight"] + self.FoliatedRepMap.nodes[v]["weight"] + 1 # add a small value to avoid zero weight.
            self.FoliatedRepMap.edges[u, v]["weight"] = self.FoliatedRepMap.nodes[u]["weight"] + self.FoliatedRepMap.nodes[v]["weight"] + 0.01

    def update_foliated_rep_map_weight(self, n, current_manifold_id, sampled_data_distribution_tag_table):
        """
        Update the weight of a node in the task graph.
        Args:
            n: the node in the task graph.
            current_manifold_id: the manifold id of the current task. (foliation id and co-parameter index)
            sampled_data_distribution_tag_table: a table with shape (number of distributions in GMM, 4).
        Returns:
            None
        """

        if n == "start" or n == "goal":
            return

        # if not in the same foliation, then continue
        if n[0] != current_manifold_id[0]:
            return

        node_foliation_id = n[0]
        node_co_parameter_id = n[1]
        node_gmm_id = n[2]

        current_similarity_score = self.total_similiarity_table[node_foliation_id][
            node_co_parameter_id, current_manifold_id[1]
        ]

        success_score = (
            current_similarity_score * sampled_data_distribution_tag_table[node_gmm_id][0] * self.success_penalty_for_foliated_rep_map
        )

        arm_env_collision_score = (
            sampled_data_distribution_tag_table[node_gmm_id][1] * self.failure_penalty_for_foliated_rep_map
        ) # no current similarity score here due to it is the arm env collision.

        path_constraint_violation_score = (
            current_similarity_score
            * sampled_data_distribution_tag_table[node_gmm_id][2]
            * self.failure_penalty_for_foliated_rep_map
        )

        obj_env_collision_score = (
            current_similarity_score
            * sampled_data_distribution_tag_table[node_gmm_id][3]
            * self.failure_penalty_for_foliated_rep_map
        )

        weight_value = (
            success_score
            + arm_env_collision_score
            + path_constraint_violation_score
            + obj_env_collision_score
        )

        self.FoliatedRepMap.nodes[n]["weight"] += weight_value

    def update_valid_invalid_counts(self, manifold_id, sampled_data_distribution_tag_table):

        foliation_id = manifold_id[0]
        co_parameter_index = manifold_id[1]

        for i in range(len(sampled_data_distribution_tag_table)):
            self.FoliatedRepMap.nodes[(foliation_id, co_parameter_index, i)]["valid_count"] += sampled_data_distribution_tag_table[i][0]
            self.FoliatedRepMap.nodes[(foliation_id, co_parameter_index, i)]["invalid_count"] += (
                sampled_data_distribution_tag_table[i][2]
                + sampled_data_distribution_tag_table[i][3]
            )
            self.FoliatedRepMap.nodes[(foliation_id, co_parameter_index, i)]["invalid_count_for_robot_env"] += sampled_data_distribution_tag_table[i][1]


    def update(self, mode_transitions, success_flag, experience, manifold_constraint):

        # print "update motion transition ", mode_transition
        # print "current manifold id ", current_manifold_id
        # print "success flag ", success_flag
        # print "experience from motion planning"
        # print "experience size = ", len(experience[4].verified_motions)

        if len(mode_transitions) == 0:
            return

        # print "update mode transitions: ", mode_transitions

        for mode_transition in mode_transitions:

            current_manifold_id = (mode_transition[0], mode_transition[1])

            if mode_transition[2] is None and mode_transition[3] is None:
                # this is the last task, then not need to update.
                continue

            if mode_transition[0] == mode_transition[2] and mode_transition[1] == mode_transition[3]:
                # this is the same manifold, then not need to update.
                continue

            if success_flag:
                self.add_penalty(
                    mode_transition[0],
                    mode_transition[1],
                    mode_transition[2],
                    mode_transition[3],
                    0.1
                )
            else:
                self.add_penalty(
                    mode_transition[0],
                    mode_transition[1],
                    mode_transition[2],
                    mode_transition[3],
                    10.0
                )


        sampled_data_distribution_tag_table = self.generate_sampled_distribution_tag_table(experience)

        if sampled_data_distribution_tag_table is None:
            return

        # update the valid and invalid counts
        self.update_valid_invalid_counts(current_manifold_id, sampled_data_distribution_tag_table)

        # only update the weight of nodes in the same manifold with the current task.
        for n in self.FoliatedRepMap.nodes():
            self.update_foliated_rep_map_weight(n, current_manifold_id, sampled_data_distribution_tag_table)

        # update the edge weight
        self.update_edge_weight(current_manifold_id[0])