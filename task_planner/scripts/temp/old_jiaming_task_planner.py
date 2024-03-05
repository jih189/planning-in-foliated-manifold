import numpy as np
import networkx as nx
from sklearn import mixture

import time

import matplotlib.pyplot as plt


class ManifoldDetail:
    """
    ManifoldDetail contains the detail of a manifold. A manifold is defined by a foliation and a co-parameter.
    """

    def __init__(self, foliation, co_parameter_index):
        # Constructor
        self.foliation = foliation
        self.co_parameter_index = co_parameter_index


class IntersectionDetail:
    """
    IntersectionDetail contains the detail of an intersection. ALl information is stored in a dictionary.
    """

    def __init__(
        self,
        intersection_data,
        configuration_in_manifold1,
        configuration_in_manifold2,
        is_goal=False,
    ):
        # Constructor
        self.intersection_data = intersection_data
        self.configuration_in_manifold1 = configuration_in_manifold1
        self.configuration_in_manifold2 = configuration_in_manifold2
        self.is_goal = is_goal


class Task:
    def __init__(
        self, manifold_detail_, start_configuration_, goal_configuration_, next_motion_
    ):
        # Constructor
        self.manifold_detail = manifold_detail_
        self.start_configuration = start_configuration_
        self.goal_configuration = goal_configuration_
        self.next_motion = next_motion_  # the robot motion after the task is completed
        self.distributions = []
        self.has_solution = False

    def set_solution(self, solution_trajectory_):
        self.solution_trajectory = solution_trajectory_
        self.has_solution = True

    def add_gaussian_distribution(self, distribution_):
        self.distributions.append(distribution_)

    def clear_distributions(self):
        self.distributions = []

    def set_task_graph_info(self, task_graph_info_):
        self.task_graph_info = task_graph_info_

    def print_task_detail(self):
        print("----------- task:")
        print("start configuration:")
        print(self.start_configuration)
        print("goal configuration:")
        print(self.goal_configuration)
        print("manifold detail:")
        self.manifold_detail.print_manifold_detail()
        print("task graph info:")
        print(self.task_graph_info)
        print("-----------")


class GaussianDistribution:
    def __init__(
        self,
        mean_,
        covariance_,
    ):
        # Constructor
        self.mean = mean_
        self.covariance = covariance_


class GMM:
    def __init__(self):
        # Constructor
        self.distributions = []
        self.edge_of_distribution = []
        self.edge_probabilities = []
        self._sklearn_gmm = None

        self.collision_free_rates = []

    def get_distribution_index(self, configuration_):
        # find which distribution the configuration belongs to
        # then return the distribution
        # configuration_ is a (d,) element array : (d = 7)
        dist_num = self._sklearn_gmm.predict(configuration_.reshape((1, -1))).squeeze()
        return dist_num.item()

    def get_distribution_indexs(self, configurations_):
        # find which distribution the configuration belongs to
        # then return the distribution
        # configuration_ is a (d,) element array : (d = 7)
        dist_nums = self._sklearn_gmm.predict(configurations_)
        return dist_nums

    def get_distribution(self, configuration_):
        # find which distribution the configuration belongs to
        # then return the distribution
        # configuration_ is a (d,) element array : (d = 7)
        dist_num = self._sklearn_gmm.predict(configuration_.reshape((1, -1))).squeeze()
        # return GaussianDistribution(self._sklearn_gmm.means_[dist_num], self._sklearn_gmm.covariances_[dist_num])
        return self.distributions[dist_num]

    def load_distributions(self, dir_name="../gmm/"):
        means = np.load(dir_name + "means.npy")
        covariances = np.load(dir_name + "covariances.npy")

        # Create an sklearn Gaussian Mixture Model
        self._sklearn_gmm = mixture.GaussianMixture(
            n_components=len(means), covariance_type="full"
        )
        self._sklearn_gmm.precisions_cholesky_ = np.linalg.cholesky(
            np.linalg.inv(covariances)
        )
        self._sklearn_gmm.weights_ = np.load(
            dir_name + "weights.npy"
        )  # how common this distribution is.
        self._sklearn_gmm.means_ = means
        self._sklearn_gmm.covariances_ = covariances

        for mean, covariance in zip(means, covariances):
            self.distributions.append(GaussianDistribution(mean, covariance))
            self.collision_free_rates.append(0.5)
        print("Loaded %d distributions " % len(means), dir_name)
        self.edge_of_distribution = np.load(dir_name + "edges.npy")
        self.edge_probabilities = np.load(dir_name + "edge_probabilities.npy")

    def update_collision_free_rates(self, pointcloud_):
        """
        update the collision-free rate of each distribution.
        """
        pass


class BaseTaskPlanner(object):
    def __init__(self):
        # Constructor
        raise NotImplementedError("Please Implement this method")

    def load_foliated_problem(self, folaited_problem):
        """
        load the foliated problem into the task planner
        """

        # add manifolds
        for foliation_index, foliation in enumerate(folaited_problem.foliations):
            for co_parameter_index, co_parameter in enumerate(foliation.co_parameters):
                self.add_manifold(
                    ManifoldDetail(foliation, co_parameter_index),
                    (foliation_index, co_parameter_index),
                )

            # set similarity matrix for each foliation
            self.set_similarity_matrix(foliation_index, foliation.similarity_matrix)

        # add intersections
        for intersection in folaited_problem.intersections:
            (
                foliation1_name,
                co_parameter1_index,
                foliation2_name,
                co_parameter2_index,
            ) = intersection.get_foliation_names_and_co_parameter_indexes()
            (
                configuration_in_manifold1,
                configuration_in_manifold2,
            ) = intersection.get_edge_configurations()
            # get index of each foliation in the foliation list
            foliation1_index = folaited_problem.get_foliation_index(foliation1_name)
            foliation2_index = folaited_problem.get_foliation_index(foliation2_name)
            self.add_intersection(
                (foliation1_index, co_parameter1_index),
                (foliation2_index, co_parameter2_index),
                IntersectionDetail(
                    intersection,
                    configuration_in_manifold1,
                    configuration_in_manifold2,
                    False,
                ),
            )

    # reset task planner
    def reset_task_planner(self):
        raise NotImplementedError("Please Implement this method")

    def read_pointcloud(self, pointcloud_):
        print(
            "-- the task planner does not support read point point because it does not use GMM --"
        )
        # raise NotImplementedError("Please Implement this method")

    def add_manifold(self, manifold_info_, manifold_id_):
        raise NotImplementedError("Please Implement this method")

    def add_intersection(self, manifold_id1_, manifold_id2_, intersection_detail_):
        """
        add intersection to the manifold
        """
        raise NotImplementedError("Please Implement this method")

    def set_start_and_goal(
        self,
        start_manifold_id_,
        start_intersection_,
        goal_manifold_id_,
        goal_intersection_,
    ):
        """
        set start and goal configurations
        both start and goal configurations are intersection here.
        """
        raise NotImplementedError("Please Implement this method")

    def generate_task_sequence(self):
        """
        generate task sequence
        """
        raise NotImplementedError("Please Implement this method")

    def update(self, task_graph_info_, plan_, solution_):
        """
        update task planner
        """
        raise NotImplementedError("Please Implement this method")

    def set_similarity_matrix(self, foliation_id_, similarity_matrix_):
        """
        set similarity matrix for a foliation
        """
        self.total_similiarity_table[foliation_id_] = similarity_matrix_

    def reset_task_solution_graph(self):
        self.task_solution_graph = nx.DiGraph()
        self.incomming_manifold_intersections = (
            {}
        )  # the incomming intersections of each manifold
        self.outgoing_manifold_intersections = (
            {}
        )  # the outgoing intersections of each manifold
        self.new_intersection_id = 0

    def add_manifold_for_task_solution_graph(self, manifold_id_):
        self.incomming_manifold_intersections[manifold_id_] = []
        self.outgoing_manifold_intersections[manifold_id_] = []

    def add_intersection_for_task_solution_graph(self, manifold_id1_, manifold_id2_):
        intersection_from_1_to_2_id = self.new_intersection_id
        self.new_intersection_id += 1

        self.task_solution_graph.add_node(
            intersection_from_1_to_2_id,
            previous_manifold_id=manifold_id1_,
            next_manifold_id=manifold_id2_,
        )

        for i in self.incomming_manifold_intersections[manifold_id1_]:
            self.task_solution_graph.add_edge(
                i,
                intersection_from_1_to_2_id,
                manifold_id=manifold_id1_,
                has_solution=False,
                solution_trajectory=None,
            )

        for i in self.outgoing_manifold_intersections[manifold_id2_]:
            self.task_solution_graph.add_edge(
                intersection_from_1_to_2_id,
                i,
                manifold_id=manifold_id2_,
                has_solution=False,
                solution_trajectory=None,
            )

        self.outgoing_manifold_intersections[manifold_id1_].append(
            intersection_from_1_to_2_id
        )
        self.incomming_manifold_intersections[manifold_id2_].append(
            intersection_from_1_to_2_id
        )

        return intersection_from_1_to_2_id

    def set_start_and_goal_for_task_solution_graph(
        self, start_manifold_id_, goal_manifold_id_
    ):
        if self.task_solution_graph.has_node("start"):
            self.task_solution_graph.remove_node("start")

        if self.task_solution_graph.has_node("goal"):
            self.task_solution_graph.remove_node("goal")

        self.task_solution_graph.add_node(
            "start", previous_manifold_id=None, next_manifold_id=start_manifold_id_
        )

        self.task_solution_graph.add_node(
            "goal", previous_manifold_id=goal_manifold_id_, next_manifold_id=None
        )

        for i in self.outgoing_manifold_intersections[start_manifold_id_]:
            self.task_solution_graph.add_edge(
                "start",
                i,
                manifold_id=start_manifold_id_,
                has_solution=False,
                solution_trajectory=None,
            )

        for i in self.incomming_manifold_intersections[goal_manifold_id_]:
            self.task_solution_graph.add_edge(
                i,
                "goal",
                manifold_id=goal_manifold_id_,
                has_solution=False,
                solution_trajectory=None,
            )

    def check_solution_existence(self, intersection_id1_, intersection_id2_):
        """
        check if the solution exists between two intersections. If the solution exists,
        then return the solution trajectory. Otherwise, return None.
        """
        if self.task_solution_graph.edges[intersection_id1_, intersection_id2_][
            "has_solution"
        ]:
            return self.task_solution_graph.edges[intersection_id1_, intersection_id2_][
                "solution_trajectory"
            ]
        else:
            return None

    def save_solution_to_task_solution_graph(
        self, intersection_id1_, intersection_id2_, solution_trajectory_
    ):
        self.task_solution_graph.edges[intersection_id1_, intersection_id2_][
            "has_solution"
        ] = True
        self.task_solution_graph.edges[intersection_id1_, intersection_id2_][
            "solution_trajectory"
        ] = solution_trajectory_

    def get_manifold_id_from_task_solution_graph(
        self, intersection_id1_, intersection_id2_
    ):
        return self.task_solution_graph.edges[intersection_id1_, intersection_id2_][
            "manifold_id"
        ]

    def value_iteration(self):
        """
        Traditional value iteration.
        """

        # if planner name is not start with "MDP", then we do not use value iteration.
        if self.planner_name[:3] != "MDP":
            # throw an error if the planner name is not start with "MDP"
            raise NotImplementedError(
                "Call value iteration if only the planner is MDP planner."
            )

        for t in range(self.value_iteration_iters):
            # start_time = time.time()

            new_value_function = {}

            self.value_function["goal"] = self.reward_of_goal

            # update the value function for each node
            for node in self.task_graph.nodes:
                # if the node has no outgoing edges, then set the value function to 0.0
                if self.task_graph.out_degree(node) == 0:
                    new_value_function[node] = -1.0
                    continue

                # Update the value function of node with max of neighbors
                new_value_function[node] = (
                    max(
                        p * (1.0 + self.value_function[v])
                        for _, v, p in self.task_graph.out_edges(
                            node, data="probability"
                        )
                    )
                    - 1.0
                ) * self.gamma

            new_value_function["goal"] = self.reward_of_goal

            # if the value function converges, then stop
            if np.all(
                np.isclose(
                    list(self.value_function.values()),
                    list(new_value_function.values()),
                    rtol=self.epsilon,
                    atol=self.epsilon,
                )
            ):
                break

            # print "value iteration: ", t, " time: ", time.time() - start_time, " diff ", np.max(np.abs(np.array(list(self.value_function.values())) - np.array(list(new_value_function.values()))))
            self.value_function = new_value_function

    def plan_on_value_function(self):
        """
        Traditional value iteration. This function is called after value iteration.
        """
        if self.planner_name[:3] != "MDP":
            # throw an error if the planner name is not start with "MDP"
            raise NotImplementedError(
                "Call this function if only the planner is MDP planner."
            )

        result = []
        current_node = "start"
        while current_node != "goal":
            # print current_node, self.value_function[current_node]
            if current_node in result:
                print("loop detected!")
                return []

            result.append(current_node)

            current_node = max(
                self.task_graph.neighbors(current_node),
                key=lambda x: self.task_graph.edges[(current_node, x)]["probability"]
                * (self.value_function[x] + 1),
            )
        result.append("goal")

        return result


class MTGTaskPlanner(BaseTaskPlanner):
    def __init__(self, planner_name_="MTGTaskPlanner", parameter_dict_={}):
        # Constructor
        super(BaseTaskPlanner, self).__init__()  # python 2
        # super().__init__() # python 3
        self.planner_name = planner_name_
        self.parameter_dict = parameter_dict_

    # MTGTaskPlanner
    def reset_task_planner(self):
        self.task_graph = nx.DiGraph()
        self.manifold_info = {}  # the detail of each manifold
        self.incomming_manifold_intersections = (
            {}
        )  # the incomming intersections of each manifold
        self.outgoing_manifold_intersections = (
            {}
        )  # the outgoing intersections of each manifold
        self.new_intersection_id = 0

        # self.reset_manifold_similarity_table()
        self.total_similiarity_table = {}

    # MTGTaskPlanner
    def add_manifold(self, manifold_info_, manifold_id_):
        self.manifold_info[manifold_id_] = manifold_info_

        self.incomming_manifold_intersections[manifold_id_] = []
        self.outgoing_manifold_intersections[manifold_id_] = []

    # MTGTaskPlanner
    def add_intersection(self, manifold_id1_, manifold_id2_, intersection_detail_):
        intersection_from_1_to_2_id = self.new_intersection_id
        self.new_intersection_id += 1

        # add node for intersection from manifold 1 to manifold 2
        self.task_graph.add_node(
            intersection_from_1_to_2_id,
            intersection=intersection_detail_,
            previous_manifold_id=manifold_id1_,
            next_manifold_id=manifold_id2_,
        )

        for i in self.incomming_manifold_intersections[manifold_id1_]:
            self.task_graph.add_edge(
                i,
                intersection_from_1_to_2_id,
                weight=0,
                manifold_id=manifold_id1_,
                has_solution=False,
                solution_trajectory=None,
            )
        for i in self.outgoing_manifold_intersections[manifold_id2_]:
            self.task_graph.add_edge(
                intersection_from_1_to_2_id,
                i,
                weight=0,
                manifold_id=manifold_id2_,
                has_solution=False,
                solution_trajectory=None,
            )

        self.outgoing_manifold_intersections[manifold_id1_].append(
            intersection_from_1_to_2_id
        )
        self.incomming_manifold_intersections[manifold_id2_].append(
            intersection_from_1_to_2_id
        )

    # MTGTaskPlanner
    def set_start_and_goal(
        self,
        start_manifold_id_,
        start_intersection_,
        goal_manifold_id_,
        goal_intersection_,
    ):
        # if start and goal are set, then remove them from the task graph
        if self.task_graph.has_node("start"):
            self.task_graph.remove_node("start")

        if self.task_graph.has_node("goal"):
            self.task_graph.remove_node("goal")

        configuration_of_start, _ = start_intersection_.get_edge_configurations()
        self.task_graph.add_node(
            "start",
            intersection=IntersectionDetail(
                start_intersection_,
                configuration_of_start,
                configuration_of_start,
                False,
            ),
            previous_manifold_id=None,
            next_manifold_id=start_manifold_id_,
        )

        configuration_of_goal, _ = goal_intersection_.get_edge_configurations()
        self.task_graph.add_node(
            "goal",
            intersection=IntersectionDetail(
                goal_intersection_, configuration_of_goal, configuration_of_goal, True
            ),
            previous_manifold_id=goal_manifold_id_,
            next_manifold_id=None,
        )

        for i in self.outgoing_manifold_intersections[start_manifold_id_]:
            self.task_graph.add_edge(
                "start",
                i,
                weight=0,
                manifold_id=start_manifold_id_,
                has_solution=False,
                solution_trajectory=None,
            )

        for i in self.incomming_manifold_intersections[goal_manifold_id_]:
            self.task_graph.add_edge(
                i,
                "goal",
                weight=0,
                manifold_id=goal_manifold_id_,
                has_solution=False,
                solution_trajectory=None,
            )

    # MTGTaskPlanner
    def generate_task_sequence(self):
        # check the connectivity of the task graph from start to goal
        if not nx.has_path(self.task_graph, "start", "goal"):
            print("no connection between start and goal!")
            return []

        # find the shortest path from start to goal
        shortest_path = nx.shortest_path(
            self.task_graph, "start", "goal", weight="weight"
        )
        task_sequence = []

        # construct the task sequence.
        for node1, node2 in zip(shortest_path[:-1], shortest_path[1:]):
            task = Task(
                self.manifold_info[self.task_graph.edges[node1, node2]["manifold_id"]],
                nx.get_node_attributes(self.task_graph, "intersection")[
                    node1
                ].configuration_in_manifold2,
                nx.get_node_attributes(self.task_graph, "intersection")[
                    node2
                ].configuration_in_manifold1,
                nx.get_node_attributes(self.task_graph, "intersection")[
                    node2
                ].intersection_data,
            )

            task.set_task_graph_info((node1, node2))
            # if solution exists, then set the solution trajectory to the task
            if self.task_graph.edges[node1, node2]["has_solution"]:
                task.set_solution(
                    self.task_graph.edges[node1, node2]["solution_trajectory"]
                )

            task_sequence.append(task)

        return task_sequence

    # MTGTaskPlanner
    def update(self, task_graph_info_, plan_, solution_):
        # get the current manifold id, previous manifold id and next manifold id of the task.
        current_manifold_id = self.task_graph.edges[task_graph_info_]["manifold_id"]
        previous_manifold_id = self.task_graph.nodes[task_graph_info_[0]][
            "previous_manifold_id"
        ]
        next_manifold_id = self.task_graph.nodes[task_graph_info_[1]][
            "next_manifold_id"
        ]

        if plan_[0]:
            self.task_graph.edges[task_graph_info_]["weight"] += 0.01

            # save the solution trajectory to the task graph.
            self.task_graph.edges[task_graph_info_]["has_solution"] = True
            self.task_graph.edges[task_graph_info_][
                "solution_trajectory"
            ] = solution_  # plan_[1]

        else:
            if previous_manifold_id != None and next_manifold_id != None:
                # we want to only update the edges if their have the similar current manifold, previous manifold, and next manifold.
                for e in self.task_graph.edges:
                    e_current_manifold_id = self.task_graph.edges[e]["manifold_id"]
                    e_previous_manifold_id = self.task_graph.nodes[e[0]][
                        "previous_manifold_id"
                    ]
                    e_next_manifold_id = self.task_graph.nodes[e[1]]["next_manifold_id"]

                    if e_previous_manifold_id == None or e_next_manifold_id == None:
                        continue

                    if (
                        e_current_manifold_id[0] == current_manifold_id[0]
                        and e_previous_manifold_id[0] == previous_manifold_id[0]
                        and e_next_manifold_id[0] == next_manifold_id[0]
                    ):  # we only update the edge in the same foliation.
                        # update the similarity score of the edge
                        current_similarity_score = self.total_similiarity_table[
                            current_manifold_id[0]
                        ][e_current_manifold_id[1], current_manifold_id[1]]
                        previous_similarity_score = self.total_similiarity_table[
                            previous_manifold_id[0]
                        ][e_previous_manifold_id[1], previous_manifold_id[1]]
                        next_similarity_score = self.total_similiarity_table[
                            next_manifold_id[0]
                        ][e_next_manifold_id[1], next_manifold_id[1]]
                        total_similarity_score = (
                            current_similarity_score
                            * previous_similarity_score
                            * next_similarity_score
                        )
                        self.task_graph.edges[e]["weight"] += (
                            1.0 * total_similarity_score
                        )


class MDPTaskPlanner(BaseTaskPlanner):
    def __init__(self, planner_name_="MDPTaskPlanner", parameter_dict_={}):
        # Constructor
        super(BaseTaskPlanner, self).__init__()  # python 2
        # super().__init__() # python 3

        self.planner_name = planner_name_

        self.parameter_dict = parameter_dict_

    # MDPTaskPlanner
    def reset_task_planner(self):
        self.task_graph = nx.DiGraph()
        self.manifold_info = {}  # the constraints of each manifold
        self.incomming_manifold_intersections = (
            {}
        )  # the incomming intersections of each manifold
        self.outgoing_manifold_intersections = (
            {}
        )  # the outgoing intersections of each manifold
        self.new_intersection_id = 0

        self.gamma = (
            self.parameter_dict["gamma"] if "gamma" in self.parameter_dict else 0.9
        )
        self.value_iteration_iters = (
            self.parameter_dict["value_iteration_iters"]
            if "value_iteration_iters" in self.parameter_dict
            else 100
        )
        self.epsilon = (
            self.parameter_dict["epsilon"]
            if "epsilon" in self.parameter_dict
            else 0.001
        )
        self.reward_of_goal = (
            self.parameter_dict["reward_of_goal"]
            if "reward_of_goal" in self.parameter_dict
            else 100.0
        )

        # self.reset_manifold_similarity_table()
        self.total_similiarity_table = {}

    # MDPTaskPlanner
    def add_manifold(self, manifold_info_, manifold_id_):
        self.manifold_info[manifold_id_] = manifold_info_

        self.incomming_manifold_intersections[manifold_id_] = []
        self.outgoing_manifold_intersections[manifold_id_] = []

    # MDPTaskPlanner
    def add_intersection(self, manifold_id1_, manifold_id2_, intersection_detail_):
        intersection_from_1_to_2_id = self.new_intersection_id
        self.new_intersection_id += 1

        # add node for intersection from manifold 1 to manifold 2
        self.task_graph.add_node(
            intersection_from_1_to_2_id,
            intersection=intersection_detail_,
            previous_manifold_id=manifold_id1_,
            next_manifold_id=manifold_id2_,
        )

        for i in self.incomming_manifold_intersections[manifold_id1_]:
            self.task_graph.add_edge(
                i,
                intersection_from_1_to_2_id,
                probability=0.5,
                manifold_id=manifold_id1_,
                has_solution=False,
                solution_trajectory=None,
            )
        for i in self.outgoing_manifold_intersections[manifold_id2_]:
            self.task_graph.add_edge(
                intersection_from_1_to_2_id,
                i,
                probability=0.5,
                manifold_id=manifold_id2_,
                has_solution=False,
                solution_trajectory=None,
            )

        self.outgoing_manifold_intersections[manifold_id1_].append(
            intersection_from_1_to_2_id
        )
        self.incomming_manifold_intersections[manifold_id2_].append(
            intersection_from_1_to_2_id
        )

    # MDPTaskPlanner
    def set_start_and_goal(
        self,
        start_manifold_id_,
        start_intersection_,
        goal_manifold_id_,
        goal_intersection_,
    ):
        # if start and goal are set, then remove them from the task graph
        if self.task_graph.has_node("start"):
            self.task_graph.remove_node("start")

        if self.task_graph.has_node("goal"):
            self.task_graph.remove_node("goal")

        configuration_of_start, _ = start_intersection_.get_edge_configurations()
        self.task_graph.add_node(
            "start",
            intersection=IntersectionDetail(
                start_intersection_,
                configuration_of_start,
                configuration_of_start,
                False,
            ),
            previous_manifold_id=None,
            next_manifold_id=start_manifold_id_,
        )

        configuration_of_goal, _ = goal_intersection_.get_edge_configurations()
        self.task_graph.add_node(
            "goal",
            intersection=IntersectionDetail(
                goal_intersection_, configuration_of_goal, configuration_of_goal, True
            ),
            previous_manifold_id=goal_manifold_id_,
            next_manifold_id=None,
        )

        for i in self.outgoing_manifold_intersections[start_manifold_id_]:
            self.task_graph.add_edge(
                "start",
                i,
                probability=0.5,
                manifold_id=start_manifold_id_,
                has_solution=False,
                solution_trajectory=None,
            )

        for i in self.incomming_manifold_intersections[goal_manifold_id_]:
            self.task_graph.add_edge(
                i,
                "goal",
                probability=0.5,
                manifold_id=goal_manifold_id_,
                has_solution=False,
                solution_trajectory=None,
            )

        # initialize the value function of each node
        self.value_function = {node: 0 for node in self.task_graph.nodes}

    # MDPTaskPlanner
    def generate_task_sequence(self):
        # check the connectivity of the task graph from start to goal
        if not nx.has_path(self.task_graph, "start", "goal"):
            print("no connection between start and goal!")
            return []

        self.value_iteration()

        # find the shortest path from start to goal
        shortest_path = self.plan_on_value_function()

        task_sequence = []
        # construct the task sequence.
        for node1, node2 in zip(shortest_path[:-1], shortest_path[1:]):
            task = Task(
                self.manifold_info[self.task_graph.edges[node1, node2]["manifold_id"]],
                nx.get_node_attributes(self.task_graph, "intersection")[
                    node1
                ].configuration_in_manifold2,
                nx.get_node_attributes(self.task_graph, "intersection")[
                    node2
                ].configuration_in_manifold1,
                nx.get_node_attributes(self.task_graph, "intersection")[
                    node2
                ].intersection_data,
            )

            if self.task_graph.edges[node1, node2]["has_solution"]:
                task.set_solution(
                    self.task_graph.edges[node1, node2]["solution_trajectory"]
                )

            task.set_task_graph_info((node1, node2))
            task_sequence.append(task)

        return task_sequence

    # MDPTaskPlanner
    def update(self, task_graph_info_, plan_, solution_):
        current_manifold_id = self.task_graph.edges[task_graph_info_]["manifold_id"]
        previous_manifold_id = self.task_graph.nodes[task_graph_info_[0]][
            "previous_manifold_id"
        ]
        next_manifold_id = self.task_graph.nodes[task_graph_info_[1]][
            "next_manifold_id"
        ]

        # TODO: we now ignore the case if the edge is at the beginning or the end of the task graph.
        # we may need to consider this case in the future.

        # for the edge at the beginning or the end of the task graph, we do not update the probability
        if previous_manifold_id != None and next_manifold_id != None:
            for e in self.task_graph.edges:
                e_current_manifold_id = self.task_graph.edges[e]["manifold_id"]
                e_previous_manifold_id = self.task_graph.nodes[e[0]][
                    "previous_manifold_id"
                ]
                e_next_manifold_id = self.task_graph.nodes[e[1]]["next_manifold_id"]

                if e_previous_manifold_id == None or e_next_manifold_id == None:
                    # for the edge at the beginning or the end of the task graph, we do not update the probability
                    continue

                # we only update the edge in the same foliation and they have the same
                # previous manifold and next manifold.
                if (
                    e_current_manifold_id[0] == current_manifold_id[0]
                    and e_previous_manifold_id[0] == previous_manifold_id[0]
                    and e_next_manifold_id[0] == next_manifold_id[0]
                ):
                    # update the similarity score
                    current_similarity_score = self.total_similiarity_table[
                        current_manifold_id[0]
                    ][e_current_manifold_id[1], current_manifold_id[1]]
                    previous_similarity_score = self.total_similiarity_table[
                        previous_manifold_id[0]
                    ][e_previous_manifold_id[1], previous_manifold_id[1]]
                    next_similarity_score = self.total_similiarity_table[
                        next_manifold_id[0]
                    ][e_next_manifold_id[1], next_manifold_id[1]]

                    total_similarity_score = (
                        current_similarity_score
                        * previous_similarity_score
                        * next_similarity_score
                    )

                    p = self.task_graph.edges[e]["probability"]
                    if plan_[0]:
                        self.task_graph.edges[e]["probability"] = (
                            total_similarity_score + (2.0 - total_similarity_score) * p
                        ) / 2.0
                    else:
                        self.task_graph.edges[e]["probability"] = p * (
                            1.0 - 0.5 * total_similarity_score
                        )

        if plan_[0]:
            # save the solution trajectory to the task graph.
            self.task_graph.edges[task_graph_info_]["has_solution"] = True
            self.task_graph.edges[task_graph_info_][
                "solution_trajectory"
            ] = solution_  # plan_[1]
            # set the current edge's probability to 1.0
            self.task_graph.edges[task_graph_info_]["probability"] = 1.0


class MTGTaskPlannerWithGMM(BaseTaskPlanner):
    def __init__(self, gmm, planner_name_="MTGTaskPlannerWithGMM", parameter_dict_={}):
        # Constructor
        super(BaseTaskPlanner, self).__init__()
        # super().__init__() # python 3

        self.gmm_ = gmm

        self.planner_name = planner_name_

        self.parameter_dict = parameter_dict_

    # MTGTaskPlannerWithGMM
    def reset_task_planner(self):
        self.reset_task_solution_graph()

        self.task_graph = nx.DiGraph()
        self.manifold_info = {}  # the constraints of each manifold

        # self.reset_manifold_similarity_table()
        self.total_similiarity_table = {}

    # MTGTaskPlannerWithGMM
    def read_pointcloud(self, pointcloud_):
        self.gmm_.update_collision_free_rates(pointcloud_)

        # update the weight of each edge in the task graph based on the collision free rate of the GMM
        # update the weight of each distribution in the task graph.
        for e in self.task_graph.edges:
            # if this is the edge accross different manifolds, then skip it for now.
            if self.task_graph.edges[e]["has_intersection"]:
                continue

            if e[0] == "start" or e[1] == "goal":
                # for the edge at the beginning or the end of the task graph, we do not update the probability
                continue

            # get the distribution id of the edge
            distribution_id_1 = e[0][2]
            distribution_id_2 = e[1][2]

            # update the weight of the edge by summing up the collision free rate of the two distributions
            self.task_graph.edges[e]["weight"] = (
                1.0 - self.gmm_.collision_free_rates[distribution_id_1]
            ) + (1.0 - self.gmm_.collision_free_rates[distribution_id_2])

    # MTGTaskPlannerWithGMM
    def add_manifold(self, manifold_info_, manifold_id_):
        self.add_manifold_for_task_solution_graph(manifold_id_)

        self.manifold_info[manifold_id_] = manifold_info_

        # construct a set of nodes represented by a tuple (foliation id, manifold id, GMM id)
        for i in range(len(self.gmm_.distributions)):
            self.task_graph.add_node(
                (manifold_id_[0], manifold_id_[1], i)
            )  # , weight = 0.0)

        for edge in self.gmm_.edge_of_distribution:
            self.task_graph.add_edge(
                (manifold_id_[0], manifold_id_[1], edge[0]),
                (manifold_id_[0], manifold_id_[1], edge[1]),
                weight=0.0,
                has_intersection=False,
                intersection_id=None,
                intersection=None,
            )

            # need to add the inverse edge
            self.task_graph.add_edge(
                (manifold_id_[0], manifold_id_[1], edge[1]),
                (manifold_id_[0], manifold_id_[1], edge[0]),
                weight=0.0,
                has_intersection=False,
                intersection_id=None,
                intersection=None,
            )

    # MTGTaskPlannerWithGMM
    def add_intersection(self, manifold_id1_, manifold_id2_, intersection_detail_):
        # connect two distribution of this intersection_detail_ between two different manifolds(manifold1 and manifold2) if they have the same ditribution id in GMM.
        # first, find the related distribution that the intersection's ends are in in different manifolds.

        (
            distribution_id_in_manifold1,
            distribution_id_in_manifold2,
        ) = self.gmm_.get_distribution_indexs(
            [
                intersection_detail_.configuration_in_manifold1,
                intersection_detail_.configuration_in_manifold2,
            ]
        )

        # if(not self.task_graph.has_edge(
        #         (manifold_id1_[0], manifold_id1_[1], distribution_id_in_manifold1),
        #         (manifold_id2_[0], manifold_id2_[1], distribution_id_in_manifold2)) and
        #    not self.task_graph.has_edge(
        #         (manifold_id2_[0], manifold_id2_[1], distribution_id_in_manifold2),
        #         (manifold_id1_[0], manifold_id1_[1], distribution_id_in_manifold1))
        # ):
        # intersection_from_1_to_2_id, intersection_from_2_to_1_id = self.add_intersection_for_task_solution_graph(manifold_id1_, manifold_id2_)
        intersection_from_1_to_2_id = self.add_intersection_for_task_solution_graph(
            manifold_id1_, manifold_id2_
        )

        self.task_graph.add_edge(
            (manifold_id1_[0], manifold_id1_[1], distribution_id_in_manifold1),
            (manifold_id2_[0], manifold_id2_[1], distribution_id_in_manifold2),
            weight=0.0,
            has_intersection=True,
            intersection=intersection_detail_,
            intersection_id=intersection_from_1_to_2_id,
        )

    # MTGTaskPlannerWithGMM
    def set_start_and_goal(
        self,
        start_manifold_id_,
        start_intersection_,
        goal_manifold_id_,
        goal_intersection_,
    ):
        self.set_start_and_goal_for_task_solution_graph(
            start_manifold_id_, goal_manifold_id_
        )

        # if start and goal are set, then remove them from the task graph
        if self.task_graph.has_node("start"):
            self.task_graph.remove_node("start")

        if self.task_graph.has_node("goal"):
            self.task_graph.remove_node("goal")

        # include start and goal configurations in the task graph
        self.task_graph.add_node("start")  # , weight = 0.0)
        self.task_graph.add_node("goal")  # , weight = 0.0)

        configuration_of_start, _ = start_intersection_.get_edge_configurations()
        self.task_graph.add_edge(
            "start",
            (
                start_manifold_id_[0],
                start_manifold_id_[1],
                self.gmm_.get_distribution_index(np.array(configuration_of_start)),
            ),
            weight=0.0,
            has_intersection=False,
            intersection=None,
            intersection_id=None,
        )

        configuration_of_goal, _ = goal_intersection_.get_edge_configurations()
        self.task_graph.add_edge(
            (
                goal_manifold_id_[0],
                goal_manifold_id_[1],
                self.gmm_.get_distribution_index(np.array(configuration_of_goal)),
            ),
            "goal",
            weight=0.0,
            has_intersection=True,
            intersection=IntersectionDetail(
                goal_intersection_, configuration_of_goal, configuration_of_goal, True
            ),
            intersection_id="goal",
        )

        self.current_start_configuration = configuration_of_start

    # MTGTaskPlannerWithGMM
    def generate_task_sequence(self):
        # print the number of nodes can achieve the goal
        # print "number of nodes can achieve the goal: ", len([node for node in self.task_graph.nodes if nx.has_path(self.task_graph, node, 'goal')])

        # check the connectivity of the task graph from start to goal
        if not nx.has_path(self.task_graph, "start", "goal"):
            print("no connection between start and goal!")
            return []

        # find the shortest path from start to goal
        shortest_path = nx.shortest_path(
            self.task_graph, "start", "goal", weight="weight"
        )

        task_sequence = []

        task_start_configuration = self.current_start_configuration
        task_gaussian_distribution = []
        last_intersection_id = "start"

        # construct the task sequence.
        for node1, node2 in zip(shortest_path[:-1], shortest_path[1:]):
            current_edge = self.task_graph.get_edge_data(node1, node2)

            if current_edge["has_intersection"]:
                # current edge is a transition from one manifold to another manifold
                task = Task(
                    self.manifold_info[
                        self.get_manifold_id_from_task_solution_graph(
                            last_intersection_id, current_edge["intersection_id"]
                        )
                    ],
                    task_start_configuration,  # start configuration of the task
                    current_edge[
                        "intersection"
                    ].configuration_in_manifold1,  # target configuration of the task
                    current_edge[
                        "intersection"
                    ].intersection_data,  # the motion after the task.
                )

                task.distributions = list(task_gaussian_distribution)

                # we use the intersection id as task graph information here
                task.set_task_graph_info(
                    (last_intersection_id, current_edge["intersection_id"])
                )

                # check if solution exists for this task, if yes, then set the solution trajectory to the task
                existing_solution = self.check_solution_existence(
                    last_intersection_id, current_edge["intersection_id"]
                )
                if existing_solution != None:
                    task.set_solution(existing_solution)

                task_sequence.append(task)

                # ready for the next task.
                if (
                    node2 != "goal"
                ):  # if the edge is to goal, then no need to prepare for the next task
                    task_gaussian_distribution = [self.gmm_.distributions[node2[2]]]
                    # consider the last state of the intersection motion as the start state of next task.
                    task_start_configuration = current_edge[
                        "intersection"
                    ].configuration_in_manifold2
                    last_intersection_id = current_edge["intersection_id"]
            else:
                # edge in the same manifold except start and goal transition
                task_gaussian_distribution.append(self.gmm_.distributions[node2[2]])

        return task_sequence

    # MTGTaskPlannerWithGMM
    def update(self, task_graph_info_, plan_, solution_):
        # use the sample data to update the task graph.
        # sampled_state_tag hint
        # 0: collision free
        # 1: arm-env collision or out of joint limit
        # 2: path constraint violation
        # 3: infeasble state, you should ignore this
        # 4: obj-env collision

        # if sampled data is empty, then skip it.
        if len(plan_[4].verified_motions) == 0:
            print("sampled data is empty.")
            return

        sampled_data_numpy = np.array(
            [sampled_data.sampled_state for sampled_data in plan_[4].verified_motions]
        )

        # if sampled_data_numpy is empty, then skip it.
        sampled_data_distribution_id = self.gmm_._sklearn_gmm.predict(
            sampled_data_numpy
        ).tolist()
        current_manifold_id = self.get_manifold_id_from_task_solution_graph(
            task_graph_info_[0], task_graph_info_[1]
        )

        if plan_[0]:
            # save the solution if it exists.
            self.save_solution_to_task_solution_graph(
                task_graph_info_[0], task_graph_info_[1], solution_
            )  # plan_[1])

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

        for i in range(len(sampled_data_distribution_id)):
            sampled_data_gmm_id = sampled_data_distribution_id[i]
            sampled_data_tag = plan_[4].verified_motions[i].sampled_state_tag

            if sampled_data_tag == 0:
                sampled_data_distribution_tag_table[sampled_data_gmm_id][0] += 1
            elif sampled_data_tag == 1:
                sampled_data_distribution_tag_table[sampled_data_gmm_id][1] += 1
            elif sampled_data_tag == 2:
                sampled_data_distribution_tag_table[sampled_data_gmm_id][2] += 1
            elif sampled_data_tag == 4:
                sampled_data_distribution_tag_table[sampled_data_gmm_id][3] += 1

        # update the weight of each distribution in the task graph.
        for e in self.task_graph.edges:
            # if this is the edge accross different manifolds, then skip it for now.
            if self.task_graph.edges[e]["has_intersection"]:
                continue

            # get the manifold id of this edge(both nodes are in the same manifold)
            e_current_manifold_id = e[0][:2]

            # if this edge is in different foliation, then skip it. We only update the edge in the same foliation.
            if e_current_manifold_id[0] != current_manifold_id[0]:
                continue

            # get the current similarity score between the current manifold and the manifold of this edge
            # current_similarity_score = self.total_similiarity_table[current_manifold_id[0]][(e_current_manifold_id[1], current_manifold_id[1])]
            current_similarity_score = self.total_similiarity_table[
                current_manifold_id[0]
            ][e_current_manifold_id[1], current_manifold_id[1]]

            # get the distribution id of the edge
            distribution_id_1 = e[0][2]
            distribution_id_2 = e[1][2]

            collision_free_score = (
                (
                    sampled_data_distribution_tag_table[distribution_id_1][0]
                    + sampled_data_distribution_tag_table[distribution_id_2][0]
                )
                * 0.01
                * current_similarity_score
            )
            arm_env_collision_score = (
                sampled_data_distribution_tag_table[distribution_id_1][1]
                + sampled_data_distribution_tag_table[distribution_id_2][1]
            ) * 1.0
            path_constraint_violation_score = (
                (
                    sampled_data_distribution_tag_table[distribution_id_1][2]
                    + sampled_data_distribution_tag_table[distribution_id_2][2]
                )
                * 1.0
                * current_similarity_score
            )
            obj_env_collision_score = (
                (
                    sampled_data_distribution_tag_table[distribution_id_1][3]
                    + sampled_data_distribution_tag_table[distribution_id_2][3]
                )
                * 1.0
                * current_similarity_score
            )

            self.task_graph.edges[e]["weight"] += (
                collision_free_score
                + arm_env_collision_score
                + path_constraint_violation_score
                + obj_env_collision_score
            )


class MDPTaskPlannerWithGMM(BaseTaskPlanner):
    def __init__(self, gmm, planner_name_="MDPTaskPlannerWithGMM", parameter_dict_={}):
        # Init the constructor
        super(BaseTaskPlanner, self).__init__()

        self.gmm_ = gmm

        self.planner_name = planner_name_

        self.parameter_dict = parameter_dict_

    # MDPTaskPlannerWithGMM
    def reset_task_planner(self):
        self.reset_task_solution_graph()

        self.task_graph = nx.DiGraph()
        self.manifold_info = {}  # the constraints of each manifold
        self.gamma = (
            self.parameter_dict["gamma"] if "gamma" in self.parameter_dict else 0.99
        )
        self.epsilon = (
            self.parameter_dict["epsilon"] if "epsilon" in self.parameter_dict else 1e-5
        )
        self.value_iteration_iters = (
            self.parameter_dict["value_iteration_iters"]
            if "value_iteration_iters" in self.parameter_dict
            else 100
        )
        self.reward_of_goal = (
            self.parameter_dict["reward_of_goal"]
            if "reward_of_goal" in self.parameter_dict
            else 10.0
        )
        self.use_shortcut = (
            self.parameter_dict["use_shortcut"]
            if "use_shortcut" in self.parameter_dict
            else False
        )
        self.shortcut_probability = (
            self.parameter_dict["shortcut_probability"]
            if "shortcut_probability" in self.parameter_dict
            else 0.5
        )
        self.lowest_probability = (
            self.parameter_dict["lowest_probability"]
            if "lowest_probability" in self.parameter_dict
            else 0.7
        )

        # this table contains the arm_env_collision count for each distribution in GMM
        self.gmm_arm_env_collision_count = {
            distribution_id: 0
            for distribution_id in range(len(self.gmm_.distributions))
        }
        # self.reset_manifold_similarity_table()
        self.total_similiarity_table = {}

    # MDPTaskPlannerWithGMM
    def read_pointcloud(self, pointcloud_):
        self.gmm_.update_collision_free_rates(pointcloud_)

        # update collision_count of distributions based on the collision free rate of the GMM
        for distribution_id in range(0, len(self.gmm_.distributions)):
            self.gmm_arm_env_collision_count[distribution_id] += (
                1.0 - self.gmm_.collision_free_rates[distribution_id]
            ) * 2.0  # by increase this value to improve the impact of collision free rate on the possibility of each edge.

        # update the possibility of each distribution in the task graph.
        for e in self.task_graph.edges:
            # if this is the edge accross different manifolds, then skip it for now.
            if self.task_graph.edges[e]["has_intersection"]:
                continue

            if e[0] == "start" or e[1] == "goal":
                # for the edge at the beginning or the end of the task graph, we do not update the probability
                continue

            # get the distribution id of the edge
            distribution_id_1 = e[0][2]
            distribution_id_2 = e[1][2]

            # update the possibility of the edge by averaging the collision free rate of the two distributions
            self.task_graph.edges[e]["probability"] = (
                self.gmm_.collision_free_rates[distribution_id_1]
                + self.gmm_.collision_free_rates[distribution_id_2]
            ) / 2.0

    # MDPTaskPlannerWithGMM
    def add_manifold(self, manifold_info_, manifold_id_):
        self.add_manifold_for_task_solution_graph(manifold_id_)

        self.manifold_info[manifold_id_] = manifold_info_

        # construct a set of nodes represented by a tuple (foliation id, manifold id, GMM id)
        for i in range(len(self.gmm_.distributions)):
            self.task_graph.add_node(
                (manifold_id_[0], manifold_id_[1], i),
                collision_free_count=0,
                path_constraint_violation_count=0,
                obj_env_collision_count=0,
            )

        for edge in self.gmm_.edge_of_distribution:
            self.task_graph.add_edge(
                (manifold_id_[0], manifold_id_[1], edge[0]),
                (manifold_id_[0], manifold_id_[1], edge[1]),
                has_intersection=False,
                intersection=None,
                intersection_id=None,
                probability=0.5 * self.lowest_probability + 0.5,
                # probability = 0.5,
            )

            # need to add the inverse edge
            self.task_graph.add_edge(
                (manifold_id_[0], manifold_id_[1], edge[1]),
                (manifold_id_[0], manifold_id_[1], edge[0]),
                has_intersection=False,
                intersection=None,
                intersection_id=None,
                probability=0.5 * self.lowest_probability + 0.5,
                # probability = 0.5,
            )

    # MDPTaskPlannerWithGMM
    def add_intersection(self, manifold_id1_, manifold_id2_, intersection_detail_):
        # connect two distribution of this intersection_detail_ between two different manifolds(manifold1 and manifold2) if they have the same ditribution id in GMM.
        # first, find the related distribution that the intersection's ends are in in different manifolds.

        (
            distribution_id_in_manifold1,
            distribution_id_in_manifold2,
        ) = self.gmm_.get_distribution_indexs(
            [
                intersection_detail_.configuration_in_manifold1,
                intersection_detail_.configuration_in_manifold2,
            ]
        )

        # if(not self.task_graph.has_edge(
        #         (manifold_id1_[0], manifold_id1_[1], distribution_id_in_manifold1),
        #         (manifold_id2_[0], manifold_id2_[1], distribution_id_in_manifold2))
        # ):
        intersection_from_1_to_2_id = self.add_intersection_for_task_solution_graph(
            manifold_id1_, manifold_id2_
        )

        self.task_graph.add_edge(
            (manifold_id1_[0], manifold_id1_[1], distribution_id_in_manifold1),
            (manifold_id2_[0], manifold_id2_[1], distribution_id_in_manifold2),
            has_intersection=True,
            intersection=intersection_detail_,
            intersection_id=intersection_from_1_to_2_id,
            probability=0.5 * self.lowest_probability + 0.5,
        )

    # MDPTaskPlannerWithGMM
    def add_intersections(
        self, list_of_manifold_id1_, list_of_manifold_id2_, list_of_intersection_detail_
    ):
        # find the related distribution that the intersection's ends are in in different manifolds.
        list_of_configuration1 = [
            i.trajectory_motion[0] for i in list_of_intersection_detail_
        ]
        list_of_configuration2 = [
            i.trajectory_motion[-1] for i in list_of_intersection_detail_
        ]

        list_of_distribution_id_in_manifold1 = self.gmm_.get_distribution_indexs(
            list_of_configuration1
        )
        list_of_distribution_id_in_manifold2 = self.gmm_.get_distribution_indexs(
            list_of_configuration2
        )

        for (
            distribution_id_in_manifold1,
            distribution_id_in_manifold2,
            manifold_id1,
            manifold_id2,
            intersection_detail,
        ) in zip(
            list_of_distribution_id_in_manifold1,
            list_of_distribution_id_in_manifold2,
            list_of_manifold_id1_,
            list_of_manifold_id2_,
            list_of_intersection_detail_,
        ):
            if not self.task_graph.has_edge(
                (manifold_id1[0], manifold_id1[1], distribution_id_in_manifold1),
                (manifold_id2[0], manifold_id2[1], distribution_id_in_manifold2),
            ):
                intersection_from_1_to_2_id = (
                    self.add_intersection_for_task_solution_graph(
                        manifold_id1, manifold_id2
                    )
                )

                self.task_graph.add_edge(
                    (manifold_id1[0], manifold_id1[1], distribution_id_in_manifold1),
                    (manifold_id2[0], manifold_id2[1], distribution_id_in_manifold2),
                    has_intersection=True,
                    intersection=intersection_detail,
                    intersection_id=intersection_from_1_to_2_id,
                    probability=0.5 * self.lowest_probability + 0.5,
                )

    # MDPTaskPlannerWithGMM
    def set_start_and_goal(
        self,
        start_manifold_id_,
        start_intersection_,
        goal_manifold_id_,
        goal_intersection_,
    ):
        self.set_start_and_goal_for_task_solution_graph(
            start_manifold_id_, goal_manifold_id_
        )

        # if start and goal are set, then remove them from the task graph
        if self.task_graph.has_node("start"):
            self.task_graph.remove_node("start")

        if self.task_graph.has_node("goal"):
            self.task_graph.remove_node("goal")

        # include start and goal configurations in the task graph
        self.task_graph.add_node(
            "start",
            collision_free_count=0,
            path_constraint_violation_count=0,
            obj_env_collision_count=0,
        )
        self.task_graph.add_node(
            "goal",
            collision_free_count=0,
            path_constraint_violation_count=0,
            obj_env_collision_count=0,
        )

        configuration_of_start, _ = start_intersection_.get_edge_configurations()
        self.task_graph.add_edge(
            "start",
            (
                start_manifold_id_[0],
                start_manifold_id_[1],
                self.gmm_.get_distribution_index(np.array(configuration_of_start)),
            ),
            has_intersection=False,
            intersection=None,
            intersection_id=None,
            probability=1.0,
        )

        configuration_of_goal, _ = goal_intersection_.get_edge_configurations()
        self.task_graph.add_edge(
            (
                goal_manifold_id_[0],
                goal_manifold_id_[1],
                self.gmm_.get_distribution_index(np.array(configuration_of_goal)),
            ),
            "goal",
            has_intersection=True,
            intersection=IntersectionDetail(
                goal_intersection_, configuration_of_goal, configuration_of_goal, True
            ),
            intersection_id="goal",
            probability=1.0,
        )

        # remove the nodes which are not reachable from start
        reachable_nodes = set(
            nx.single_source_shortest_path_length(self.task_graph, "start").keys()
        )
        removed_nodes = set(self.task_graph.nodes) - reachable_nodes
        for node in removed_nodes:
            self.task_graph.remove_node(node)

        self.current_start_configuration = configuration_of_start

        # initialize the value function of each node
        self.value_function = {node: 0 for node in self.task_graph.nodes}

    # MDPTaskPlannerWithGMM
    def generate_task_sequence(self):
        # check the connectivity of the task graph from start to goal
        if not self.task_graph.has_node("goal"):
            print("no connection between start and goal!")
            return []

        # self.new_value_iteration()
        self.value_iteration()

        # find the shortest path from start to goal
        shortest_path = self.plan_on_value_function()

        task_sequence = []

        task_start_configuration = self.current_start_configuration
        task_gaussian_distribution = []
        last_intersection_id = "start"

        # construct the task sequence.
        for node1, node2 in zip(shortest_path[:-1], shortest_path[1:]):
            current_edge = self.task_graph.get_edge_data(node1, node2)

            if current_edge["has_intersection"]:
                # current edge is a transition from one manifold to another manifold
                task = Task(
                    self.manifold_info[
                        self.get_manifold_id_from_task_solution_graph(
                            last_intersection_id, current_edge["intersection_id"]
                        )
                    ],
                    task_start_configuration,
                    current_edge["intersection"].configuration_in_manifold1,
                    current_edge["intersection"].intersection_data,
                )

                task.distributions = list(task_gaussian_distribution)

                # we use the intersection id as task graph information here
                task.set_task_graph_info(
                    (last_intersection_id, current_edge["intersection_id"])
                )

                existing_solution = self.check_solution_existence(
                    last_intersection_id, current_edge["intersection_id"]
                )
                if existing_solution != None:
                    task.set_solution(existing_solution)

                task_sequence.append(task)

                # ready for the next task.
                if (
                    node2 != "goal"
                ):  # if the edge is to goal, then no need to prepare for the next task
                    task_gaussian_distribution = [self.gmm_.distributions[node2[2]]]
                    # consider the last state of the intersection motion as the start state of next task
                    # task_start_configuration = current_edge['intersection'].trajectory_motion[-1]
                    task_start_configuration = current_edge[
                        "intersection"
                    ].configuration_in_manifold2
                    last_intersection_id = current_edge["intersection_id"]
            else:
                # edge in the same manifold except start and goal transition
                task_gaussian_distribution.append(self.gmm_.distributions[node2[2]])

        return task_sequence

    # MDPTaskPlannerWithGMM
    def update(self, task_graph_info_, plan_, solution_):
        # use the sample data to update the task graph.
        # sampled_state_tag hint
        # 0: collision free
        # 1: arm-env collision or out of joint limit
        # 2: path constraint violation
        # 3: infeasble state, you should ignore this
        # 4: obj-env collision

        # if sampled data is empty, then skip it.
        if len(plan_[4].verified_motions) == 0:
            print("sampled data is empty.")
            return

        sampled_data_numpy = np.array(
            [sampled_data.sampled_state for sampled_data in plan_[4].verified_motions]
        )
        sampled_data_distribution_id = self.gmm_._sklearn_gmm.predict(
            sampled_data_numpy
        ).tolist()

        current_manifold_id = self.get_manifold_id_from_task_solution_graph(
            task_graph_info_[0], task_graph_info_[1]
        )

        if plan_[0]:
            # save the solution if it exists.
            self.save_solution_to_task_solution_graph(
                task_graph_info_[0], task_graph_info_[1], solution_
            )  # plan_[1])

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

        for i in range(len(sampled_data_distribution_id)):
            sampled_data_gmm_id = sampled_data_distribution_id[i]
            sampled_data_tag = plan_[4].verified_motions[i].sampled_state_tag

            if sampled_data_tag == 0:
                sampled_data_distribution_tag_table[sampled_data_gmm_id][0] += 1
            elif sampled_data_tag == 1:
                sampled_data_distribution_tag_table[sampled_data_gmm_id][1] += 1
            elif sampled_data_tag == 2:
                sampled_data_distribution_tag_table[sampled_data_gmm_id][2] += 1
            elif sampled_data_tag == 4:
                sampled_data_distribution_tag_table[sampled_data_gmm_id][3] += 1

        # update the count value of each distribution in this manifold.
        # update the arm-env collision count of each distribution
        for distribution_id in range(0, len(self.gmm_.distributions)):
            self.gmm_arm_env_collision_count[
                distribution_id
            ] += sampled_data_distribution_tag_table[distribution_id][1]
            for manifold_id in self.manifold_info.keys():
                # only update the manifold in the same foliation
                if manifold_id[0] == current_manifold_id[
                    0
                ] and self.task_graph.has_node(
                    (manifold_id[0], manifold_id[1], distribution_id)
                ):
                    # current_similarity_score = self.total_similiarity_table[current_manifold_id[0]][(manifold_id[1], current_manifold_id[1])]
                    current_similarity_score = self.total_similiarity_table[
                        current_manifold_id[0]
                    ][manifold_id[1], current_manifold_id[1]]

                    if current_similarity_score == 0.0:
                        continue

                    self.task_graph.nodes[
                        (manifold_id[0], manifold_id[1], distribution_id)
                    ]["collision_free_count"] += (
                        sampled_data_distribution_tag_table[distribution_id][0]
                        * current_similarity_score
                    )
                    self.task_graph.nodes[
                        (manifold_id[0], manifold_id[1], distribution_id)
                    ]["path_constraint_violation_count"] += (
                        sampled_data_distribution_tag_table[distribution_id][2]
                        * current_similarity_score
                    )
                    self.task_graph.nodes[
                        (manifold_id[0], manifold_id[1], distribution_id)
                    ]["obj_env_collision_count"] += (
                        sampled_data_distribution_tag_table[distribution_id][3]
                        * current_similarity_score
                    )

        # update the weight of each distribution in the task graph.
        for e in self.task_graph.edges:
            # if this is the edge accross different manifolds, then skip it for now.
            if self.task_graph.edges[e]["has_intersection"]:
                continue

            # get the manifold id of this edge(both nodes are in the same manifold)
            e_current_manifold_id = e[0][:2]

            # if this edge is in different foliation, then skip it. We only update the edge in the same foliation.
            if e_current_manifold_id[0] != current_manifold_id[0]:
                continue

            # get the distribution id of the edge
            distribution_id_1 = e[0][2]
            distribution_id_2 = e[1][2]

            collision_free_score = (
                self.task_graph.nodes[e[0]]["collision_free_count"]
                + self.task_graph.nodes[e[1]]["collision_free_count"]
            )
            path_constraint_violation_score = (
                self.task_graph.nodes[e[0]]["path_constraint_violation_count"]
                + self.task_graph.nodes[e[1]]["path_constraint_violation_count"]
            )
            obj_env_collision_score = (
                self.task_graph.nodes[e[0]]["obj_env_collision_count"]
                + self.task_graph.nodes[e[1]]["obj_env_collision_count"]
            )
            arm_env_collision_score = (
                self.gmm_arm_env_collision_count[distribution_id_1]
                + self.gmm_arm_env_collision_count[distribution_id_2]
            )

            # for the case that all scores are 0, we set the probability to 0.5
            if (
                collision_free_score
                + path_constraint_violation_score
                + obj_env_collision_score
                + arm_env_collision_score
            ) == 0:
                self.task_graph.edges[e]["probability"] = (
                    0.5 * self.lowest_probability + 0.5
                )
            else:
                new_probability = (1 + collision_free_score) / (
                    1
                    + collision_free_score
                    + path_constraint_violation_score
                    + obj_env_collision_score
                    + arm_env_collision_score
                )
                self.task_graph.edges[e]["probability"] = (
                    self.lowest_probability
                    + (1.0 - self.lowest_probability) * new_probability
                )

        if self.use_shortcut and plan_[0]:
            # find the start node and goal node of the solution
            solution_start_node = "start"
            solution_goal_node = "goal"

            for e in self.task_graph.edges:
                if self.task_graph.edges[e]["has_intersection"]:
                    if (
                        self.task_graph.edges[e]["intersection_id"]
                        == task_graph_info_[0]
                    ):
                        solution_start_node = e[1]

                    if (
                        self.task_graph.edges[e]["intersection_id"]
                        == task_graph_info_[1]
                    ):
                        solution_goal_node = e[0]

            # add edge from solution_start_node to solution_goal_node with 1.0 probability
            self.task_graph.add_edge(
                solution_start_node,
                solution_goal_node,
                has_intersection=False,
                intersection=None,
                intersection_id=None,
                probability=self.shortcut_probability,
            )
