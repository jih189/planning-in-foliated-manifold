#!/usr/bin/env python
import os
import json
import numpy as np
from tqdm import tqdm
from abc import ABCMeta, abstractmethod
import networkx as nx

# user needs to implement this function
class BaseIntersection:
    __metaclass__ = ABCMeta
    """
        This class represents a base intersection.
        It is an abstract class. It is used to represent the intersection of two manifolds.
        User needs to implement both get and inverse functions.
    """

    def set_foliation_names_and_co_parameter_indexes(
        self, foliation1_name, co_parameter1_index, foliation2_name, co_parameter2_index
    ):
        """Set the foliation and co-parameter index"""
        self.foliation1_name = foliation1_name
        self.co_parameter1_index = co_parameter1_index
        self.foliation2_name = foliation2_name
        self.co_parameter2_index = co_parameter2_index

    def get_foliation_names_and_co_parameter_indexes(self):
        """Get the foliation and co-parameter index"""
        return (
            self.foliation1_name,
            self.co_parameter1_index,
            self.foliation2_name,
            self.co_parameter2_index,
        )

    @abstractmethod
    def get_edge_configurations(self):
        """Return two edge configurations of intersection motion"""
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def inverse(self):
        """Return the inverse of the intersection, user needs to implement this function"""
        """Inverse function does not inverse the foliation and co-parameter index."""
        # Return inversed the intersection
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def save(self, file_path):
        """Save the intersection"""
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def get_task_motion(self):
        """Return the task motion"""
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    @abstractmethod
    def load(file_path):
        """Load the intersection"""
        raise NotImplementedError("Please Implement this method")


class BaseFoliation:
    __metaclass__ = ABCMeta

    def __init__(
        self,
        foliation_name,
        constraint_parameters,
        co_parameters=[],
        similarity_matrix=None,
    ):
        if not isinstance(co_parameters, list):
            raise Exception("co_parameters is not a list")
        if co_parameters.__len__() == 0:
            raise Exception("co_parameters is empty")
        # check if constraint_parameters is a dictionary
        if not isinstance(constraint_parameters, dict):
            raise Exception("constraint_parameters is not a dictionary")
        # check if constraint_parameters is empty
        if constraint_parameters.__len__() == 0:
            raise Exception("constraint_parameters is empty")
        # check if similarity_matrix is a numpy array
        if similarity_matrix is not None and not isinstance(
            similarity_matrix, np.ndarray
        ):
            raise Exception("similarity_matrix is not a numpy array")
        # check if size of similarity_matrix is correct
        if similarity_matrix is not None and similarity_matrix.shape != (
            co_parameters.__len__(),
            co_parameters.__len__(),
        ):
            raise Exception("similarity_matrix has incorrect size")

        self.foliation_name = foliation_name
        self.constraint_parameters = constraint_parameters  # constraint_parameters is a set of constraint parameters in directory.
        self.co_parameters = co_parameters  # list of co-parameters
        self.similarity_matrix = (
            similarity_matrix  # similarity matrix between co-parameters
        )

    @abstractmethod
    def save(self, file_path):
        """Save the foliation"""
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    @abstractmethod
    def load(file_path):
        """Load the foliation and return a foliation object"""
        raise NotImplementedError("Please Implement this method")


class FoliatedIntersection:
    """This class represents a foliated intersection"""

    def __init__(
        self,
        foliation1,
        foliation2,
        sampling_function,
        prepare_sample_function=None,
        sample_done_function=None,
    ):
        # check if the input is BaseFoliation class
        if not isinstance(foliation1, BaseFoliation):
            raise Exception("foliation1 is not a BaseFoliation class")
        if not isinstance(foliation2, BaseFoliation):
            raise Exception("foliation2 is not a BaseFoliation class")
        if not callable(sampling_function):
            raise Exception("sampling_function is not a function")

        self.foliation1 = foliation1
        self.foliation2 = foliation2
        # the sampling function will receive two list of co_parameters from each foliation, then return a BaseIntersection class.
        self.prepare_sample_function = prepare_sample_function
        self.sampling_function = sampling_function
        self.sample_done_function = sample_done_function

    def prepare_sample(self):
        if self.prepare_sample_function is None:
            return
        self.prepare_sample_function()

    def sample_done(self):
        if self.sample_done_function is None:
            return
        self.sample_done_function()

    def sample(self):
        """
        Sample a configuration from the foliated intersection.
        The sampling function will receive two list of co_parameters from each foliation, then return a success flag, co_parameter_index from first foliation, co_paramter_index from second foliation, and BaseIntersection class.
        """

        (
            success_flag,
            co_parameter1_index,
            co_parameter2_index,
            sampled_intersection,
        ) = self.sampling_function(
            self.foliation1, self.foliation2
        )

        if not isinstance(success_flag, bool):
            raise Exception(
                "The first return value of sampling function is not a boolean value!!!"
            )

        if not isinstance(co_parameter1_index, int):
            raise Exception(
                "The second return value(the index of sampled co-parameter1) of sampling function is not an integer value!!!"
            )

        if not isinstance(co_parameter2_index, int):
            raise Exception(
                "The third return value(the index of sampled co-parameter2) of sampling function is not an integer value!!!"
            )

        if (
            not success_flag
        ):  # if the success flag is false, then return false and whatever.
            return (
                success_flag,
                co_parameter1_index,
                co_parameter2_index,
                sampled_intersection,
            )

        if not isinstance(sampled_intersection, BaseIntersection):
            raise Exception("Sampled intersection is not a BaseIntersection class")

        return (
            success_flag,
            co_parameter1_index,
            co_parameter2_index,
            sampled_intersection,
        )


class FoliatedProblem:
    def __init__(self, problem_name):
        """Constructor for FoliatedProblem class"""
        self.problem_name = problem_name
        self.has_setup = False
        self.foliations = []
        self.intersections = []  # list of intersections.
        self.foliated_intersections = []
        self.start_manifold_candidates = []
        self.goal_manifold_candidates = []

    def get_foliation_index(self, foliation_name):
        """Return the index of the foliation"""
        for i, foliation in enumerate(self.foliations):
            if foliation.foliation_name == foliation_name:
                return i
        raise Exception("The foliation does not exist!!!")

    def set_foliation_n_foliated_intersection(self, foliations, foliated_intersections):
        """Set foliations and foliated intersections to the experiment"""
        self.foliations = foliations
        self.foliated_intersections = foliated_intersections

        self.has_setup = True

    def sample_intersections(self, number_of_samples=500):
        if not self.has_setup:
            raise Exception("The foliated problem has not been setup!!!")

        for foliated_intersection in self.foliated_intersections:
            # print "sample between " + foliated_intersection.foliation1.foliation_name + " and " + foliated_intersection.foliation2.foliation_name
            foliated_intersection.prepare_sample()

            # use tqdm to show the progress bar
            for i in tqdm(range(0, number_of_samples)):
                (
                    success_flag,
                    co_parameter1_index,
                    co_parameter2_index,
                    sampled_intersection,
                ) = foliated_intersection.sample()

                if success_flag:
                    # print 'co_parameter1_index: ' + str(co_parameter1_index) + ', co_parameter2_index: ' + str(co_parameter2_index)

                    sampled_intersection.set_foliation_names_and_co_parameter_indexes(
                        foliated_intersection.foliation1.foliation_name,
                        co_parameter1_index,
                        foliated_intersection.foliation2.foliation_name,
                        co_parameter2_index,
                    )
                    self.intersections.append(sampled_intersection)

                    # inverse the intersection
                    inversed_sampled_intersection = sampled_intersection.inverse()
                    inversed_sampled_intersection.set_foliation_names_and_co_parameter_indexes(
                        foliated_intersection.foliation2.foliation_name,
                        co_parameter2_index,
                        foliated_intersection.foliation1.foliation_name,
                        co_parameter1_index,
                    )

                    # append the inverse intersection
                    self.intersections.append(inversed_sampled_intersection)

            foliated_intersection.sample_done()
            print(
                "sampled "
                + str(self.intersections.__len__())
                + " intersections bewteen foliations ",
                foliated_intersection.foliation1.foliation_name,
                " and ",
                foliated_intersection.foliation2.foliation_name,
            )

    def set_start_manifold_candidates(self, start_manifold_candidates):
        """Set start manifold candidates"""
        if not isinstance(start_manifold_candidates, list):
            raise Exception("start_manifold_candidates is not a list")
        if start_manifold_candidates.__len__() == 0:
            raise Exception("start_manifold_candidates is empty")
        # check if each element in start_manifold_candidates is a tuple with two integers
        for manifold in start_manifold_candidates:
            # check if manifold is a tuple with two integers
            if (
                not isinstance(manifold, tuple)
                or manifold.__len__() != 2
                or not isinstance(manifold[0], int)
                or not isinstance(manifold[1], int)
            ):
                raise Exception(
                    "Each element in start_manifold_candidates should be a tuple with two integers!!"
                )

        self.start_manifold_candidates = start_manifold_candidates

    def set_goal_manifold_candidates(self, goal_manifold_candidates):
        """Set goal manifold candidates"""
        if not isinstance(goal_manifold_candidates, list):
            raise Exception("goal_manifold_candidates is not a list")
        if goal_manifold_candidates.__len__() == 0:
            raise Exception("goal_manifold_candidates is empty")
        # check if each element in goal_manifold_candidates is a tuple with two integers
        for manifold in goal_manifold_candidates:
            # check if manifold is a tuple with two integers
            if (
                not isinstance(manifold, tuple)
                or manifold.__len__() != 2
                or not isinstance(manifold[0], int)
                or not isinstance(manifold[1], int)
            ):
                raise Exception(
                    "Each element in goal_manifold_candidates should be a tuple with two integers!!"
                )

        self.goal_manifold_candidates = goal_manifold_candidates

    def sampleStartAndGoal(self):
        """randomly sample one start and goal manifold candidates"""

        if (
            self.start_manifold_candidates.__len__() == 0
            or self.goal_manifold_candidates.__len__() == 0
        ):
            raise Exception(
                "start or goal manifold candidates of the problem is empty!!!"
            )

        start_manifold_index = np.random.randint(
            0, self.start_manifold_candidates.__len__()
        )
        start_manifold = self.start_manifold_candidates[start_manifold_index]
        goal_manifold_index = np.random.randint(
            0, self.goal_manifold_candidates.__len__()
        )
        goal_manifold = self.goal_manifold_candidates[goal_manifold_index]

        attempt_time = 0

        while (
            start_manifold == goal_manifold
        ):  # we should not sample the same manifold for start and goal
            start_manifold_index = np.random.randint(
                0, self.start_manifold_candidates.__len__()
            )
            start_manifold = self.start_manifold_candidates[start_manifold_index]
            goal_manifold_index = np.random.randint(
                0, self.goal_manifold_candidates.__len__()
            )
            goal_manifold = self.goal_manifold_candidates[goal_manifold_index]
            attempt_time += 1
            if attempt_time > 100:
                raise Exception(
                    "The start and goal manifold candidates are not enough!!!"
                )

        return (start_manifold, goal_manifold)

    def save(self, dir_name):
        """Save the foliated problem"""

        if not self.has_setup:
            raise Exception("The foliated problem has not been setup!!!")

        if os.path.exists(dir_name):
            # delete the directory
            os.system("rm -rf " + dir_name)

        os.makedirs(dir_name)
        problem_data = {
            "problem_name": self.problem_name,
            "foliations": [],
            "intersections": [],
        }

        # create a directory for foliations
        os.makedirs(dir_name + "/foliations")

        for foliation in self.foliations:
            foliation.save(dir_name + "/foliations")
            problem_data["foliations"].append(foliation.foliation_name)

        # create a directory for intersections
        os.makedirs(dir_name + "/intersections")

        for i, intersection in enumerate(self.intersections):
            intersection.save(
                dir_name + "/intersections/intersection_" + str(i) + ".json"
            )
            problem_data["intersections"].append("intersection_" + str(i))

        # save the problem data
        with open(dir_name + "/problem_data.json", "w") as f:
            json.dump(problem_data, f, indent=4)

        # save start and goal manifold candidates
        with open(dir_name + "/start_manifold_candidates.json", "w") as f:
            for manifold in self.start_manifold_candidates:
                f.write(str(manifold) + "\n")

        with open(dir_name + "/goal_manifold_candidates.json", "w") as f:
            for manifold in self.goal_manifold_candidates:
                f.write(str(manifold) + "\n")

    @staticmethod
    def load(foliation_class, intersection_class, dir_name):
        """Load the foliated problem"""
        """
            The loaded foliated problem can't save again.
        """

        # check if foliation_class is a subclass of BaseFoliation
        if not issubclass(foliation_class, BaseFoliation):
            raise Exception("foliation_class is not a subclass of BaseFoliation!!!")

        # check if intersection_class is a subclass of BaseIntersection
        if not issubclass(intersection_class, BaseIntersection):
            raise Exception(
                "intersection_class is not a subclass of BaseIntersection!!!"
            )

        # check if dir_name exists
        if not os.path.exists(dir_name):
            raise Exception("The directory does not exist!!!")

        # check if problem_data.json exists
        if not os.path.exists(dir_name + "/problem_data.json"):
            raise Exception("The problem_data.json does not exist!!!")

        # check if foliations directory exists
        if not os.path.exists(dir_name + "/foliations"):
            raise Exception("The foliations directory does not exist!!!")

        # check if start_manifold_candidates.json exists
        if not os.path.exists(dir_name + "/start_manifold_candidates.json"):
            raise Exception("The start_manifold_candidates.json does not exist!!!")

        # check if goal_manifold_candidates.json exists
        if not os.path.exists(dir_name + "/goal_manifold_candidates.json"):
            raise Exception("The goal_manifold_candidates.json does not exist!!!")

        # load the problem data
        with open(dir_name + "/problem_data.json", "r") as f:
            problem_data = json.load(f)

        loaded_problem = FoliatedProblem(problem_data["problem_name"])

        loaded_problem.foliations = [
            foliation_class.load(dir_name + "/foliations/" + foliation_name + ".json")
            for foliation_name in problem_data["foliations"]
        ]
        loaded_problem.intersections = [
            intersection_class.load(
                dir_name + "/intersections/" + intersection_name + ".json"
            )
            for intersection_name in problem_data["intersections"]
        ]

        # load start and goal manifold candidates
        with open(dir_name + "/start_manifold_candidates.json", "r") as f:
            for line in f:
                loaded_problem.start_manifold_candidates.append(eval(line))

        with open(dir_name + "/goal_manifold_candidates.json", "r") as f:
            for line in f:
                loaded_problem.goal_manifold_candidates.append(eval(line))

        return loaded_problem


class BaseMotionPlanner:
    __metaclass__ = ABCMeta

    @abstractmethod
    def prepare_planner(self):
        # Prepares the planner
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def plan(
        self,
        start_configuration,
        goal_configuration,
        foliation_constraints,
        co_parameter,
        planning_hint,
        use_atlas,
    ):
        # Returns a success flag, a motion plan which can be visualized, and an experience which can be used to update the task planner.
        raise NotImplementedError("Please Implement this method")

    def _plan(
        self,
        start_configuration,
        goal_configuration,
        foliation_constraints,
        co_parameter,
        planning_hint,
        use_atlas,
    ):
        """
        This function must return a success flag, a motion plan, and an experience.
        """
        success_flag, task_motion_result, experience, manifold_constraint = self.plan(
            start_configuration,
            goal_configuration,
            foliation_constraints,
            co_parameter,
            planning_hint,
            use_atlas,
        )
        if not isinstance(success_flag, bool):
            raise Exception(
                "The first return value of plan function is not a boolean value!!!"
            )
        if not isinstance(task_motion_result, BaseTaskMotion):
            raise Exception(
                "The second return value of plan function is not a BaseTaskMotion class!!!"
            )
        return success_flag, task_motion_result, experience, manifold_constraint

    @abstractmethod
    def shutdown_planner(self):
        # Deletes the planner
        raise NotImplementedError("Please Implement this method")


class BaseTaskMotion:
    __metaclass__ = ABCMeta
    """
        This class is used to store the motion plan for each task. Then, the visualizer can use this class to visualize the motion plan.
        For BaseIntersection and motion planner's result, they should provide a function to convert them to this class.
        So, the visualizer can use this class to visualize the motion plan.
    """
    # def __init__(self, motion_plan):
    #     # check it motion plan is a dictionary
    #     if not isinstance(motion_plan, dict):
    #         raise TypeError("motion plan must be a dictionary")
    #     self.motion_plan = motion_plan

    @abstractmethod
    def get(self):
        # user has to implement this function properly based on how they use
        # the visualizer to visualize the motion plan.
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def cost(self):
        # user has to implement this function properly based on how they use
        # define the cost of the motion plan.
        raise NotImplementedError("Please Implement this method")


class BaseVisualizer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def visualize_plan(self, list_of_motion_plan):
        raise NotImplementedError("Please Implement this method")

    def _visualize_plan(self, list_of_motion_plan):
        # check it each element in list_of_motion_plan is a BaseTaskMotion class
        for motion_plan in list_of_motion_plan:
            if not isinstance(motion_plan, BaseTaskMotion):
                raise TypeError(
                    "Each element in list_of_motion_plan must be a BaseTaskMotion class"
                )
        self.visualize_plan(list_of_motion_plan)


class Task:
    def __init__(
        self, manifold_detail_, start_configuration_, goal_configuration_, next_motion_, use_atlas
    ):
        # Constructor
        self.manifold_detail = manifold_detail_
        self.start_configuration = start_configuration_
        self.goal_configuration = goal_configuration_
        self.next_motion = next_motion_  # the robot motion after the task is completed
        self.related_experience = []
        self.use_atlas = use_atlas

    # set which edge of the task graph this task is.
    # so user can use this information to update the task graph.
    def set_task_graph_info(self, task_graph_info_):
        self.task_graph_info = task_graph_info_


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


class BaseTaskPlanner:
    __metaclass__ = ABCMeta

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

    @abstractmethod
    def reset_task_planner(self, hard_reset):
        # reset the task planner
        raise NotImplementedError("Please Implement this method")

    def read_pointcloud(self, pointcloud_):
        print(
            "-- the task planner does not support read point point because it does not use GMM --"
        )
        raise NotImplementedError(
            "Please Implement this method if you need to read point cloud"
        )

    @abstractmethod
    def add_manifold(self, manifold_info_, manifold_id_):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def add_intersection(self, manifold_id1_, manifold_id2_, intersection_detail_):
        """
        add intersection to the manifold
        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
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

    @abstractmethod
    def generate_task_sequence(self):
        """
        generate task sequence
        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def update(self, task_graph_info_, plan_, manifold_constraint_):
        """
        update task planner
        """
        raise NotImplementedError("Please Implement this method")

    def set_similarity_matrix(self, foliation_id_, similarity_matrix_):
        """
        set similarity matrix for a foliation
        """
        self.total_similiarity_table[foliation_id_] = similarity_matrix_


    # def add_edge_distance_for_all_edges(self):
    #     nx.set_edge_attributes(self.task_graph, 0.0, "edge_dist")
    #     for node1, node2 in self.task_graph.edges():
    #         if node1 != "start" and node2 != "goal" and node1 != "goal" and node2 != "start":
    #             dist1, dist2 = node1[2], node2[2]
    #             dist_between_two_distributions = (
    #                 self.get_position_difference_between_distributions(
    #                     self.gmm_.distributions[dist1].mean,
    #                     self.gmm_.distributions[dist2].mean,
    #                 )
    #             )
    #             self.task_graph.edges[node1, node2]["edge_dist"] = dist_between_two_distributions 

    # def setup_dynamic_planner(self):

    #     self.add_edge_distance_for_all_edges()        
    #     nx.set_node_attributes(self.task_graph, np.inf, "dist_to_start")
    #     nx.set_node_attributes(self.task_graph, np.inf, "dist_to_goal")
    #     self.task_graph.nodes["start"]["dist_to_start"] = 0.0
    #     self.task_graph.nodes["goal"]["dist_to_goal"] = 0.0

    #     self.compute_distance_to_start_and_goal()
    #     self.current_graph_distance_radius = (
    #         nx.shortest_path_length(
    #             self.task_graph, "start", "goal", weight="edge_dist"
    #         )
    #         + 1e-8
    #     )
    #     self.expand_current_task_graph(self.current_graph_distance_radius)

    def get_position_difference_between_distributions(self, dist_mean_1, dist_mean_2):
        """
        Get the distance between two distributions. It is simply the norm of them
        """
        return np.linalg.norm(np.array(dist_mean_1) - np.array(dist_mean_2))

    def expand_current_task_graph(self, distance):
        """
        Get the subset of nodes that are within the distance .
        """
        subset_of_nodes_fast = [node for node, dist in self.total_start_goal_distance_per_node if dist <= distance]
        self.current_task_graph = self.task_graph.subgraph(subset_of_nodes_fast)

    def compute_distance_to_start_and_goal(self):
        """ """
        lengths_to_goal = nx.shortest_path_length(
            self.task_graph, target="goal", weight="edge_dist"
        )
        lengths_to_start = nx.shortest_path_length(
            self.task_graph, source="start", weight="edge_dist"
        )

        self.total_start_goal_distance_per_node = []

        for node in self.task_graph.nodes():
            if node in lengths_to_start:
                self.task_graph.nodes[node]["dist_to_start"] = lengths_to_start[node]
            if node in lengths_to_goal:
                self.task_graph.nodes[node]["dist_to_goal"] = lengths_to_goal[node]

            self.total_start_goal_distance_per_node.append(
                (node, self.task_graph.nodes[node]["dist_to_start"] + self.task_graph.nodes[node]["dist_to_goal"])
            )

    #########################################################################################
    # task solution graph is a graph used to save the solution. Later, the task planner can
    # use this graph to check if the solution exists, then return the solution trajectory.
    # def reset_task_solution_graph(self):
    #     self.task_solution_graph = nx.DiGraph()
    #     self.incomming_manifold_intersections = {} # the incomming intersections of each manifold
    #     self.outgoing_manifold_intersections = {} # the outgoing intersections of each manifold
    #     self.new_intersection_id = 0

    # def add_manifold_for_task_solution_graph(self, manifold_id_):
    #     self.incomming_manifold_intersections[manifold_id_] = []
    #     self.outgoing_manifold_intersections[manifold_id_] = []

    # def add_intersection_for_task_solution_graph(self, manifold_id1_, manifold_id2_):
    #     intersection_from_1_to_2_id = self.new_intersection_id
    #     self.new_intersection_id += 1

    #     self.task_solution_graph.add_node(
    #         intersection_from_1_to_2_id,
    #         previous_manifold_id=manifold_id1_,
    #         next_manifold_id=manifold_id2_)

    #     for i in self.incomming_manifold_intersections[manifold_id1_]:
    #         self.task_solution_graph.add_edge(
    #             i,
    #             intersection_from_1_to_2_id,
    #             manifold_id=manifold_id1_,
    #             has_solution=False,
    #             solution_trajectory=None)

    #     for i in self.outgoing_manifold_intersections[manifold_id2_]:
    #         self.task_solution_graph.add_edge(
    #             intersection_from_1_to_2_id,
    #             i,
    #             manifold_id=manifold_id2_,
    #             has_solution=False,
    #             solution_trajectory=None)

    #     self.outgoing_manifold_intersections[manifold_id1_].append(intersection_from_1_to_2_id)
    #     self.incomming_manifold_intersections[manifold_id2_].append(intersection_from_1_to_2_id)

    #     return intersection_from_1_to_2_id

    # def set_start_and_goal_for_task_solution_graph(self, start_manifold_id_, goal_manifold_id_):
    #     if self.task_solution_graph.has_node('start'):
    #         self.task_solution_graph.remove_node('start')

    #     if self.task_solution_graph.has_node('goal'):
    #         self.task_solution_graph.remove_node('goal')

    #     self.task_solution_graph.add_node(
    #         'start',
    #         previous_manifold_id = None,
    #         next_manifold_id = start_manifold_id_
    #     )

    #     self.task_solution_graph.add_node(
    #         'goal',
    #         previous_manifold_id = goal_manifold_id_,
    #         next_manifold_id = None
    #     )

    #     for i in self.outgoing_manifold_intersections[start_manifold_id_]:
    #         self.task_solution_graph.add_edge(
    #             'start',
    #             i,
    #             manifold_id=start_manifold_id_,
    #             has_solution=False,
    #             solution_trajectory=None)

    #     for i in self.incomming_manifold_intersections[goal_manifold_id_]:
    #         self.task_solution_graph.add_edge(
    #             i,
    #             'goal',
    #             manifold_id=goal_manifold_id_,
    #             has_solution=False,
    #             solution_trajectory=None)

    # def check_solution_existence(self, intersection_id1_, intersection_id2_):
    #     '''
    #     check if the solution exists between two intersections. If the solution exists,
    #     then return the solution trajectory. Otherwise, return None.
    #     '''
    #     if self.task_solution_graph.edges[intersection_id1_, intersection_id2_]['has_solution']:
    #         return self.task_solution_graph.edges[intersection_id1_, intersection_id2_]['solution_trajectory']
    #     else:
    #         return None

    # def save_solution_to_task_solution_graph(self, intersection_id1_, intersection_id2_, solution_trajectory_):
    #     self.task_solution_graph.edges[intersection_id1_, intersection_id2_]['has_solution'] = True
    #     self.task_solution_graph.edges[intersection_id1_, intersection_id2_]['solution_trajectory'] = solution_trajectory_

    # def get_manifold_id_from_task_solution_graph(self, intersection_id1_, intersection_id2_):
    #     return self.task_solution_graph.edges[intersection_id1_, intersection_id2_]['manifold_id']


class FoliationConfig:
    def __init__(self, foliation_set, foliated_intersection_set):
        for foliation in foliation_set:
            # check if foliation contains key 'name', 'co-parameter-type', co-parameter-set', and 'similarity matrix'
            if (
                "name" not in foliation
                or "co-parameter-type" not in foliation
                or "co-parameter-set" not in foliation
                or "similarity-matrix" not in foliation
            ):
                raise Exception(
                    "Each foliation in foliation_set should contain key 'name', 'co-parameter-type', 'co-parameter-set', and 'similarity matrix'"
                )

        for intersection in foliated_intersection_set:
            # check if intersection contains key 'name', 'foliation1', and 'foliation2'
            if (
                "name" not in intersection
                or "foliation1" not in intersection
                or "foliation2" not in intersection
            ):
                raise Exception(
                    "Each intersection in foliated_intersection_set should contain key 'name', 'foliation1', and 'foliation2'"
                )
        self.foliation_set = foliation_set
        self.foliated_intersection_set = foliated_intersection_set
