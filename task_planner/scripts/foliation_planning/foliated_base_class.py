#!/usr/bin/env python
import os
import json
import numpy as np
from abc import ABCMeta, abstractmethod

# user needs to implement this function
class BaseIntersection:
    __metaclass__ = ABCMeta
    """
        This class represents a base intersection.
        It is an abstract class. It is used to represent the intersection of two manifolds.
        Remember, this intersection can be a point, a motion action.
    """

    def __init__(self, foliation1_name, co_parameter1_index, foliation2_name, co_parameter2_index, intersection_action):
        self.foliation1_name = foliation1_name
        self.co_parameter1_index = co_parameter1_index
        self.foliation2_name = foliation2_name
        self.co_parameter2_index = co_parameter2_index
        self.intersection_action = intersection_action # the action to transit from one manifold to another manifold

    @abstractmethod
    def get_intersection_action(self):
        """Return the intersection action. That is, after the agent reaches the intersection, it will perform this action to switch to the next manifold."""
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def get_edge_configurations(self):
        """
        Return two edge configurations of intersection motion. That is, this two edge configurations 
        are the start and goal configurations of the intersection motion. While start configuration is
        the configuration in the first manifold, and the goal configuration is the configuration in the
        second manifold.
        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def inverse_action(self):
        """Return the inverse of the intersection action, user needs to implement this function"""
        # Return inversed the intersection
        raise NotImplementedError("Please Implement this method")

    def inverse(self):
        return BaseIntersection(
            self.foliation2_name,
            self.co_parameter2_index,
            self.foliation1_name,
            self.co_parameter1_index,
            self.inverse_action(),
        )

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

class IntersectionRule:
    """
    User need to provide a intersection rule function to define how two manifolds are intersected.
    That is, this function will take two foliations and returns a list of pairs of co-parameters from two foliations.
    """
    def __init__(self, intersection_rule_function):
        self.intersection_rule_function = intersection_rule_function

    def find_connected_co_parameters(self, foliation1, foliation2):
        result = self.intersection_rule_function(foliation1, foliation2)
        # check if result is a list of tuples
        if not isinstance(result, list):
            raise Exception("The result of intersection rule function is not a list")
        for pair in result:
            # check if pair is a tuple
            if not isinstance(pair, tuple):
                raise Exception("The element in the result of intersection rule function is not a tuple")
            # check if pair has two elements
            if pair.__len__() != 2:
                raise Exception("The element in the result of intersection rule function does not have two elements")
        return result

class FoliatedIntersection:
    """
        This class represents a foliated intersection.
        It should contain the information about how to sample intersection between two manifolds
        from foliation1 and foliation2. Thus, the intersection sampler can use this information
        to sample intersection between two manifolds.
    """

    def __init__(
        self,
        name,
        foliation1_name,
        foliation2_name,
        intersection_detail
    ):
        # Constructor
        self.name = name
        self.foliation1_name = foliation1_name
        self.foliation2_name = foliation2_name
        self.intersection_detail = intersection_detail # the intersection detail is a dictionary which contains the detail of the intersection
        # that is, it contains the information about how manifolds from foliation1 and foliation2 are intersected. For example, they
        # can be parallel or crossing structure based on user's implementation.

class BaseIntersectionSampler:
    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_final_configuration(self, foliation, co_parameter_index, goal_configuration):
        # sample final configuration from the manifold
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def generate_configurations_on_intersection(self, foliation1, co_parameter_1_index, foliation2, co_parameter_2_index, intersection_detail):
        # sample intersection between two manifolds from foliation1 and foliation2
        raise NotImplementedError("Please Implement this method")

class FoliatedProblem:
    def __init__(self, problem_name, foliation_configuration, intersection_rule):
        """Constructor for FoliatedProblem class"""
        self.problem_name = problem_name  # name of the problem
        
        # check if foliation_configuration is a FoliationConfig class
        if not isinstance(foliation_configuration, FoliationConfig):
            raise Exception("foliation_configuration is not a FoliationConfig class")

        self.foliations = foliation_configuration.get_foliations()  # list of foliations.
        self.foliated_intersections = foliation_configuration.get_foliated_intersections()  # list of foliated intersections.
        self.intersection_rule = intersection_rule  # intersection rule

    def get_foliation_with_name(self, foliation_name):
        """Return the foliation based on the name"""
        for foliation in self.foliations:
            if foliation.foliation_name == foliation_name:
                return foliation
        raise Exception("The foliation does not exist!!!")

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
        goal_configurations,
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
        goal_configurations,
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
            goal_configurations,
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
        self, foliation_name_, co_parameter_index_, goal_configurations_with_following_action_, use_atlas_
    ):
        # Constructor
        self.foliation_name = foliation_name_
        self.co_parameter_index = co_parameter_index_
        self.goal_configurations_with_following_action = goal_configurations_with_following_action_ # a list of configuration with following action
        self.related_experience = []
        self.use_atlas = use_atlas_

class BaseTaskPlanner:
    __metaclass__ = ABCMeta

    def set_intersection_sampler(self, intersection_sampler_):
        self.intersection_sampler = intersection_sampler_
    
    def load_foliated_problem(self, folaited_problem):
        """
        load the foliated problem into the task planner
        """
        self.total_similiarity_table = {}
        self.foliations_set = {}
        self.intersection_rule = folaited_problem.intersection_rule

        for foliation in folaited_problem.foliations:
            # add foliation to the task planner
            self.foliations_set[foliation.foliation_name] = foliation
            # add similaity matrix to the task planner
            self.total_similiarity_table[foliation.foliation_name] = foliation.similarity_matrix
            # process each mainfold defined by co-parameters in the foliation
            for co_parameter_index in range(len(foliation.co_parameters)):
                self.add_manifold(foliation.foliation_name, co_parameter_index)

        for foliated_intersection in folaited_problem.foliated_intersections:
            self.add_foliated_intersection(
                foliated_intersection.foliation1_name,
                foliated_intersection.foliation2_name,
                foliated_intersection.intersection_detail
            )

    @abstractmethod
    def reset_task_planner(self):
        # reset the task planner
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def add_manifold(self, foliation_name, co_parameter_index):
        """
        add manifold to the task planner
        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def add_foliated_intersection(self, foliation1_name, foliation2_name, intersection_detail):
        """
        add intersection rule which defines how two foliations are intersected.
        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def set_start_and_goal(
        self,
        start_foliation_name_,
        start_co_parameter_index_,
        start_configuration_,
        goal_foliation_name_,
        goal_co_parameter_index_,
        goal_configuration_,
    ):
        """
        set start and goal configurations
        both start and goal configurations are intersection here.
        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def generate_lead_sequence(self):
        """
        generate lead sequence
        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def update(self, task_graph_info_, plan_, manifold_constraint_):
        """
        update task planner
        """
        raise NotImplementedError("Please Implement this method")

class FoliationConfig:
    __metaclass__ = ABCMeta
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
                or "intersection_detail" not in intersection
            ):
                raise Exception(
                    "Each intersection in foliated_intersection_set should contain key 'name', 'foliation1', and 'foliation2'"
                )
        
        # based on the user's implementation, load the foliation and foliated intersection
        self.foliation_set = [self.load_foliation(f) for f in foliation_set]
        self.foliated_intersection_set = [self.load_foliated_intersection(i) for i in foliated_intersection_set]

    @abstractmethod
    def load_foliation(self, foliation):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def load_foliated_intersection(self, intersection):
        raise NotImplementedError("Please Implement this method")

    def get_foliations(self):
        return self.foliation_set

    def get_foliated_intersections(self):
        return self.foliated_intersection_set