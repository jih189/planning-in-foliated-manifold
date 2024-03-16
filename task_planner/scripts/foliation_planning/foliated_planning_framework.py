import time


class FoliatedPlanningFramework:
    """
    This class implements the foliated planning framework. In this class, the framework will call both task planner
    and motion planner to solve the problem with foliated structure.
    """

    def __init__(self):
        self.max_attempt_time = 1
        self.has_visualizer = False
        self.has_task_planner = False
        self.has_motion_planner = False

    def setFoliatedProblem(self, foliated_problem):
        """
        This function sets the foliated problem to the planning framework.
        """
        self.foliated_problem = foliated_problem
        self.has_new_foliated_problem = True

    def setIntersectionSampler(self, intersection_sampler):
        """
        This function sets the intersection sampler to the planning framework.
        """
        self.intersection_sampler = intersection_sampler
        self.has_intersection_sampler = True

    def setMotionPlanner(self, motion_planner):
        """
        This function sets the motion planner to the planning framework.
        """
        self.motion_planner = motion_planner
        self.has_motion_planner = True

    def setTaskPlanner(self, task_planner):
        """
        This function sets the task planner to the planning framework.
        """
        self.task_planner = task_planner
        self.has_task_planner = True

    def setVisualizer(self, visualizer):
        """
        This function sets the visualizer to the planning framework.
        """
        self.visualizer = visualizer
        self.has_visualizer = True

    def setMaxAttemptTime(self, max_attempt_time=10):
        """
        This function sets the maximum attempt time for the planning framework.
        """
        self.max_attempt_time = max_attempt_time

    def setStartAndGoal(
        self,
        start_foliation_name,
        start_co_parameter_index,
        start_configuration,
        goal_foliation_name,
        goal_co_parameter_index,
        goal_configuration,
    ):
        """
        This function sets the start and goal configuration to the planning framework.
        Both start and goal manifolds should not be the same.
        """
        if (
            start_foliation_name == goal_foliation_name
            and start_co_parameter_index == goal_co_parameter_index
        ):
            raise Exception("Start and goal manifolds should not be the same.")

        self.start_foliation_name = start_foliation_name
        self.start_co_parameter_index = start_co_parameter_index
        self.start_configuration = start_configuration
        self.goal_foliation_name = goal_foliation_name
        self.goal_co_parameter_index = goal_co_parameter_index        
        self.goal_configuration = goal_configuration

    def solve(self):
        """
        This function solves the problem with foliated structure.
        If the solution is found, the framework will return a list of motion plan for each task in sequence.
        That is, the result is a list of motion plan, and the motion planner need to consider it as a list for
        visualization later.
        """

        if not self.has_task_planner:
            raise Exception("No task planner is set to the planning framework.")

        # if not self.has_motion_planner:
        #     raise Exception("No motion planner is set to the planning framework.")

        # reset the task planner
        self.task_planner.reset_task_planner()

        # set the intersection sampler to task planner.
        self.task_planner.set_intersection_sampler(self.intersection_sampler)

        # load the foliated problem
        self.task_planner.load_foliated_problem(self.foliated_problem)

        # # set the start and goal
        self.task_planner.set_start_and_goal(
            self.start_foliation_name, 
            self.start_co_parameter_index,
            self.start_configuration,
            self.goal_foliation_name, 
            self.goal_co_parameter_index,
            self.goal_configuration,
        )

        current_start_configuration = self.start_configuration

        for attempt_time in range(self.max_attempt_time):
            # generate the lead sequence
            lead_sequence = self.task_planner.generate_lead_sequence()

            if len(lead_sequence) == 0:
                return False, None

            first_task_in_lead_sequence = lead_sequence[0]
            
            self.motion_planner.plan(
                current_start_configuration,
                [i.intersection_action[0] for i in first_task_in_lead_sequence.goal_configurations_with_following_action],
                first_task_in_lead_sequence.foliation_constraints,
                first_task_in_lead_sequence.co_parameter,
                first_task_in_lead_sequence.related_experience,
                first_task_in_lead_sequence.use_atlas,
            )

        #     list_of_motion_plan = []
        #     found_solution = True

        #     # solve the problem
        #     for task_index, task in enumerate(task_sequence):
        #         # plan the motion
        #         (
        #             success_flag,
        #             motion_plan_result,
        #             experience,
        #             manifold_constraint,
        #         ) = self.motion_planner._plan(
        #             task.start_configuration,
        #             task.goal_configuration,
        #             task.manifold_detail.foliation.constraint_parameters,
        #             task.manifold_detail.foliation.co_parameters[
        #                 task.manifold_detail.co_parameter_index
        #             ],
        #             task.related_experience,
        #             task.use_atlas,
        #         )

        #         self.task_planner.update(
        #             task.task_graph_info, experience, manifold_constraint
        #         )

        #         if success_flag:
        #             list_of_motion_plan.append(motion_plan_result)
        #             # add the intersection action to the list of motion plan
        #             list_of_motion_plan.append(task.next_motion.get_task_motion())
        #         else:
        #             found_solution = False
        #             break

        #     if not found_solution:
        #         continue
        #     else:
        #         return True, list_of_motion_plan

        # return False, None

    def shutdown(self):
        """
        This function shuts down the planning framework.
        """
        # self.task_planner.shutdown_task_planner()
        self.motion_planner.shutdown_planner()
