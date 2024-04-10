import time

class FoliatedPlanningFramework:
    """
    This class implements the foliated planning framework. In this class, the framework will call both task planner
    and motion planner to solve the problem with foliated structure.
    """

    def __init__(self):
        self.max_attempt_time = 10
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

        if not self.has_motion_planner:
            raise Exception("No motion planner is set to the planning framework.")

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

        current_foliation_name = self.start_foliation_name
        current_co_parameter_index = self.start_co_parameter_index
        current_start_configuration = self.start_configuration
        current_solution_trajectory = []
        print "========================================================================"
        print "============================ solving problem ==========================="
        print "========================================================================"
        print "-"
        print "-"

        for attempt_time in range(self.max_attempt_time):
            # generate the lead sequence which is a list of task with mode transition.
            print "----------------------------------------------------------"
            print "attempt: ", attempt_time, "/", self.max_attempt_time
            print "current foliation name: ", current_foliation_name
            print "current co parameter index: ", current_co_parameter_index
            print "current start configuration: ", current_start_configuration
            lead_sequence, is_last_task = self.task_planner.generate_lead_sequence(
                    current_start_configuration, 
                    current_foliation_name, 
                    current_co_parameter_index
                )

            if len(lead_sequence) == 0:
                # if the lead sequence is empty, it means that there is no solution.
                return []

            current_task = lead_sequence[0]

            # print all goal configurations
            goal_configurations = [g.get_intersection_motion()[0] for g in current_task.goal_configurations_with_following_action] 
            print "goal configurations"
            for g in goal_configurations:
                print g
            print "is last task: ", is_last_task

            # plan the motion
            (
                success_flag,
                generate_task_motion,
                next_motion,
                experience,
                manifold_constraint,
                last_configuration, # last configuration of the planned motion.
                next_manifold_id
            ) = self.motion_planner.plan(
                current_start_configuration,
                current_task.goal_configurations_with_following_action, #[i.intersection_action[0] for i in current_task.goal_configurations_with_following_action],
                current_task.foliation_constraints,
                current_task.co_parameter,
                current_task.related_experience,
                current_task.use_atlas,
            )

            if is_last_task and success_flag: # this is the last motion planning task.
                # if the solution is found, then we can return the result.
                current_solution_trajectory.append(generate_task_motion)
                return current_solution_trajectory

            current_mode_transitions = set()
            # Due to different intersection may lead to different manifolds. If the planning is failed,
            # then we need to tell the task planner which intersection is for now infeasible.
            # On the other hand, if the planning is successful, then we need to update the task planner
            # certain intersection is feasible.
            if not success_flag:
                # if planning failed, we need to update the task planner based on all those possible intersections.
                for i in current_task.goal_configurations_with_following_action:
                    next_foliation_name, next_co_parameter_index = i.get_next_manifold_id()
                    current_mode_transitions.add((current_foliation_name, current_co_parameter_index, next_foliation_name, next_co_parameter_index))
            else:
                current_mode_transitions.add((current_foliation_name, current_co_parameter_index, next_manifold_id[0], next_manifold_id[1]))

            # update the task planner based on the result from motion planner.
            self.task_planner.update(
                list(current_mode_transitions),
                success_flag,
                experience,
                manifold_constraint,
            )

            if success_flag:
                # if the motion planning is successful, then we can append the result to the current solution trajectory.
                print "next manifold id: ", next_manifold_id

                current_solution_trajectory.append(generate_task_motion)
                current_solution_trajectory.append(next_motion)
                current_foliation_name = next_manifold_id[0]
                current_co_parameter_index = next_manifold_id[1]
                current_start_configuration = last_configuration
            else:
                # if the motion planning is failed, then we need to replan the motion.
                current_solution_trajectory = []
                current_foliation_name = self.start_foliation_name
                current_co_parameter_index = self.start_co_parameter_index
                current_start_configuration = self.start_configuration
        
        return []

    def shutdown(self):
        """
        This function shuts down the planning framework.
        """
        # self.task_planner.shutdown_task_planner()
        self.motion_planner.shutdown_planner()
