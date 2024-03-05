import time


class FoliatedPlanningFramework:
    """
    This class implements the foliated planning framework. In this class, the framework will call both task planner
    and motion planner to solve the problem with foliated structure.
    """

    # def __init__(self, task_planner, motion_planner):
    #     self.task_planner = task_planner
    #     self.motion_planner = motion_planner
    #     self.max_attempt_time = 10
    #     self.has_visualizer = False

    def __init__(self):
        self.max_attempt_time = 10
        self.has_visualizer = False
        self.has_task_planner = False
        self.has_motion_planner = False
        self.has_new_foliated_problem = False
        self.has_new_task_planner = False

    def setFoliatedProblem(self, foliated_problem):
        """
        This function sets the foliated problem to the planning framework.
        """
        self.foliated_problem = foliated_problem
        self.has_new_foliated_problem = True

    def setMotionPlanner(self, motion_planner):
        """
        This function sets the motion planner to the planning framework.
        """
        self.motion_planner = motion_planner
        self.motion_planner.prepare_planner()
        self.has_motion_planner = True

    def setTaskPlanner(self, task_planner):
        """
        This function sets the task planner to the planning framework.
        """
        self.task_planner = task_planner
        self.has_task_planner = True
        self.has_new_task_planner = True

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
        start_foliation_index,
        start_co_parameter_index,
        start_configuration,
        goal_foliation_index,
        goal_co_parameter_index,
        goal_configuration,
    ):
        """
        This function sets the start and goal configuration to the planning framework.
        Both start and goal manifolds should not be the same.
        """
        if (
            start_foliation_index == goal_foliation_index
            and start_co_parameter_index == goal_co_parameter_index
        ):
            raise Exception("Start and goal manifolds should not be the same.")
        self.start_foliation_index = start_foliation_index
        self.start_co_parameter_index = start_co_parameter_index
        self.goal_foliation_index = goal_foliation_index
        self.goal_co_parameter_index = goal_co_parameter_index
        self.start_configuration = start_configuration
        self.goal_configuration = goal_configuration

    def evaluation(self):
        """
        This function evaluates the planning framework. This function is exactly the same as the solve function, but it will return a success flag and the planning time
        of task node sequence genertation, motion planning, and updating.
        """
        if not self.has_task_planner:
            raise Exception("No task planner is set to the planning framework.")

        if not self.has_motion_planner:
            raise Exception("No motion planner is set to the planning framework.")

        if self.has_new_foliated_problem or self.has_new_task_planner:
            # if the problem is new, we need to reset the task planner and load the foliated problem.

            # reset the task planner. This reset requires the task planner to load problem again.
            self.task_planner.reset_task_planner(hard_reset=True)

            # load the foliated problem
            self.task_planner.load_foliated_problem(self.foliated_problem)

            self.has_new_foliated_problem = False
            self.has_new_task_planner = False
        else:
            # reset the task planner. This reset does not require the task planner to load problem again. It just reset
            # the task planner to the initial state.
            self.task_planner.reset_task_planner(hard_reset=False)

        # set the start and goal
        t1 = time.time()
        self.task_planner.set_start_and_goal(
            (self.start_foliation_index, self.start_co_parameter_index),
            self.start_configuration,
            (self.goal_foliation_index, self.goal_co_parameter_index),
            self.goal_configuration,
        )
        set_start_and_goal_time = time.time() - t1

        total_solve_time_start = time.time()
        task_node_sequence_generation_time = 0
        motion_planning_time = 0
        updating_time = 0

        for attempt_time in range(self.max_attempt_time):
            # generate the task sequence
            task_node_sequence_generation_time_start = time.time()
            task_sequence = self.task_planner.generate_task_sequence()
            task_node_sequence_generation_time += (
                time.time() - task_node_sequence_generation_time_start
            )

            if len(task_sequence) == 0:
                return False, None, None, None, None

            list_of_motion_plan = []
            found_solution = True

            # solve the problem
            for task_index, task in enumerate(task_sequence):
                # plan the motion
                motion_planning_time_start = time.time()
                (
                    success_flag,
                    motion_plan_result,
                    experience,
                    manifold_constraint,
                ) = self.motion_planner._plan(
                    task.start_configuration,
                    task.goal_configuration,
                    task.manifold_detail.foliation.constraint_parameters,
                    task.manifold_detail.foliation.co_parameters[
                        task.manifold_detail.co_parameter_index
                    ],
                    task.related_experience,
                    task.use_atlas,
                )
                motion_planning_time += time.time() - motion_planning_time_start

                updating_time_start = time.time()
                self.task_planner.update(
                    task.task_graph_info, experience, manifold_constraint
                )
                updating_time += time.time() - updating_time_start

                if not success_flag:
                    found_solution = False
                    break
                else:
                    list_of_motion_plan.append(motion_plan_result)
                    # add the intersection action to the list of motion plan
                    list_of_motion_plan.append(task.next_motion.get_task_motion())

            if not found_solution:
                continue
            else:
                # get the length of the motion plan
                path_length = sum(
                    [motion_plan.cost() for motion_plan in list_of_motion_plan]
                )
                return (
                    True,
                    task_node_sequence_generation_time,
                    motion_planning_time,
                    updating_time,
                    path_length,
                    attempt_time + 1,
                    time.time() - total_solve_time_start,
                    set_start_and_goal_time,
                )

        return False, None, None, None, None, None, None, None

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
        self.task_planner.reset_task_planner(hard_reset=True)

        # load the foliated problem
        self.task_planner.load_foliated_problem(self.foliated_problem)

        # set the start and goal
        self.task_planner.set_start_and_goal(
            (self.start_foliation_index, self.start_co_parameter_index),
            self.start_configuration,
            (self.goal_foliation_index, self.goal_co_parameter_index),
            self.goal_configuration,
        )

        for attempt_time in range(self.max_attempt_time):
            print("attempt time: ", attempt_time)

            # generate the task sequence
            task_sequence = self.task_planner.generate_task_sequence()

            # print detail of generetated task_sequence
            # for t, task in enumerate(task_sequence):
            #     print "task ", t, "-----------------------"
            #     print "foliation name: ", task.manifold_detail.foliation.foliation_name
            # print "start configuration"
            # print(task.start_configuration)
            # print "goal configuration"
            # print(task.goal_configuration)

            if len(task_sequence) == 0:
                return False, None

            list_of_motion_plan = []
            found_solution = True

            # solve the problem
            for task_index, task in enumerate(task_sequence):
                # plan the motion
                (
                    success_flag,
                    motion_plan_result,
                    experience,
                    manifold_constraint,
                ) = self.motion_planner._plan(
                    task.start_configuration,
                    task.goal_configuration,
                    task.manifold_detail.foliation.constraint_parameters,
                    task.manifold_detail.foliation.co_parameters[
                        task.manifold_detail.co_parameter_index
                    ],
                    task.related_experience,
                    task.use_atlas,
                )

                # the following code is for debugging with visualizer.
                if self.has_visualizer:
                    sampled_data = [
                        (sampled_data.sampled_state, sampled_data.sampled_state_tag)
                        for sampled_data in experience[4].verified_motions
                    ]

                    # visualize the sampled data
                    self.visualizer.visualize_for_debug(
                        sampled_data,
                        task_constraint_parameters=task.manifold_detail.foliation.constraint_parameters,
                        start_configuration=task.start_configuration,
                        goal_configuration=task.goal_configuration,
                        action_name=task.manifold_detail.foliation.foliation_name,
                        co_parameter=task.manifold_detail.foliation.co_parameters[
                            task.manifold_detail.co_parameter_index
                        ],
                    )

                self.task_planner.update(
                    task.task_graph_info, experience, manifold_constraint
                )

                print "Task Progress: {}/{}".format(task_index + 1, len(task_sequence)),
                print "Success: " + ("True" if success_flag else "False")

                if success_flag:
                    list_of_motion_plan.append(motion_plan_result)
                    # add the intersection action to the list of motion plan
                    list_of_motion_plan.append(task.next_motion.get_task_motion())
                else:
                    found_solution = False
                    break

            if not found_solution:
                continue
            else:
                return True, list_of_motion_plan

        return False, None

    def visualizeSolutionTrajectory(self, list_of_motion_plan):
        """
        This function visualizes the solution path.
        """
        if self.has_visualizer:
            self.visualizer._visualize_plan(list_of_motion_plan)
        else:
            raise Exception("No visualizer is set to the planning framework.")

    def shutdown(self):
        """
        This function shuts down the planning framework.
        """
        # self.task_planner.shutdown_task_planner()
        self.motion_planner.shutdown_planner()
