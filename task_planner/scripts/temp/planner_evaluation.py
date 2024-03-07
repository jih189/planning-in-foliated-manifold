#!/usr/bin/env python
from foliated_base_class import FoliatedProblem, FoliatedIntersection
from manipulation_foliations_and_intersections import (
    ManipulationFoliation,
    ManipulationIntersection,
)
from foliated_planning_framework import FoliatedPlanningFramework
from jiaming_GMM import GMM
from jiaming_task_planner import (
    MTGTaskPlanner,
    MTGTaskPlannerWithGMM,
    MTGTaskPlannerWithAtlas,
    DynamicMTGTaskPlannerWithGMM,
    DynamicMTGPlannerWithAtlas,
)
from jiaming_motion_planner import MoveitMotionPlanner

import rospy
import rospkg
from tqdm import tqdm
import json

"""
On the side, for each foliation problem, we need to provide the possible start and goal manifolds.
The pipeline for the evaluation of the foliated planning framework.

Add a new function to the foliated planning framework:
    evaluation(): this function will solve the problem and return the success flag and planning time of task sequence generation, motion planning, and updating time.

folaition_problem = load the foliated problem.
start_and_goal_list = generate different a list of start and goal for the foliation problem.
for planner in planner_list:
    planning_time = 0
    success_count = 0
    For start, goal in start_and_goal_list:
        foliated_planning_framework.setStartAndGoal(start, goal)
        found_solution, solution_trajectory = foliated_planning_framework.solve()
        if found_solution:
            success_count += 1
            planning_time += foliated_planning_framework.planning_time()

    print "planner: ", planner, " success rate: ", success_count / len(start_and_goal_list), " average planning time: ", planning_time / len(start_and_goal_list)
    save the result to a file.
"""


if __name__ == "__main__":
    number_of_tasks = 50  # number of tasks to be sampled
    max_attempt_time = 50  # maximum attempt time for each task

    ########################################

    rospy.init_node("evaluation_node", anonymous=True)

    task_name = rospy.get_param("~task_name", "")

    rospack = rospkg.RosPack()

    # Get the path of the desired package
    package_path = rospack.get_path("task_planner")

    problem_file_path = package_path + "/" + task_name + "/check"

    # load the foliated problem
    loaded_foliated_problem = FoliatedProblem.load(
        ManipulationFoliation, ManipulationIntersection, problem_file_path
    )

    # set the result file path
    result_file_path = package_path + "/" + task_name + "/result.json"

    print "problem file path: ", problem_file_path
    print "result file path: ", result_file_path

    # sampled random start and goal
    sampled_start_and_goal_list = [
        loaded_foliated_problem.sampleStartAndGoal() for _ in range(number_of_tasks)
    ]

    # load the gmm
    gmm_dir_path = package_path + "/computed_gmms_dir/dpgmm/"
    # gmm_dir_path = package_path + '/computed_gmms_dir/gmm/'
    gmm = GMM()
    gmm.load_distributions(gmm_dir_path)

    # initialize the motion planner
    motion_planner = MoveitMotionPlanner()
    motion_planner.prepare_planner()

    # initialize the foliated planning framework
    foliated_planning_framework = FoliatedPlanningFramework()
    foliated_planning_framework.setMotionPlanner(motion_planner)
    foliated_planning_framework.setMaxAttemptTime(max_attempt_time)
    # set the foliated problem
    foliated_planning_framework.setFoliatedProblem(loaded_foliated_problem)

    # load it into the task planner.
    task_planners = [
        MTGTaskPlanner(),
        MTGTaskPlannerWithGMM(gmm),
        MTGTaskPlannerWithAtlas(gmm, motion_planner.move_group.get_current_state()),
        DynamicMTGTaskPlannerWithGMM(gmm, planner_name_="DynamicMTGTaskPlannerWithGMM_25.0", threshold=25.0),
        DynamicMTGPlannerWithAtlas(gmm, motion_planner.move_group.get_current_state(), planner_name_="DynamicMTGPlannerWithAtlas_25.0", threshold=25.0),
        DynamicMTGTaskPlannerWithGMM(gmm, planner_name_="DynamicMTGTaskPlannerWithGMM_50.0", threshold=50.0),
        DynamicMTGPlannerWithAtlas(gmm, motion_planner.move_group.get_current_state(), planner_name_="DynamicMTGPlannerWithAtlas_50.0", threshold=50.0),
        DynamicMTGTaskPlannerWithGMM(gmm, planner_name_="DynamicMTGTaskPlannerWithGMM_75.0", threshold=75.0),
        DynamicMTGPlannerWithAtlas(gmm, motion_planner.move_group.get_current_state(), planner_name_="DynamicMTGPlannerWithAtlas_75.0", threshold=75.0),
    ]

    with open(result_file_path, "w") as result_file:
        for task_planner in task_planners:
            print("=== Evaluate task planner ", task_planner.planner_name, " ===")

            foliated_planning_framework.setTaskPlanner(task_planner)

            for task_info in tqdm(sampled_start_and_goal_list):
                start, goal = task_info

                # set the start and goal
                foliated_planning_framework.setStartAndGoal(
                    start[0],
                    start[1],
                    ManipulationIntersection(
                        action="start",
                        motion=[[-1.28, 1.51, 0.35, 1.81, 0.0, 1.47, 0.0]],
                        active_joints=motion_planner.move_group.get_active_joints(),
                    ),
                    goal[0],
                    goal[1],
                    ManipulationIntersection(
                        action="goal",
                        motion=[[-1.28, 1.51, 0.35, 1.81, 0.0, 1.47, 0.0]],
                        active_joints=motion_planner.move_group.get_active_joints(),
                    ),
                )

                # solve the problem
                (
                    success_flag,
                    task_planning_time,
                    motion_planning_time,
                    updating_time,
                    solution_length,
                    num_attempts,
                    total_solve_time,
                    set_start_and_goal_time,
                ) = foliated_planning_framework.evaluation()

                if success_flag:
                    result_data = {
                        "planner_name": task_planner.planner_name,
                        "start": start,
                        "goal": goal,
                        "success": "true",
                        "total_planning_time": total_solve_time,
                        "task_planning_time": task_planning_time,
                        "motion_planning_time": motion_planning_time,
                        "set_start_and_goal_time": set_start_and_goal_time,
                        "updating_time": updating_time,
                        "solution_length": solution_length,
                        "num_attempts": num_attempts,
                    }
                    json.dump(result_data, result_file)
                    result_file.write("\n")
                else:
                    result_data = {
                        "planner_name": task_planner.planner_name,
                        "start": start,
                        "goal": goal,
                        "success": "false",
                        "total_planning_time": -1,
                        "task_planning_time": -1,
                        "motion_planning_time": -1,
                        "set_start_and_goal_time": -1,
                        "updating_time": -1,
                        "solution_length": -1,
                        "num_attempts": -1,
                    }
                    json.dump(result_data, result_file)
                    result_file.write("\n")

    # shutdown the planning framework
    foliated_planning_framework.shutdown()
