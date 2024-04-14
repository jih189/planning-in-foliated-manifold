#!/usr/bin/env python
import rospy
import rospkg

import numpy as np
import random
from custom_foliated_class import CustomFoliationConfig, custom_intersection_rule
from foliation_planning.foliated_base_class import FoliatedProblem, IntersectionRule
from foliation_planning.foliated_planning_framework import FoliatedPlanningFramework
from MTG_task_planner import MTGTaskPlanner
from FoliatedRepMap_task_planner import FoliatedRepMapTaskPlanner
from Atlas_FoliatedRepMap_task_planner import AtlasFoliatedRepMapTaskPlanner
from jiaming_motion_planner import MoveitMotionPlanner
from custom_intersection_sampler import CustomIntersectionSampler
from jiaming_helper import generate_similarity_matrix, GRIPPER_ROTATION, INIT_ACTIVE_JOINT_POSITIONS
from custom_visualizer import MoveitVisualizer
from geometry_msgs.msg import Pose
from jiaming_GMM import GMM

import tqdm
import time

def get_position_difference_between_poses(pose_1_, pose_2_):
    """
    Get the position difference between two poses.
    pose_1_ and pose_2_ are both 4x4 numpy matrices.
    """
    return np.linalg.norm(pose_1_[:3, 3] - pose_2_[:3, 3])

def get_path_length(plan):

    plan_length = 0.0
    solution_path = []

    for task_motion in plan:
        if task_motion is None:
            continue
        (
            motion_trajectory,
            has_object_in_hand,
            object_pose,
            object_mesh_path,
            obstacle_pose,
            obstacle_mesh_path,
        ) = task_motion.get()

        for p in motion_trajectory.joint_trajectory.points:
            solution_path.append(p.positions)

    # given a solution path, calculate the path length
    for i in range(len(solution_path) - 1):
        plan_length += np.linalg.norm(np.array(solution_path[i]) - np.array(solution_path[i + 1]))

    return plan_length

if __name__ == "__main__":
    rospy.init_node("demo_node", anonymous=True)
    # Get the path of the desired package
    package_path = rospkg.RosPack().get_path("task_planner")

    obstacle_mesh_path = package_path + "/mesh_dir/desk.stl"
    manipulated_object_mesh_path = package_path + "/mesh_dir/cup.stl"

    obstacle_pose = np.array([[0,-1,0,0.51], [1,0,0,0.05], [0,0,1,-0.02], [0,0,0,1]])
    start_object_pose = np.array([[1,0,0,0.55], [0,1,0,-0.15], [0,0,1,0.78], [0,0,0,1]])
    goal_object_pose = np.array([[1,0,0,0.55], [0,1,0,0.15], [0,0,1,0.78],[0,0,0,1]])
    table_top_pose = np.array([[1, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 0.78], [0, 0, 0, 1]])

    loaded_array = np.load(package_path + "/mesh_dir/cup.npz")
    grasp_set = [np.dot(loaded_array[loaded_array.files[ind]], GRIPPER_ROTATION) for ind in random.sample(list(range(len(loaded_array.files))), 100)]
    foliation_slide_object_similarity_matrix = generate_similarity_matrix(grasp_set, get_position_difference_between_poses)
    grasp_inv_set = [np.linalg.inv(g) for g in grasp_set]

    foliation_approach_object = {
        "name": "approach_object",
        "co-parameter-type": "placement",
        "object_mesh": manipulated_object_mesh_path,
        "co-parameter-set": [
            start_object_pose
        ],
        "similarity-matrix": np.identity(1),
        "obstacle_pose": obstacle_pose, 
        "obstacle_mesh": obstacle_mesh_path
    }

    foliation_slide_object = {
        "name": "slide_object",
        "co-parameter-type": "grasp",
        "object_mesh": manipulated_object_mesh_path,
        "object_constraints": {
            "frame_id": "base_link",
            "reference_pose": table_top_pose,
            "orientation_tolerance": [0.001, 0.001, 6.28],
            "position_tolerance": np.array([2000, 2000, 0.0008]),
        },
        "co-parameter-set": grasp_inv_set,
        "similarity-matrix": foliation_slide_object_similarity_matrix,
        "obstacle_pose": obstacle_pose, 
        "obstacle_mesh": obstacle_mesh_path
    }

    foliation_reset_robot = {
        "name": "reset_robot",
        "co-parameter-type": "placement",
        "object_mesh": manipulated_object_mesh_path,
        "co-parameter-set": [goal_object_pose],
        "similarity-matrix": np.identity(1),
        "obstacle_pose": obstacle_pose, 
        "obstacle_mesh": obstacle_mesh_path
    }

    intersection_approach_object_slide_object = {
        "name": "approach_object_slide_object",
        "foliation1": "approach_object",
        "foliation2": "slide_object", 
        "intersection_detail": {
            "obstacle_pose": obstacle_pose, 
            "obstacle_mesh": obstacle_mesh_path,
            "object_mesh": manipulated_object_mesh_path
        },
        
    }

    intersection_slide_object_reset_robot = {
        "name": "slide_object_reset_robot",
        "foliation1": "slide_object",
        "foliation2": "reset_robot",
        "intersection_detail": {
            "obstacle_pose": obstacle_pose, 
            "obstacle_mesh": obstacle_mesh_path,
            "object_mesh": manipulated_object_mesh_path
        },
    }

    foliation_config = CustomFoliationConfig(
        [
            foliation_approach_object, 
            foliation_slide_object,
            foliation_reset_robot
        ],[
            intersection_approach_object_slide_object, 
            intersection_slide_object_reset_robot
        ]
    )

    foliation_problem = FoliatedProblem(
        "sliding_cup_on_desk", 
        foliation_config, 
        IntersectionRule(custom_intersection_rule)
    )
    
    foliated_planning_framework = FoliatedPlanningFramework()
    foliated_planning_framework.setMaxAttemptTime(20)
    motion_planner = MoveitMotionPlanner()
    #########################################################
    # prepare the task planners
    task_planners = []

    MTG_task_planner = MTGTaskPlanner()
    task_planners.append(MTG_task_planner)

    # load the gmm
    gmm_dir_path = package_path + "/computed_gmms_dir/dpgmm/"
    gmm = GMM()
    gmm.load_distributions(gmm_dir_path)
    FoliatedRepMap_task_planner = FoliatedRepMapTaskPlanner(gmm)
    task_planners.append(FoliatedRepMap_task_planner)

    AtlasFoliatedRepMap_task_planner = AtlasFoliatedRepMapTaskPlanner(gmm,  motion_planner.move_group.get_current_state())
    task_planners.append(AtlasFoliatedRepMap_task_planner)
    #########################################################
    
    intersection_sampler = CustomIntersectionSampler(motion_planner.robot, motion_planner.scene)
    foliated_planning_framework.setIntersectionSampler(intersection_sampler)
    foliated_planning_framework.setMotionPlanner(motion_planner)
    foliated_planning_framework.setFoliatedProblem(foliation_problem)
    foliated_planning_framework.setStartAndGoal(
        "approach_object",
        0,
        INIT_ACTIVE_JOINT_POSITIONS,
        "reset_robot",
        0,
        INIT_ACTIVE_JOINT_POSITIONS,
    )

    total_attempts = 100
    task_planners_name = []
    task_planners_average_planning_time = []
    task_planners_average_planning_length = []
    task_planners_success_rate = []

    for task_planner in task_planners:

        foliated_planning_framework.setTaskPlanner(task_planner)
        total_planning_time = 0.0
        total_planning_length = 0.0
        total_planning_success = 0

        for _ in tqdm.tqdm(range(total_attempts), desc="Evaluating task planner: " + task_planner.planner_name):

            start_time = time.time()
            solution_path = foliated_planning_framework.evaluation()
            planning_time = time.time() - start_time
            if len(solution_path) == 0:
                # failed to find a solution
                pass
            else:
                solution_path_length = get_path_length(solution_path)
                total_planning_time += planning_time
                total_planning_length += solution_path_length
                total_planning_success += 1

        task_planners_name.append(task_planner.planner_name)
        if total_planning_success == 0:
            task_planners_average_planning_time.append(0.0)
            task_planners_average_planning_length.append(0.0)
            task_planners_success_rate.append(0.0)
        else:
            task_planners_average_planning_time.append(total_planning_time / total_planning_success)
            task_planners_average_planning_length.append(total_planning_length / total_planning_success)
            task_planners_success_rate.append(float(total_planning_success) / total_attempts)

    print("Evaluation results:")
    for task_planner_name, task_planner_average_planning_time, task_planner_average_planning_length, task_planner_success_rate in zip(task_planners_name, task_planners_average_planning_time, task_planners_average_planning_length, task_planners_success_rate):
        print("Task planner: ", task_planner_name)
        print("Average planning time: ", task_planner_average_planning_time)
        print("Average planning length: ", task_planner_average_planning_length)
        print("Success rate: ", task_planner_success_rate)
        print("=====================================")
    print ("Done!")