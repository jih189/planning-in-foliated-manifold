#!/usr/bin/env python
import rospy
import rospkg

import numpy as np
import random
from custom_foliated_class import CustomFoliationConfig, custom_intersection_rule
from foliation_planning.foliated_base_class import FoliatedProblem, IntersectionRule
from foliation_planning.foliated_planning_framework import FoliatedPlanningFramework
from MTG_task_planner import MTGTaskPlanner
from jiaming_motion_planner import MoveitMotionPlanner
from custom_intersection_sampler import CustomIntersectionSampler
from jiaming_helper import generate_similarity_matrix, FETCH_GRIPPER_ROTATION
from custom_visualizer import MoveitVisualizer
from geometry_msgs.msg import Pose

def get_position_difference_between_poses(pose_1_, pose_2_):
    """
    Get the position difference between two poses.
    pose_1_ and pose_2_ are both 4x4 numpy matrices.
    """
    return np.linalg.norm(pose_1_[:3, 3] - pose_2_[:3, 3])

if __name__ == "__main__":
    rospy.init_node("demo_node", anonymous=True)
    # Get the path of the desired package
    package_path = rospkg.RosPack().get_path("task_planner")

    obstacle_mesh_path = package_path + "/mesh_dir/desk.stl"
    manipulated_object_mesh_path = package_path + "/mesh_dir/cup.stl"

    obstacle_pose = np.array([[0,-1,-0,0.51], [1,0,0,0.05], [0,0,1,-0.02], [0,0,0,1]])
    start_object_pose = np.array([[1,0,0,0.6], [0,1,0,-0.15], [0,0,1,0.78], [0,0,0,1]])
    pre_pouring_object_pose = np.array([[1,0,0,0.64], [0,1,0,-0.15], [0,0,1,0.9],[0,0,0,1]])
    after_pouring_object_pose =  np.array([[1,0,0,0.64], [0,0,1,-0.15], [0,-1,0,0.9],[0,0,0,1]])
    goal_object_pose = np.array([[1,0,0,0.65], [0,1,0,-0.15], [0,0,1,0.78],[0,0,0,1]])
    table_top_pose = np.array([[1, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 0.78], [0, 0, 0, 1]])

    loaded_array = np.load(package_path + "/mesh_dir/cup.npz")
    grasp_set = [np.dot(loaded_array[loaded_array.files[ind]], FETCH_GRIPPER_ROTATION) for ind in random.sample(list(range(len(loaded_array.files))), 100)]
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

    foliation_move_object = {
        "name": "move_object",
        "co-parameter-type": "grasp",
        "object_mesh": manipulated_object_mesh_path,
        "object_constraints": {
            "frame_id": "base_link",
            "reference_pose": table_top_pose,
            "orientation_tolerance": [0.001, 0.001, 6.28],
            "position_tolerance": np.array([2000, 2000, 2000]),
        },
        "co-parameter-set": grasp_inv_set,
        "similarity-matrix": foliation_slide_object_similarity_matrix,
        "obstacle_pose": obstacle_pose, 
        "obstacle_mesh": obstacle_mesh_path
    }

    foliation_pour_object = {
        "name": "pour_object",
        "co-parameter-type": "grasp",
        "object_mesh": manipulated_object_mesh_path,
        "object_constraints": {
            "frame_id": "base_link",
            "reference_pose": pre_pouring_object_pose,
            "orientation_tolerance": [6.28, 6.28, 6.28],
            "position_tolerance": np.array([0.0008, 0.0008, 0.0008]),
        },
        "co-parameter-set": grasp_inv_set,
        "similarity-matrix": foliation_slide_object_similarity_matrix,
        "obstacle_pose": obstacle_pose, 
        "obstacle_mesh": obstacle_mesh_path
    }

    foliation_place_object = {
        "name": "place_object",
        "co-parameter-type": "grasp",
        "object_mesh": manipulated_object_mesh_path,
        "object_constraints": {
            "frame_id": "base_link",
            "reference_pose": table_top_pose,
            "orientation_tolerance": [6.28, 3.14, 3.14],
            "position_tolerance": np.array([2000, 2000, 2000]),
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

    intersection_approach_object_move_object = {
        "name": "approach_object_move_object",
        "foliation1": "approach_object",
        "foliation2": "move_object", 
        "intersection_detail": {
            "obstacle_pose": obstacle_pose, 
            "obstacle_mesh": obstacle_mesh_path,
            "object_mesh": manipulated_object_mesh_path
        },
    }

    intersection_move_object_pour_object = {
        "name": "move_object_pour_object",
        "foliation1": "move_object",
        "foliation2": "pour_object",
        "intersection_detail": {
            "obstacle_pose": obstacle_pose, 
            "obstacle_mesh": obstacle_mesh_path,
            "object_mesh": manipulated_object_mesh_path,
            "object_reference_pose": pre_pouring_object_pose,
            "object_orientation_tolerance": [0.001, 0.001, 0.001],
            "object_position_tolerance": np.array([0.0008, 0.0008, 0.0008])
        },
    }

    intersection_pour_object_place_object = {
        "name": "pour_object_place_object",
        "foliation1": "pour_object",
        "foliation2": "place_object",
        "intersection_detail": {
            "obstacle_pose": obstacle_pose, 
            "obstacle_mesh": obstacle_mesh_path,
            "object_mesh": manipulated_object_mesh_path,
            "object_reference_pose": after_pouring_object_pose,
            "object_orientation_tolerance": [0.001, 0.001, 6.28],
            "object_position_tolerance": np.array([0.0008, 0.0008, 0.0008])
        },
    }

    intersection_place_object_reset_robot = {
        "name": "place_object_reset_robot",
        "foliation1": "place_object",
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
            foliation_move_object,
            foliation_pour_object,
            foliation_place_object,
            foliation_reset_robot
        ],[
            intersection_approach_object_move_object, 
            intersection_move_object_pour_object,
            intersection_pour_object_place_object,
            intersection_place_object_reset_robot
        ]
    )

    foliation_problem = FoliatedProblem(
        "pouring_cup_on_desk", 
        foliation_config, 
        IntersectionRule(custom_intersection_rule)
    )
    
    foliated_planning_framework = FoliatedPlanningFramework()
    foliated_planning_framework.setMaxAttemptTime(40)
    task_planner = MTGTaskPlanner()
    foliated_planning_framework.setTaskPlanner(task_planner)
    motion_planner = MoveitMotionPlanner()
    intersection_sampler = CustomIntersectionSampler(motion_planner.robot, motion_planner.scene)
    foliated_planning_framework.setIntersectionSampler(intersection_sampler)
    foliated_planning_framework.setMotionPlanner(motion_planner)
    foliated_planning_framework.setFoliatedProblem(foliation_problem)

    foliated_planning_framework.setStartAndGoal(
        "approach_object",
        0,
        [-1.28, 1.51, 0.35, 1.81, 0.0, 1.47, 0.0],
        "reset_robot",
        0,
        [-1.28, 1.51, 0.35, 1.81, 0.0, 1.47, 0.0],
    )

    planned_solution = foliated_planning_framework.solve()

    if len(planned_solution) > 0:
        print("Planned solution is found.")
        visualizer = MoveitVisualizer()
        visualizer.prepare_visualizer(motion_planner.active_joints, motion_planner.robot)
        visualizer.visualize_plan(planned_solution)
    else:
        print("No solution is found.")