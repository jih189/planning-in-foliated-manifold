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
from jiaming_motion_planner import MoveitMotionPlanner
from custom_intersection_sampler import CustomIntersectionSampler
from jiaming_helper import generate_similarity_matrix, GRIPPER_ROTATION, INIT_ACTIVE_JOINT_POSITIONS
from custom_visualizer import MoveitVisualizer
from geometry_msgs.msg import Pose
from jiaming_GMM import GMM

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
        "name": "pour_object_reset_robot",
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
    foliated_planning_framework.setMaxAttemptTime(2)
    #########################################################
    task_planner = MTGTaskPlanner()
    
    # # load the gmm
    # gmm_dir_path = package_path + "/computed_gmms_dir/dpgmm/"
    # gmm = GMM()
    # gmm.load_distributions(gmm_dir_path)
    # task_planner = FoliatedRepMapTaskPlanner(gmm)
    #########################################################
    foliated_planning_framework.setTaskPlanner(task_planner)
    motion_planner = MoveitMotionPlanner()
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

    planned_solution = foliated_planning_framework.solve()

    if len(planned_solution) > 0:
        print("Planned solution is found.")
        visualizer = MoveitVisualizer()
        visualizer.prepare_visualizer(motion_planner.active_joints, motion_planner.robot)
        visualizer.visualize_plan(planned_solution)
    else:
        print("No solution is found.")