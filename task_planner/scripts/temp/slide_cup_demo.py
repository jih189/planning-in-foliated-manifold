#!/usr/bin/env python
from foliated_base_class import FoliatedProblem, FoliatedIntersection, FoliationConfig
from manipulation_foliations_and_intersections import (
    ManipulationFoliation,
    ManipulationIntersection,
)
from foliated_planning_framework import FoliatedPlanningFramework
from jiaming_GMM import GMM
from jiaming_task_planner import (
    MTGTaskPlanner,
    # MTGTaskPlannerWithGMM,
    # MTGTaskPlannerWithAtlas,
)
from jiaming_motion_planner import MoveitMotionPlanner
from jiaming_visualizer import MoveitVisualizer

import os
import sys
import rospy
import rospkg
import moveit_commander
from moveit_msgs.msg import Constraints, OrientationConstraint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Quaternion, Point, Pose, PoseStamped, Point32
import numpy as np
from jiaming_lead_planner import LeadPlanner
import random
from jiaming_helper import generate_similarity_matrix, FETCH_GRIPPER_ROTATION, INIT_JOINT_NAMES, INIT_JOINT_POSITIONS

def get_position_difference_between_poses(pose_1_, pose_2_):
    """
    Get the position difference between two poses.
    pose_1_ and pose_2_ are both 4x4 numpy matrices.
    """
    return np.linalg.norm(pose_1_[:3, 3] - pose_2_[:3, 3])

if __name__ == "__main__":
    # Get the path of the desired package
    package_path = rospkg.RosPack().get_path("task_planner")

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("slide_cup_on_table", anonymous=True)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    scene.clear()

    move_group = moveit_commander.MoveGroupCommander("arm")

    # set initial joint state
    joint_state_publisher = rospy.Publisher("/move_group/fake_controller_joint_states", JointState, queue_size=1)

    # Create a JointState message
    joint_state = JointState()
    joint_state.header.stamp = rospy.Time.now()
    joint_state.name = INIT_JOINT_NAMES
    joint_state.position = INIT_JOINT_POSITIONS

    rate = rospy.Rate(10)
    while (joint_state_publisher.get_num_connections() < 1):
        rate.sleep()
    joint_state_publisher.publish(joint_state)

    ########################################## create a foliated problem ##########################################

    table_top_pose = np.array([[1, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 0.78], [0, 0, 0, 1]])
    env_pose = PoseStamped()
    env_pose.header.frame_id = "base_link"
    env_pose.pose.position.x = 0.51
    env_pose.pose.position.y = 0.05
    env_pose.pose.position.z = -0.02
    env_pose.pose.orientation.x = 0
    env_pose.pose.orientation.y = 0
    env_pose.pose.orientation.z = 0.707
    env_pose.pose.orientation.w = 0.707
    env_mesh_path = package_path + "/mesh_dir/desk.stl"
    manipulated_object_mesh_path = package_path + "/mesh_dir/cup.stl"

    # define object's start and goal placement
    start_placement = np.array([[1, 0, 0, 0.75], [0, 1, 0, -0.55], [0, 0, 1, 0.78], [0, 0, 0, 1]])
    goal_placement = np.array([[1, 0, 0, 0.75], [0, 1, 0, -0.15], [0, 0, 1, 0.78], [0, 0, 0, 1]])

    # find all feasible grasps as the co-parameter for sliding foliation
    loaded_array = np.load(package_path + "/mesh_dir/cup.npz")
    feasible_grasps = [np.dot(loaded_array[loaded_array.files[ind]], FETCH_GRIPPER_ROTATION) for ind in random.sample(list(range(len(loaded_array.files))), 10)]
    sliding_similarity_matrix = generate_similarity_matrix(feasible_grasps, get_position_difference_between_poses)

    foliation_approach_object = {
        "name": "approach_object",
        "co-parameter-type": "placement",
        "co-parameter-set": [start_placement],
        "similarity-matrix": np.identity(1)
    }

    foliation_slide_object = {
        "name": "slide_object",
        "co-parameter-type": "grasp",
        "co-parameter-set": feasible_grasps,
        "similarity-matrix": np.identity(len(feasible_grasps))
    }

    foliation_reset_robot = {
        "name": "reset_robot",
        "co-parameter-type": "placement",
        "co-parameter-set": [goal_placement],
        "similarity-matrix": np.identity(1)
    }

    intersection_approach_object_slide_object = {
        "name": "approach_object_slide_object",
        "foliation1": foliation_approach_object,
        "foliation2": foliation_slide_object
    }

    intersection_pour_object_reset_robot = {
        "name": "pour_object_reset_robot",
        "foliation1": foliation_slide_object, 
        "foliation2": foliation_reset_robot
    }

    foliation_problem = FoliationConfig(
        [foliation_approach_object, foliation_slide_object, foliation_reset_robot],
        [intersection_approach_object_slide_object, intersection_pour_object_reset_robot]
    )
    # ####################################################################################################################

    # initialize the planning framework.
    # foliated_planning_framework = FoliatedPlanningFramework()

    # # initialize the motion planner
    # motion_planner = MoveitMotionPlanner()

    # # initialize the task planner
    # task_planner = MTGTaskPlanner()

    # # set the foliated problem
    # foliated_planning_framework.setFoliatedProblem(foliation_problem)

    # # set start and goal
    # foliated_planning_framework.setStartAndGoal(
    #     "approach_object",
    #     0,
    #     [-1.28, 1.51, 0.35, 1.81, 0.0, 1.47, 0.0], # start arm configuration
    #     "reset_robot",
    #     0,
    #     [-1.28, 1.51, 0.35, 1.81, 0.0, 1.47, 0.0] # goal arm configuration
    # )

    ##############################################################################################


    # shutdown the moveit
    moveit_commander.roscpp_shutdown()
    moveit_commander.os._exit(0)
