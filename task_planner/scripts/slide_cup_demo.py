#!/usr/bin/env python
# from experiment_scripts.experiment_helper import Experiment, Manifold, Intersection
from foliated_base_class import FoliatedProblem, FoliatedIntersection
from manipulation_foliations_and_intersections import (
    ManipulationFoliation,
    ManipulationIntersection,
)

import os
import sys
import rospy
import rospkg
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
from moveit_msgs.srv import GetJointWithConstraints, GetJointWithConstraintsRequest
from moveit_msgs.msg import RobotState
from moveit_msgs.msg import Constraints, OrientationConstraint
from moveit_msgs.msg import MoveItErrorCodes
from sensor_msgs.msg import JointState
import tf.transformations as tf_trans
from ros_numpy import numpify, msgify
from geometry_msgs.msg import Quaternion, Point, Pose, PoseStamped, Point32
import trimesh
from trimesh import transformations
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField, PointCloud
import struct
from jiaming_lead_planner import LeadPlanner

# from manipulation_test.srv import *
import random
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
import json
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from jiaming_helper import gaussian_similarity

from foliated_base_class import FoliatedProblem, FoliatedIntersection, FoliationConfig
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
)
from jiaming_motion_planner import MoveitMotionPlanner
from jiaming_visualizer import MoveitVisualizer


def convert_pose_stamped_to_matrix(pose_stamped):
    pose_matrix = transformations.quaternion_matrix(
        [
            pose_stamped.pose.orientation.w,
            pose_stamped.pose.orientation.x,
            pose_stamped.pose.orientation.y,
            pose_stamped.pose.orientation.z,
        ]
    )
    pose_matrix[0, 3] = pose_stamped.pose.position.x
    pose_matrix[1, 3] = pose_stamped.pose.position.y
    pose_matrix[2, 3] = pose_stamped.pose.position.z
    return pose_matrix


if __name__ == "__main__":
    rospack = rospkg.RosPack()

    # Get the path of the desired package
    package_path = rospack.get_path("task_planner")

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("slide_cup_on_table", anonymous=True)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    # remove all objects from the scene, you need to give some time to the scene to update.
    rospy.sleep(0.5)
    scene.clear()

    move_group = moveit_commander.MoveGroupCommander("arm")
    rospy.wait_for_service("/compute_ik")
    compute_ik_srv = rospy.ServiceProxy("/compute_ik", GetPositionIK)

    # set initial joint state
    joint_state_publisher = rospy.Publisher(
        "/move_group/fake_controller_joint_states", JointState, queue_size=1
    )

    # Create a JointState message
    joint_state = JointState()
    joint_state.header.stamp = rospy.Time.now()
    joint_state.name = [
        "torso_lift_joint",
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "upperarm_roll_joint",
        "elbow_flex_joint",
        "wrist_flex_joint",
        "l_gripper_finger_joint",
        "r_gripper_finger_joint",
    ]
    joint_state.position = [0.38, -1.28, 1.52, 0.35, 1.81, 1.47, 0.04, 0.04]

    rate = rospy.Rate(10)
    while (
        joint_state_publisher.get_num_connections() < 1
    ):  # need to wait until the publisher is ready.
        rate.sleep()
    joint_state_publisher.publish(joint_state)


    ########################################## create a foliated problem ##########################################

    table_top_pose = np.array(
        [[1, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 0.78], [0, 0, 0, 1]]
    )

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

    start_placement = np.array(
        [[1, 0, 0, 0.75], [0, 1, 0, -0.55], [0, 0, 1, 0.78], [0, 0, 0, 1]]
    )

    goal_placement = np.array(
        [[1, 0, 0, 0.75], [0, 1, 0, -0.15], [0, 0, 1, 0.78], [0, 0, 0, 1]]
    )

    # find all feasible grasps as the co-parameter for sliding foliation
    feasible_grasps = []

    loaded_array = np.load(package_path + "/mesh_dir/cup.npz")
    rotated_matrix = np.array(
        [[1, 0, 0, -0.17], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )

    for ind in random.sample(list(range(len(loaded_array.files))), 40):
        feasible_grasps.append(
            np.dot(loaded_array[loaded_array.files[ind]], rotated_matrix)
        )  # add the grasp poses in object frame

    # ####################################################################################################################
    # generate the similarity matrix for both foliations
    def get_position_difference_between_poses(pose_1_, pose_2_):
        """
        Get the position difference between two poses.
        pose_1_ and pose_2_ are both 4x4 numpy matrices.
        """
        return np.linalg.norm(pose_1_[:3, 3] - pose_2_[:3, 3])

    # for sliding
    different_matrix = np.zeros((len(feasible_grasps), len(feasible_grasps)))
    for i, grasp in enumerate(feasible_grasps):
        for j, grasp in enumerate(feasible_grasps):
            if i == j:
                different_matrix[i, j] = 0
            different_matrix[i, j] = get_position_difference_between_poses(
                feasible_grasps[i], feasible_grasps[j]
            )

    sliding_similarity_matrix = np.zeros((len(feasible_grasps), len(feasible_grasps)))
    max_distance = np.max(different_matrix)
    for i, grasp in enumerate(feasible_grasps):
        for j, grasp in enumerate(feasible_grasps):
            sliding_similarity_matrix[i, j] = gaussian_similarity(
                different_matrix[i, j], max_distance, sigma=0.1
            )

    # ####################################################################################################################
    # initilize the foliated problem

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

    # lead_planner = LeadPlanner()
    # lead_planner.load_config(foliation_problem)

    # ####################################################################################################################

    # initialize the planning framework.
    foliated_planning_framework = FoliatedPlanningFramework()

    # initialize the motion planner
    motion_planner = MoveitMotionPlanner()

    # initialize the task planner
    task_planner = MTGTaskPlanner()

    # set the foliated problem
    foliated_planning_framework.setFoliatedProblem(foliation_problem)


    ##############################################################################################


    # shutdown the moveit
    moveit_commander.roscpp_shutdown()
    moveit_commander.os._exit(0)
