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

# from manipulation_test.srv import *
import random
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
import json
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


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
    rospy.init_node("create_experiment_node", anonymous=True)

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

    ###############################################################################################################
    # need to visualize all feasbile placement and the obstacle as markerarray in the rviz

    problem_publisher = rospy.Publisher(
        "/problem_visualization_marker_array", MarkerArray, queue_size=5
    )

    ########################################## create a foliated problem ##########################################

    # For the maze problem, we have two foliations:
    # 1. The foliation for sliding.
    # 2. The foliation for re-grasping.
    # find all co-parameter for both foliations

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

    # env_mesh_path = package_path + "/mesh_dir/maze.stl"
    env_mesh_path = package_path + "/mesh_dir/desk.stl"
    manipulated_object_mesh_path = package_path + "/mesh_dir/cup.stl"

    env_mesh = trimesh.load_mesh(env_mesh_path)
    env_mesh.apply_transform(convert_pose_stamped_to_matrix(env_pose))

    collision_manager = trimesh.collision.CollisionManager()
    collision_manager.add_object("env", env_mesh)

    # find all feasible placements as the co-parameter for re-grasping foliation
    # num_of_row = 5
    # num_of_col = 9
    # x_shift = 0.6
    # y_shift = 0.09
    # z_shift = 0.78
    # feasible_placements = []
    # for i in range(num_of_row):
    #     for j in range(num_of_col):
    #         obj_mesh = trimesh.load_mesh(manipulated_object_mesh_path)

    #         obj_pose = PoseStamped()
    #         obj_pose.header.frame_id = "base_link"
    #         obj_pose.pose.position.x = i * 0.1 - num_of_row * 0.1 / 2 + x_shift
    #         obj_pose.pose.position.y = j * 0.1 - num_of_col * 0.1 / 2 + y_shift
    #         obj_pose.pose.position.z = z_shift
    #         obj_pose.pose.orientation.x = 0
    #         obj_pose.pose.orientation.y = 0
    #         obj_pose.pose.orientation.z = 0
    #         obj_pose.pose.orientation.w = 1

    #         obj_mesh.apply_transform(convert_pose_stamped_to_matrix(obj_pose))

    #         collision_manager.add_object("obj", obj_mesh)

    #         if not collision_manager.in_collision_internal():
    #             feasible_placements.append(convert_pose_stamped_to_matrix(obj_pose))

    #         collision_manager.remove_object("obj")
    feasible_placements = []

    start_obj_pose = PoseStamped()
    start_obj_pose.header.frame_id = "base_link"
    start_obj_pose.pose.position.x = 0.65
    start_obj_pose.pose.position.y = 0.0
    start_obj_pose.pose.position.z = 0.78
    start_obj_pose.pose.orientation.x = 0
    start_obj_pose.pose.orientation.y = 0
    start_obj_pose.pose.orientation.z = 0
    start_obj_pose.pose.orientation.w = 1
    feasible_placements.append(convert_pose_stamped_to_matrix(start_obj_pose))

    goal_obj_pose = PoseStamped()
    goal_obj_pose.header.frame_id = "base_link"
    goal_obj_pose.pose.position.x = 0.45
    goal_obj_pose.pose.position.y = 0.0
    goal_obj_pose.pose.position.z = 0.78
    goal_obj_pose.pose.orientation.x = 0
    goal_obj_pose.pose.orientation.y = 0
    goal_obj_pose.pose.orientation.z = 0
    goal_obj_pose.pose.orientation.w = 1
    feasible_placements.append(convert_pose_stamped_to_matrix(goal_obj_pose))

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

    ####################################################################################################################
    # generate the similarity matrix for both foliations
    def get_position_difference_between_poses(pose_1_, pose_2_):
        """
        Get the position difference between two poses.
        pose_1_ and pose_2_ are both 4x4 numpy matrices.
        """
        return np.linalg.norm(pose_1_[:3, 3] - pose_2_[:3, 3])

    def gaussian_similarity(distance, max_distance, sigma=0.01):
        """
        Calculate the similarity score using Gaussian function.
        distance: the distance between two configurations
        sigma: the sigma of the Gaussian function
        max_distance: the maximum distance between two configurations
        The score is between 0 and 1. The larger the score, the more similar the two configurations are.
        If sigma is heigher, the scope of the Gaussian function is wider.
        """
        if distance == 0:  # when the distance is 0, the score should be 1
            return 1.0

        # Calculate the similarity score using Gaussian function
        score = np.exp(-(distance**2) / (2 * sigma**2))
        max_score = np.exp(-(max_distance**2) / (2 * sigma**2))
        score = (score - max_score) / (1 - max_score)

        if score < 0.001:
            score = 0.0

        return score

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

    ####################################################################################################################

    # build the foliations for both re-grasping and sliding
    foliation_regrasp = ManipulationFoliation(
        foliation_name="regrasp",
        constraint_parameters={
            "frame_id": "base_link",
            "is_object_in_hand": False,
            "object_mesh_path": manipulated_object_mesh_path,
            "obstacle_mesh": env_mesh_path,
            "obstacle_pose": convert_pose_stamped_to_matrix(env_pose),
        },
        co_parameters=feasible_placements,
        similarity_matrix=np.identity(feasible_placements.__len__()),
    )

    print("Number of feasible placements: ", feasible_placements.__len__())

    foliation_slide = ManipulationFoliation(
        foliation_name="slide",
        constraint_parameters={
            "frame_id": "base_link",
            "is_object_in_hand": True,
            "object_mesh_path": manipulated_object_mesh_path,
            "obstacle_mesh": env_mesh_path,
            "obstacle_pose": convert_pose_stamped_to_matrix(env_pose),
            "reference_pose": table_top_pose,
            "orientation_tolerance": np.array([0.05, 0.05, 0.05]),
            "position_tolerance": np.array([2000, 0.0008, 0.0008]),
        },
        co_parameters=feasible_grasps,
        similarity_matrix=sliding_similarity_matrix,
    )

    print("Number of feasible grasps: ", feasible_grasps.__len__())

    # the function to prepare the sampler
    def prepare_sampling_function():
        scene.add_mesh("env_obstacle", env_pose, env_mesh_path)

    # the function to clear the sampler
    def sampling_done_function():
        scene.clear()

    # define the sampling function
    def slide_regrasp_sampling_function(co_parameters1, co_parameters2):
        # co_parameters1 is the co-parameters for sliding foliation
        # co_parameters2 is the co-parameters for regrasping foliation
        # return a ManipulationIntersection class

        # randomly select a index for both co_parameters1 and co_parameters2
        selected_co_parameters1_index = random.randint(0, len(co_parameters1) - 1)
        selected_co_parameters2_index = random.randint(0, len(co_parameters2) - 1)

        # randomly sample a placement
        placement = co_parameters2[selected_co_parameters2_index]

        # randomly sample a grasp
        grasp = co_parameters1[selected_co_parameters1_index]

        # need to calculate the grasp pose in the base_link frame
        grasp_pose_mat = np.dot(placement, grasp)
        pre_grasp_pose_mat = np.dot(
            grasp_pose_mat,
            np.array([[1, 0, 0, -0.05], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        )

        # set the ik target pose
        ik_target_pose = PoseStamped()
        ik_target_pose.header.stamp = rospy.Time.now()
        ik_target_pose.header.frame_id = "base_link"
        ik_target_pose.pose = msgify(geometry_msgs.msg.Pose, grasp_pose_mat)

        ik_req = GetPositionIKRequest()
        ik_req.ik_request.group_name = "arm"
        ik_req.ik_request.avoid_collisions = True
        ik_req.ik_request.pose_stamped = ik_target_pose

        # set the robot state randomly
        random_moveit_robot_state = robot.get_current_state()
        random_position_list = list(random_moveit_robot_state.joint_state.position)
        for joint_name, joint_value in zip(
            move_group.get_joints(), move_group.get_random_joint_values()
        ):
            random_position_list[
                random_moveit_robot_state.joint_state.name.index(joint_name)
            ] = joint_value
        random_moveit_robot_state.joint_state.position = tuple(random_position_list)
        ik_req.ik_request.robot_state = random_moveit_robot_state

        ik_res = compute_ik_srv(ik_req)

        if not ik_res.error_code.val == MoveItErrorCodes.SUCCESS:
            return (
                False,
                selected_co_parameters1_index,
                selected_co_parameters2_index,
                None,
            )

        # need to check the motion from grasp to pre-grasp
        moveit_robot_state = robot.get_current_state()
        moveit_robot_state.joint_state.position = ik_res.solution.joint_state.position

        move_group.set_start_state(moveit_robot_state)
        (planned_motion, fraction) = move_group.compute_cartesian_path(
            [msgify(geometry_msgs.msg.Pose, pre_grasp_pose_mat)], 0.01, 0.0
        )

        if fraction < 0.97:
            return (
                False,
                selected_co_parameters1_index,
                selected_co_parameters2_index,
                None,
            )

        intersection_motion = np.array(
            [p.positions for p in planned_motion.joint_trajectory.points]
        )

        return (
            True,
            selected_co_parameters1_index,
            selected_co_parameters2_index,
            ManipulationIntersection(
                "release",
                intersection_motion,
                move_group.get_active_joints(),
                placement,
                manipulated_object_mesh_path,
                convert_pose_stamped_to_matrix(env_pose),
                env_mesh_path,
            ),
        )

    foliated_intersection = FoliatedIntersection(
        foliation_slide,
        foliation_regrasp,
        slide_regrasp_sampling_function,
        prepare_sampling_function,
        sampling_done_function,
    )

    # foliated_problem = FoliatedProblem("maze_task")
    foliated_problem = FoliatedProblem("desk_task")
    foliated_problem.set_foliation_n_foliated_intersection(
        [foliation_regrasp, foliation_slide], [foliated_intersection]
    )
    foliated_problem.sample_intersections(1000)

    # set the start and goal candidates
    start_candidates = []
    goal_candidates = []
    for p in range(len(feasible_placements)):
        start_candidates.append((0, p))
        goal_candidates.append((0, p))

    foliated_problem.set_start_manifold_candidates(start_candidates)
    foliated_problem.set_goal_manifold_candidates(goal_candidates)

    ###############################################################################################################
    # visualize both intermedaite placements and obstacles
    marker_array = MarkerArray()

    # visualize the obstacle
    obstacle_marker = Marker()
    obstacle_marker.header.frame_id = "base_link"
    obstacle_marker.header.stamp = rospy.Time.now()
    obstacle_marker.ns = "obstacle"
    obstacle_marker.id = 0
    obstacle_marker.type = Marker.MESH_RESOURCE
    obstacle_marker.action = Marker.ADD
    obstacle_marker.pose = env_pose.pose
    obstacle_marker.scale = Point(1, 1, 1)
    obstacle_marker.color = ColorRGBA(0.5, 0.5, 0.5, 1)
    obstacle_marker.mesh_resource = (
        "package://task_planner/mesh_dir/" + os.path.basename(env_mesh_path)
    )
    marker_array.markers.append(obstacle_marker)

    # visualize the placements
    for i, placement in enumerate(feasible_placements):
        object_marker = Marker()
        object_marker.header.frame_id = "base_link"
        object_marker.header.stamp = rospy.Time.now()
        object_marker.ns = "placement"
        object_marker.id = i + 1
        object_marker.type = Marker.MESH_RESOURCE
        object_marker.action = Marker.ADD
        object_marker.pose = msgify(geometry_msgs.msg.Pose, placement)
        object_marker.scale = Point(1, 1, 1)
        object_marker.color = ColorRGBA(0.5, 0.5, 0.5, 1)
        object_marker.mesh_resource = (
            "package://task_planner/mesh_dir/"
            + os.path.basename(manipulated_object_mesh_path)
        )
        marker_array.markers.append(object_marker)

    problem_publisher.publish(marker_array)

    ###############################################################################################################

    # save the foliated problem
    foliated_problem.save(package_path + "/check")

    # load the foliated problem
    loaded_foliated_problem = FoliatedProblem.load(
        ManipulationFoliation, ManipulationIntersection, package_path + "/check"
    )

    # shutdown the moveit
    moveit_commander.roscpp_shutdown()
    moveit_commander.os._exit(0)
