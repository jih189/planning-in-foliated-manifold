#!/usr/bin/env python
from experiment_helper import Experiment, Manifold, Intersection

import sys
import copy
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

if __name__ == "__main__":
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("create_experiment_node", anonymous=True)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
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

    # load the obstacle
    table_pose = PoseStamped()
    table_pose.header.frame_id = "base_link"
    table_pose.pose.position.x = 0.28
    table_pose.pose.position.y = 0.92
    table_pose.pose.position.z = 0
    table_pose.pose.orientation.x = 0
    table_pose.pose.orientation.y = 0
    table_pose.pose.orientation.z = 0.707
    table_pose.pose.orientation.w = -0.707
    print("Add the table to the planning scene")
    rospack = rospkg.RosPack()
    # Get the path of the desired package
    package_path = rospack.get_path("task_planner")
    scene.add_mesh("table", table_pose, package_path + "/mesh_dir/table.stl")

    # load the cup
    init_cup_pose = PoseStamped()
    init_cup_pose.header.frame_id = "base_link"
    init_cup_pose.pose.position.x = 0.44
    init_cup_pose.pose.position.y = 0.23
    init_cup_pose.pose.position.z = 0.83
    init_cup_pose.pose.orientation.x = 0
    init_cup_pose.pose.orientation.y = 0
    init_cup_pose.pose.orientation.z = 0
    init_cup_pose.pose.orientation.w = 1
    scene.add_mesh("init_cup", init_cup_pose, package_path + "/mesh_dir/cup.stl")

    target_cup_pose = PoseStamped()
    target_cup_pose.header.frame_id = "base_link"
    target_cup_pose.pose.position.x = 0.44
    target_cup_pose.pose.position.y = -0.23
    target_cup_pose.pose.position.z = 0.83
    target_cup_pose.pose.orientation.x = 0
    target_cup_pose.pose.orientation.y = 0
    target_cup_pose.pose.orientation.z = 0
    target_cup_pose.pose.orientation.w = 1
    scene.add_mesh("target_cup", target_cup_pose, package_path + "/mesh_dir/cup.stl")

    grasp_pose_list = []

    loaded_array = np.load(package_path + "/mesh_dir/cup.npz")

    rotated_matrix = np.array(
        [[1, 0, 0, -0.17], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )

    # randomly select 20 elements from the loaded_array.files
    for ind in random.sample(list(range(len(loaded_array.files))), 30):
        array_name = loaded_array.files[ind]
        grasp_pose_list.append(np.dot(loaded_array[array_name], rotated_matrix))

    # rospy.wait_for_service('visualize_regrasp')
    # grasp_visualizer = rospy.ServiceProxy('visualize_regrasp', VisualizeRegrasp)

    display_trajectory_publisher = rospy.Publisher(
        "/move_group/display_planned_path",
        moveit_msgs.msg.DisplayTrajectory,
        queue_size=10,
    )

    grasp_poses_for_vis = []

    ###################################### start to generate the pick-and-place experiment ######################################

    experiment = Experiment()

    experiment.setup(
        "pick_and_place",
        package_path + "/mesh_dir/cup.stl",
        package_path + "/mesh_dir/table.stl",
        numpify(table_pose.pose),
        robot.get_current_state().joint_state.position,
        robot.get_current_state().joint_state.name,
        move_group.get_joints(),
    )

    # add the manifold for pre-grasp
    pre_grasp_manifold = Manifold(
        0,  # foliation id
        0,  # manifold id
        "cup",  # object name
        package_path + "/mesh_dir/cup.stl",  # object mesh file name
        False,
    )  # is the object in hand

    # set the initial object placement pose into pre-grasp manifold
    pre_grasp_manifold.add_object_placement(numpify(init_cup_pose.pose))
    experiment.add_manifold(pre_grasp_manifold)

    post_grasp_manifold = Manifold(
        2,  # foliation id
        0,  # manifold id
        "cup",  # object name
        package_path + "/mesh_dir/cup.stl",  # object mesh file name
        False,
    )  # is the object in hand

    # set the target object placement pose into post-grasp manifold
    post_grasp_manifold.add_object_placement(numpify(target_cup_pose.pose))
    experiment.add_manifold(post_grasp_manifold)

    num_of_grasp_manifolds = 0

    for g in grasp_pose_list:
        # for each grasp pose, we need to check the feasibility of both pick and place
        # That is, we need to seach for the intersection between pre-grasp and grasp manifolds
        # and the intersection between grasp and post-grasp manifolds. If both intersections exist,
        # then we can create the manifold for this grasp with related intersections.

        intersections_from_pre_grasp_to_grasp = []
        intersections_from_grasp_to_post_grasp = []

        ###### need to search intersections from pre-grasp to grasp ######

        # First to compute the ik solution for checking the feasibility
        grasp_pose_mat = np.dot(numpify(init_cup_pose.pose), g)
        pre_grasp_pose_mat = np.dot(
            grasp_pose_mat,
            np.array([[1, 0, 0, -0.09], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
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

        possible_ik_solutions = []

        for _ in range(10):
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

            if not ik_res.error_code.val == 1:
                continue

            # if there is a similar solution in possible ik solution, then skip it
            if len(possible_ik_solutions) > 0:
                if (
                    np.linalg.norm(
                        np.array(ik_res.solution.joint_state.position)
                        - np.array(possible_ik_solutions),
                        axis=1,
                    ).min()
                    < 0.01
                ):
                    continue

            possible_ik_solutions.append(ik_res.solution.joint_state.position)

            # need to check the motion for pre-grasp
            moveit_robot_state = robot.get_current_state()
            moveit_robot_state.joint_state.position = (
                ik_res.solution.joint_state.position
            )

            move_group.set_start_state(moveit_robot_state)
            (approach_plan, fraction) = move_group.compute_cartesian_path(
                [msgify(geometry_msgs.msg.Pose, pre_grasp_pose_mat)], 0.01, 0.0
            )

            if fraction < 0.97:
                continue

            # save the intersection motion from pre-grasp manifold to grasp manifold.
            intersections_from_pre_grasp_to_grasp.append(
                np.array([p.positions for p in approach_plan.joint_trajectory.points])
            )

        if len(intersections_from_pre_grasp_to_grasp) == 0:
            continue

        ######### need to seach the intersection between grasp and post-grasp manifolds #########

        # First to compute the ik solution for checking the feasibility
        grasp_pose_mat = np.dot(numpify(target_cup_pose.pose), g)
        post_grasp_pose_mat = np.dot(
            grasp_pose_mat,
            np.array([[1, 0, 0, -0.09], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
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

        possible_ik_solutions = []

        for _ in range(10):
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

            if not ik_res.error_code.val == 1:
                continue

            # if there is a similar solution in possible ik solution, then skip it
            if len(possible_ik_solutions) > 0:
                if (
                    np.linalg.norm(
                        np.array(ik_res.solution.joint_state.position)
                        - np.array(possible_ik_solutions),
                        axis=1,
                    ).min()
                    < 0.01
                ):
                    continue

            possible_ik_solutions.append(ik_res.solution.joint_state.position)

            # need to check the motion for post-grasp
            moveit_robot_state = robot.get_current_state()
            moveit_robot_state.joint_state.position = (
                ik_res.solution.joint_state.position
            )

            move_group.set_start_state(moveit_robot_state)
            (release_plan, fraction) = move_group.compute_cartesian_path(
                [msgify(geometry_msgs.msg.Pose, post_grasp_pose_mat)], 0.01, 0.0
            )

            if fraction < 0.97:
                continue

            # save the intersection motion from pre-grasp manifold to grasp manifold.
            intersections_from_grasp_to_post_grasp.append(
                np.array([p.positions for p in release_plan.joint_trajectory.points])
            )

        if len(intersections_from_grasp_to_post_grasp) == 0:
            continue

        current_grasp_manifold_id = num_of_grasp_manifolds

        # create the manifold for this grasp
        grasp_manifold = Manifold(
            1,  # foliation id
            current_grasp_manifold_id,  # manifold id
            "cup",  # object name
            package_path + "/mesh_dir/cup.stl",  # object mesh file name
            True,
        )  # is the object in hand

        grasp_manifold.add_constraint(
            g,  # grasp pose in the object frame
            np.eye(4),  # constraint pose
            np.array([3.14, 3.14, 3.14]),  # orientation constraint
            np.array([2000, 2000, 2000]),  # position constraint
        )

        experiment.add_manifold(grasp_manifold)
        num_of_grasp_manifolds += 1

        # need to save the intersection motion from pre-grasp manifold to grasp manifold.
        for intersection_motion in intersections_from_pre_grasp_to_grasp:
            intersection = Intersection(
                1,
                current_grasp_manifold_id,
                0,
                0,
                False,
                intersection_motion,
                numpify(init_cup_pose.pose),
                package_path + "/mesh_dir/cup.stl",
                "cup",
            )
            experiment.add_intersection(intersection)

        # need to save the intersection motion from grasp manifold to post-grasp manifold.
        for intersection_motion in intersections_from_grasp_to_post_grasp:
            intersection = Intersection(
                1,
                current_grasp_manifold_id,
                2,
                0,
                False,
                intersection_motion,
                numpify(target_cup_pose.pose),
                package_path + "/mesh_dir/cup.stl",
                "cup",
            )
            experiment.add_intersection(intersection)

        # print("Find a feasible grasp pose")
        # grasp_pose_stamped = PoseStamped()
        # grasp_pose_stamped.header.stamp = rospy.Time.now()
        # grasp_pose_stamped.header.frame_id = 'base_link'
        # grasp_pose_stamped.pose = msgify(geometry_msgs.msg.Pose, np.dot(numpify(init_cup_pose.pose), g))
        # grasp_poses_for_vis.append(grasp_pose_stamped)
        # break

    # grasp_visualizer(grasp_poses_for_vis,
    #              [0.08 for _ in range(len(grasp_poses_for_vis))],
    #              [0 for _ in range(len(grasp_poses_for_vis))])

    # need to set start and goal foliation manifold id
    experiment.set_start_and_goal_foliation_manifold_id(0, 0, 2, 0)

    # save the experiment
    experiment.save(package_path + "/experiment_dir/" + experiment.experiment_name)

    # shutdown the moveit
    moveit_commander.roscpp_shutdown()
    moveit_commander.os._exit(0)
