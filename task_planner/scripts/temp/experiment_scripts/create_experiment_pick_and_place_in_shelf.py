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
    shelf_pose = PoseStamped()
    shelf_pose.header.frame_id = "base_link"
    shelf_pose.pose.position.x = 0.95
    shelf_pose.pose.position.y = 0.3
    shelf_pose.pose.position.z = 0
    shelf_pose.pose.orientation.x = 0
    shelf_pose.pose.orientation.y = 0
    shelf_pose.pose.orientation.z = 1.0
    shelf_pose.pose.orientation.w = 0
    print("Add the shelf to the planning scene")
    rospack = rospkg.RosPack()
    # Get the path of the desired package
    package_path = rospack.get_path("task_planner")
    scene.add_mesh("shelf", shelf_pose, package_path + "/mesh_dir/shelf.stl")

    # sleep for a while to make sure the planning scene is ready
    rospy.sleep(1)

    # load the cup
    init_cup_pose = PoseStamped()
    init_cup_pose.header.frame_id = "base_link"
    init_cup_pose.pose.position.x = 0.7
    init_cup_pose.pose.position.y = 0.0
    init_cup_pose.pose.position.z = 0.9
    init_cup_pose.pose.orientation.x = 0
    init_cup_pose.pose.orientation.y = 0
    init_cup_pose.pose.orientation.z = 0
    init_cup_pose.pose.orientation.w = 1
    scene.add_mesh("init_cup", init_cup_pose, package_path + "/mesh_dir/cup.stl")

    regrasp_cup_pose = PoseStamped()
    regrasp_cup_pose.header.frame_id = "base_link"
    regrasp_cup_pose.pose.position.x = 0.5
    regrasp_cup_pose.pose.position.y = 0.0
    regrasp_cup_pose.pose.position.z = 0.9
    regrasp_cup_pose.pose.orientation.x = 0
    regrasp_cup_pose.pose.orientation.y = 0
    regrasp_cup_pose.pose.orientation.z = 0
    regrasp_cup_pose.pose.orientation.w = 1
    scene.add_mesh("regrasp_cup", regrasp_cup_pose, package_path + "/mesh_dir/cup.stl")

    target_cup_pose = PoseStamped()
    target_cup_pose.header.frame_id = "base_link"
    target_cup_pose.pose.position.x = 0.5
    target_cup_pose.pose.position.y = 0.0
    target_cup_pose.pose.position.z = 1.23
    target_cup_pose.pose.orientation.x = 0
    target_cup_pose.pose.orientation.y = 0
    target_cup_pose.pose.orientation.z = 0
    target_cup_pose.pose.orientation.w = 1
    scene.add_mesh("target_cup", target_cup_pose, package_path + "/mesh_dir/cup.stl")

    rospy.sleep(1)

    # load the grasp poses for the cup

    grasp_pose_list = []

    loaded_array = np.load(package_path + "/mesh_dir/cup.npz")

    rotated_matrix = np.array(
        [[1, 0, 0, -0.17], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )

    # randomly select 20 elements from the loaded_array.files
    for ind in random.sample(list(range(len(loaded_array.files))), 100):
        array_name = loaded_array.files[ind]
        grasp_pose_list.append(np.dot(loaded_array[array_name], rotated_matrix))

    ###################################### start to generate the pick-and-place-in-shelf experiment ######################################

    experiment = Experiment()

    experiment.setup(
        "pick_and_place_in_shelf",
        package_path + "/mesh_dir/cup.stl",
        package_path + "/mesh_dir/shelf.stl",
        numpify(shelf_pose.pose),
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

    # add the manifold for re-grasp
    re_grasp_manifold = Manifold(
        2,  # foliation id
        0,  # manifold id
        "cup",  # object name
        package_path + "/mesh_dir/cup.stl",  # object mesh file name
        False,
    )  # is the object in hand

    # set the initial object placement pose into pre-grasp manifold
    re_grasp_manifold.add_object_placement(numpify(regrasp_cup_pose.pose))
    experiment.add_manifold(re_grasp_manifold)

    post_grasp_manifold = Manifold(
        4,  # foliation id
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
        intersections_from_pre_grasp_to_grasp = []
        intersections_from_grasp_to_regrasp = []
        intersections_from_regrasp_to_grasp = []
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

        ###### need to search intersections from grasp to re-grasp ######

        # First to compute the ik solution for checking the feasibility
        re_grasp_pose_mat = np.dot(numpify(regrasp_cup_pose.pose), g)
        pre_re_grasp_pose_mat = np.dot(
            re_grasp_pose_mat,
            np.array([[1, 0, 0, -0.09], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        )

        # set the ik target pose
        ik_target_pose = PoseStamped()
        ik_target_pose.header.stamp = rospy.Time.now()
        ik_target_pose.header.frame_id = "base_link"
        ik_target_pose.pose = msgify(geometry_msgs.msg.Pose, re_grasp_pose_mat)

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
                [msgify(geometry_msgs.msg.Pose, pre_re_grasp_pose_mat)], 0.01, 0.0
            )

            if fraction < 0.97:
                continue

            # save the intersection motion from pre-grasp manifold to grasp manifold.
            intersections_from_grasp_to_regrasp.append(
                np.array([p.positions for p in approach_plan.joint_trajectory.points])
            )
            intersections_from_regrasp_to_grasp.append(
                np.array([p.positions for p in approach_plan.joint_trajectory.points])
            )

        # ######### need to seach the intersection between grasp and post-grasp manifolds #########

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

        if (
            len(intersections_from_pre_grasp_to_grasp) == 0
            and len(intersections_from_grasp_to_post_grasp) == 0
            and len(intersections_from_regrasp_to_grasp) == 0
            and len(intersections_from_grasp_to_regrasp) == 0
        ):
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

        lower_shelf_pose = np.array(
            [[1, 0, 0, 0.7], [0, 1, 0, 0], [0, 0, 1, 0.9], [0, 0, 0, 1]]
        )

        grasp_manifold.add_constraint(
            g,  # grasp pose in the object frame
            lower_shelf_pose,  # constraint pose
            np.array([0.1, 0.1, 3.14 * 2]),  # orientation constraint
            np.array([2000, 2000, 0.05]),  # position constraint
        )

        experiment.add_manifold(grasp_manifold)

        # create the manifold for this grasp
        grasp_manifold = Manifold(
            3,  # foliation id
            current_grasp_manifold_id,  # manifold id
            "cup",  # object name
            package_path + "/mesh_dir/cup.stl",  # object mesh file name
            True,
        )  # is the object in hand

        grasp_manifold.add_constraint(
            g,  # grasp pose in the object frame
            np.eye(4),  # constraint pose
            np.array([0.1, 0.1, 3.14 * 2]),  # orientation constraint
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

        # need to save the intersection motion from grasp manifold to re-grasp manifold.
        for intersection_motion in intersections_from_grasp_to_regrasp:
            intersection = Intersection(
                1,
                current_grasp_manifold_id,
                2,
                0,
                False,
                intersection_motion,
                numpify(regrasp_cup_pose.pose),
                package_path + "/mesh_dir/cup.stl",
                "cup",
            )
            experiment.add_intersection(intersection)

        # need to save the intersection motion from re-grasp manifold to grasp manifold.
        for intersection_motion in intersections_from_regrasp_to_grasp:
            intersection = Intersection(
                3,
                current_grasp_manifold_id,
                2,
                0,
                False,
                intersection_motion,
                numpify(regrasp_cup_pose.pose),
                package_path + "/mesh_dir/cup.stl",
                "cup",
            )
            experiment.add_intersection(intersection)

        # need to save the intersection motion from grasp manifold to post-grasp manifold.
        for intersection_motion in intersections_from_grasp_to_post_grasp:
            intersection = Intersection(
                3,
                current_grasp_manifold_id,
                4,
                0,
                False,
                intersection_motion,
                numpify(target_cup_pose.pose),
                package_path + "/mesh_dir/cup.stl",
                "cup",
            )
            experiment.add_intersection(intersection)

    # need to set start and goal foliation manifold id
    experiment.set_start_and_goal_foliation_manifold_id(0, 0, 4, 0)

    # save the experiment
    experiment.save(package_path + "/experiment_dir/" + experiment.experiment_name)

    # shutdown the moveit
    moveit_commander.roscpp_shutdown()
    moveit_commander.os._exit(0)
