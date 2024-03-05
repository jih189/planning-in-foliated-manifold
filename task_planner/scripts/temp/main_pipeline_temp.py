#!/usr/bin/env python
from experiment_scripts.experiment_helper import Experiment, Manifold, Intersection
from jiaming_task_planner import (
    MTGTaskPlanner,
    MDPTaskPlanner,
    MTGTaskPlannerWithGMM,
    MDPTaskPlannerWithGMM,
    GMM,
    ManifoldDetail,
    IntersectionDetail,
)
from jiaming_helper import (
    convert_joint_values_to_robot_trajectory,
    convert_joint_values_to_robot_state,
    get_no_constraint,
    construct_moveit_constraint,
    make_mesh,
)

import sys
import copy
import rospy
import rospkg
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from moveit_msgs.srv import (
    GetStateValidity,
    GetStateValidityRequest,
    GetJointWithConstraints,
    GetJointWithConstraintsRequest,
)
from moveit_msgs.msg import (
    RobotState,
    Constraints,
    OrientationConstraint,
    MoveItErrorCodes,
    AttachedCollisionObject,
)
from sensor_msgs.msg import JointState
from ros_numpy import numpify, msgify
from geometry_msgs.msg import Quaternion, Point, Pose, PoseStamped, Point32
import trimesh
from trimesh import transformations
from trimesh_util import sample_points_on_mesh
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField, PointCloud

# import struct
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
import time
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import threading
import os


running_flag = True


def combine_robot_trajectory_list(robot_trajectory_list):
    result_trajectory = moveit_msgs.msg.RobotTrajectory()
    result_trajectory.joint_trajectory.header.frame_id = robot_trajectory_list[
        0
    ].joint_trajectory.header.frame_id
    result_trajectory.joint_trajectory.joint_names = robot_trajectory_list[
        0
    ].joint_trajectory.joint_names
    time_from_start = rospy.Duration(0)
    for robot_trajectory in robot_trajectory_list:
        for joint_trajectory_point in robot_trajectory.joint_trajectory.points:
            # append the deep copy of joint_trajectory_point to the result_trajectory
            result_trajectory.joint_trajectory.points.append(
                copy.deepcopy(joint_trajectory_point)
            )
            result_trajectory.joint_trajectory.points[
                -1
            ].time_from_start = time_from_start
            time_from_start += rospy.Duration(0.01)

    return result_trajectory


# np.set_printoptions(suppress=True, precision = 3)
if __name__ == "__main__":
    ##########################################################
    #################### experiment setup ####################
    max_attempt_times = 100

    # experiment_name = "pick_and_place"
    # experiment_name = "move_mouse_with_constraint"
    # experiment_name = "open_door"
    # experiment_name = "move_mouse"
    # experiment_name = "maze"
    # experiment_name = "pick_and_place_in_shelf"
    experiment_name = "pick_and_place_with_constraint"

    use_mtg = True  # use mtg or mdp
    use_gmm = True  # use gmm or not

    ##########################################################

    rospack = rospkg.RosPack()
    # Get the path of the desired package
    package_path = rospack.get_path("task_planner")

    # load the gmm
    gmm_dir_path = package_path + "/computed_gmms_dir/dpgmm/"
    gmm = GMM()
    gmm.load_distributions(gmm_dir_path)

    # load the expierment
    experiment = Experiment()
    experiment.load(package_path + "/experiment_dir/" + experiment_name)

    # load the experiment into the task planner
    if use_mtg:
        if use_gmm:
            task_planner = MTGTaskPlannerWithGMM(gmm)
        else:
            task_planner = MTGTaskPlanner()
    else:
        if use_gmm:
            # task_planner = MDPTaskPlannerWithGMM(gmm, 'MDPTaskPlannerWithGMMWithShortcut', {'use_shortcut': True}) # this is the example of customizing the mdp task planner
            task_planner = MDPTaskPlannerWithGMM(gmm)
        else:
            task_planner = MDPTaskPlanner()

    task_planner.reset_task_planner()

    #############################################################################
    # setup the task graph.
    # add manifolds
    for manifold in experiment.manifolds:
        # the manifold id in the task planner should be a pair of (foliation_id, manifold_id)
        # the constraint here has (moveit constraint, has_object_in_hand, object_pose, object mesh, object name)
        # if the has_object_in_hand is true, then object pose here is the in-hand-object-pose which is the grasp pose in the object frame
        # if the has_object_in_hand is false, then object pose here is the object placement pose which is the placement pose in the world frame

        # print hte constraint detail here if the object is in hand.
        manifold_constraint = (
            construct_moveit_constraint(
                np.linalg.inv(manifold.in_hand_pose),
                manifold.constraint_pose,
                manifold.orientation_constraint,
                manifold.position_constraint,
            )
            if manifold.has_object_in_hand
            else get_no_constraint()
        )

        # if the manifold has object in hand, then the object pose is the grasp pose in the object frame
        # if the manifold has no object in hand, then the object pose is the placement pose in the world frame
        manifold_object_pose = (
            manifold.in_hand_pose
            if manifold.has_object_in_hand
            else manifold.object_pose
        )

        # manifold.has_object_in_hand
        task_planner.add_manifold(
            ManifoldDetail(
                manifold_constraint,
                manifold.has_object_in_hand,
                manifold_object_pose,
                manifold.object_mesh,
                manifold.object_name,
            ),
            (manifold.foliation_id, manifold.manifold_id),
        )

    # add intersections
    for intersection in experiment.intersections:
        task_planner.add_intersection(
            (intersection.foliation_id_1, intersection.manifold_id_1),
            (intersection.foliation_id_2, intersection.manifold_id_2),
            IntersectionDetail(
                intersection.has_object_in_hand,
                intersection.trajectory_motion,
                intersection.in_hand_pose,
                intersection.object_mesh,
                intersection.object_name,
            ),
        )

    # #############################################################################
    # # uncomment this to draw the similarity distance plot

    # task_planner.draw_similarity_distance_plot()

    #############################################################################
    # initialize the motion planner and planning scene of moveit
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("main_pipeline_node", anonymous=True)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    rospy.sleep(0.5)  # wait for the planning scene to be ready
    scene.clear()
    move_group = moveit_commander.MoveGroupCommander("arm")

    move_group.set_planner_id("CDISTRIBUTIONRRTConfigDefault")
    # move_group.set_planner_id('CBIRRTConfigDefault')

    move_group.set_planning_time(2.0)

    display_trajectory_publisher = rospy.Publisher(
        "/move_group/result_display_trajectory",
        moveit_msgs.msg.DisplayTrajectory,
        queue_size=5,
    )

    display_robot_state_publisher = rospy.Publisher(
        "/move_group/result_display_robot_state",
        moveit_msgs.msg.DisplayRobotState,
        queue_size=5,
    )

    # set initial joint state
    joint_state_publisher = rospy.Publisher(
        "/move_group/fake_controller_joint_states", JointState, queue_size=1
    )

    # Create a JointState message
    initial_joint_state = JointState()
    initial_joint_state.header.stamp = rospy.Time.now()
    initial_joint_state.name = [
        "torso_lift_joint",
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "upperarm_roll_joint",
        "elbow_flex_joint",
        "wrist_flex_joint",
        "l_gripper_finger_joint",
        "r_gripper_finger_joint",
    ]
    initial_joint_state.position = [0.38, -1.28, 1.51, 0.35, 1.81, 1.47, 0.04, 0.04]

    rate = rospy.Rate(10)
    while (
        joint_state_publisher.get_num_connections() < 1
    ):  # need to wait until the publisher is ready.
        rate.sleep()
    joint_state_publisher.publish(initial_joint_state)

    ############################ marker publisher ################################

    marker_publisher = rospy.Publisher(
        "/visualization_marker", MarkerArray, queue_size=10
    )

    intermediate_object_publisher = rospy.Publisher(
        "/intermediate_object", Marker, queue_size=10
    )

    # publish a marker with threading.
    def publish_marker_thread(
        object_marker_array_, obstacle_pointcloud_, marker_publisher_
    ):
        # we use this as the helper to publish the marker for debugging.
        global running_flag

        while not rospy.is_shutdown() and running_flag:
            for m in object_marker_array_.markers:
                m.header.stamp = rospy.Time.now()
            marker_publisher_.publish(object_marker_array_)

            # convert it to marker
            obstacle_pointcloud_marker = Marker()
            obstacle_pointcloud_marker.header.frame_id = "base_link"
            obstacle_pointcloud_marker.id = 3
            obstacle_pointcloud_marker.type = Marker.POINTS
            obstacle_pointcloud_marker.action = Marker.ADD
            obstacle_pointcloud_marker.pose = Pose()
            obstacle_pointcloud_marker.scale = Point(0.01, 0.01, 0.01)
            obstacle_pointcloud_marker.color = ColorRGBA(0, 1, 0, 1)
            obstacle_pointcloud_marker.points = [
                Point32(p[0], p[1], p[2]) for p in obstacle_pointcloud_
            ]  # convert the numpy array to the point32 list
            # marker_publisher_.publish([obstacle_pointcloud_marker])

            rospy.sleep(0.1)

        # clear the marker
        for m in object_marker_array_.markers:
            m.header.stamp = rospy.Time.now()
            m.action = Marker.DELETE
        marker_publisher_.publish(object_marker_array_)

    ##############################################################################################

    ##############################################################################

    # load the obstacle into the planning scene.
    obstacle_pose_stamped = PoseStamped()
    obstacle_pose_stamped.header.frame_id = "base_link"
    obstacle_pose_stamped.pose = msgify(
        geometry_msgs.msg.Pose, experiment.obstacle_mesh_pose
    )
    scene.add_mesh(
        "obstacle", obstacle_pose_stamped, experiment.obstacle_mesh, size=(1, 1, 1)
    )

    ##############################################################################################

    # set start and goal configurations with relative foliation and manifold id
    task_planner.set_start_and_goal(
        (
            experiment.start_foliation_id,
            experiment.start_manifold_id,
        ),  # start manifold id
        move_group.get_current_joint_values(),  # start configuration
        (experiment.goal_foliation_id, experiment.goal_manifold_id),  # goal manifold id
        move_group.get_current_joint_values(),  # goal configuration
    )

    object_marker_array = MarkerArray()

    # add the obstacle into the marker array
    obstacle_marker = Marker()
    obstacle_marker.header.frame_id = "base_link"
    obstacle_marker.id = 0
    obstacle_marker.type = Marker.MESH_RESOURCE
    obstacle_marker.action = Marker.ADD
    obstacle_marker.pose = msgify(geometry_msgs.msg.Pose, experiment.obstacle_mesh_pose)
    obstacle_marker.scale = Point(1, 1, 1)
    obstacle_marker.color = ColorRGBA(0.5, 0.5, 0.5, 1)
    obstacle_marker.mesh_resource = (
        "package://task_planner/mesh_dir/" + os.path.basename(experiment.obstacle_mesh)
    )

    object_marker_array.markers.append(obstacle_marker)

    # generate pointcloud based on the obstacle mesh
    obstacle_mesh = trimesh.load_mesh(experiment.obstacle_mesh)
    sampling_num_point_on_mesh = int(
        obstacle_mesh.area / 0.003
    )  # you can reduce the value here to make more points on the mesh.
    obstacle_pointcloud = sample_points_on_mesh(
        obstacle_mesh, sampling_num_point_on_mesh
    )
    # apply the obstacle pose to the pointcloud
    obstacle_pointcloud = np.dot(
        experiment.obstacle_mesh_pose,
        np.vstack((obstacle_pointcloud.T, np.ones((1, obstacle_pointcloud.shape[0])))),
    ).T[:, 0:3]

    # if the task planner does not use gmm, then this function is the empty function of the base class.
    task_planner.read_pointcloud(obstacle_pointcloud)

    ####################################################################################################################
    # going to publish the object marker here.
    if not task_planner.manifold_info[
        (experiment.start_foliation_id, experiment.start_manifold_id)
    ].has_object_in_hand:
        # the object is not in hand initially, so we can publish the object marker here.

        init_object_marker = Marker()
        init_object_marker.header.frame_id = "base_link"
        init_object_marker.id = 1
        init_object_marker.type = Marker.MESH_RESOURCE
        init_object_marker.action = Marker.ADD
        init_object_marker.pose = msgify(
            geometry_msgs.msg.Pose,
            task_planner.manifold_info[
                (experiment.start_foliation_id, experiment.start_manifold_id)
            ].object_pose,
        )
        init_object_marker.scale = Point(1, 1, 1)
        init_object_marker.color = ColorRGBA(0, 0, 1, 1)
        # the rviz only takes the mesh file name start with package://
        init_object_marker.mesh_resource = (
            "package://task_planner/mesh_dir/"
            + os.path.basename(
                task_planner.manifold_info[
                    (experiment.start_foliation_id, experiment.start_manifold_id)
                ].object_mesh
            )
        )

        object_marker_array.markers.append(init_object_marker)

    if not task_planner.manifold_info[
        (experiment.goal_foliation_id, experiment.goal_manifold_id)
    ].has_object_in_hand:
        goal_object_marker = Marker()
        goal_object_marker.header.frame_id = "base_link"
        goal_object_marker.id = 2
        goal_object_marker.type = Marker.MESH_RESOURCE
        goal_object_marker.action = Marker.ADD
        goal_object_marker.pose = msgify(
            geometry_msgs.msg.Pose,
            task_planner.manifold_info[
                (experiment.goal_foliation_id, experiment.goal_manifold_id)
            ].object_pose,
        )
        goal_object_marker.scale = Point(1, 1, 1)
        goal_object_marker.color = ColorRGBA(1, 0, 0, 1)
        goal_object_marker.mesh_resource = (
            "package://task_planner/mesh_dir/"
            + os.path.basename(
                task_planner.manifold_info[
                    (experiment.goal_foliation_id, experiment.goal_manifold_id)
                ].object_mesh
            )
        )

        object_marker_array.markers.append(goal_object_marker)

    marker_thread = threading.Thread(
        target=publish_marker_thread,
        args=(object_marker_array, obstacle_pointcloud, marker_publisher),
    )

    marker_thread.start()

    ##############################################################################
    # create attachedCollisionObject
    current_object_pose_stamped = PoseStamped()
    current_object_pose_stamped.header.frame_id = "wrist_roll_link"
    current_object_pose_stamped.pose = Pose()
    manipulated_object = make_mesh(
        "object", current_object_pose_stamped, experiment.manipulated_object_mesh
    )

    attached_object = AttachedCollisionObject()
    attached_object.link_name = "wrist_roll_link"
    attached_object.object = manipulated_object
    attached_object.touch_links = [
        "l_gripper_finger_link",
        "r_gripper_finger_link",
        "gripper_link",
    ]

    ##############################################################################
    # start the main pipeline
    found_solution = False

    for attempt_time in range(max_attempt_times):
        print("attempt: ", attempt_time)
        # generate task sequence
        task_planning_start_time = time.time()
        task_sequence = task_planner.generate_task_sequence()
        task_planning_end_time = time.time()
        print("task planning time: ", task_planning_end_time - task_planning_start_time)
        if len(task_sequence) == 0:  # if no task sequence found, then break the loop
            print("no task sequence found")
            break

        found_solution = True
        solution_path = []
        object_pose_in_solution_path = (
            []
        )  # this is a array of object pose in the solution path: [(is_in_hand, object_pose), ...]

        # motion planner tries to solve each task in the task sequence
        print("--- tasks ---")
        print("length: ", len(task_sequence))
        # for task in task_sequence:
        #     print " has solution: ", task.has_solution

        for task in task_sequence:
            # print the task detail here
            # task.print_task_detail()

            # if solution exists for the task, then we can skip the task.
            if task.has_solution:
                solution_path.append(task.solution_trajectory)
                if task.manifold_detail.has_object_in_hand:
                    object_pose_in_solution_path.append(
                        (True, np.linalg.inv(task.manifold_detail.object_pose))
                    )
                else:
                    object_pose_in_solution_path.append(
                        (False, task.manifold_detail.object_pose)
                    )

                # add the intersection motion to the solution path
                if len(task.next_motion) > 1:
                    intersection_motion = convert_joint_values_to_robot_trajectory(
                        task.next_motion, move_group.get_active_joints()
                    )
                    solution_path.append(intersection_motion)
                    object_pose_in_solution_path.append(
                        (False, task.manifold_detail.object_pose)
                    )
                continue

            if task.manifold_detail.has_object_in_hand:  # has object in hand
                # do the motion planning
                move_group.clear_path_constraints()
                move_group.clear_in_hand_pose()

                # set start and goal congfiguration to motion planner.
                start_moveit_robot_state = convert_joint_values_to_robot_state(
                    task.start_configuration, move_group.get_active_joints(), robot
                )

                # add the attached object to the start state
                attached_object.object.pose = msgify(
                    geometry_msgs.msg.Pose,
                    np.linalg.inv(task.manifold_detail.object_pose),
                )
                start_moveit_robot_state.attached_collision_objects.append(
                    attached_object
                )

                move_group.set_start_state(start_moveit_robot_state)
                move_group.set_joint_value_target(task.goal_configuration)
                move_group.set_path_constraints(task.manifold_detail.constraint)
                move_group.set_in_hand_pose(
                    msgify(
                        geometry_msgs.msg.Pose,
                        np.linalg.inv(task.manifold_detail.object_pose),
                    )
                )

                motion_plan_result = move_group.plan()

                # if the planner uses gmm, then it will convert sampled data in the robot state format.
                # so we need to convert it back to the numpy format based on the active joints.
                if use_gmm:
                    for m in motion_plan_result[4].verified_motions:
                        m.sampled_state = [
                            m.sampled_state.joint_state.position[
                                m.sampled_state.joint_state.name.index(j)
                            ]
                            for j in move_group.get_active_joints()
                        ]

                task_planner.update(task.task_graph_info, motion_plan_result)

                if motion_plan_result[
                    0
                ]:  # if the motion planner find motion solution, then add it to the solution path
                    solution_path.append(motion_plan_result[1])
                    object_pose_in_solution_path.append(
                        (True, np.linalg.inv(task.manifold_detail.object_pose))
                    )
                    # add the intersection motion to the solution path
                    if len(task.next_motion) > 1:
                        intersection_motion = convert_joint_values_to_robot_trajectory(
                            task.next_motion, move_group.get_active_joints()
                        )
                        solution_path.append(intersection_motion)
                        object_pose_in_solution_path.append(
                            (False, task.manifold_detail.object_pose)
                        )
                else:  # if the motion planner can't find a solution, then replan
                    found_solution = False

                if not found_solution:
                    break

            else:
                # add the object to the planning scene
                target_object_pose = PoseStamped()
                target_object_pose.header.frame_id = "base_link"
                target_object_pose.pose = msgify(
                    geometry_msgs.msg.Pose, task.manifold_detail.object_pose
                )
                scene.add_mesh(
                    task.manifold_detail.object_name,
                    target_object_pose,
                    task.manifold_detail.object_mesh,
                    size=(1, 1, 1),
                )

                # check whether the object is in the planning scene
                # if it is not, wait for short time.
                while (
                    task.manifold_detail.object_name
                    not in scene.get_known_object_names()
                ):
                    rospy.sleep(0.0001)

                # do the motion planning
                move_group.clear_path_constraints()
                move_group.clear_in_hand_pose()

                # set start and goal congfiguration to motion planner.
                start_moveit_robot_state = convert_joint_values_to_robot_state(
                    task.start_configuration, move_group.get_active_joints(), robot
                )
                move_group.set_start_state(start_moveit_robot_state)
                move_group.set_joint_value_target(task.goal_configuration)
                move_group.set_path_constraints(task.manifold_detail.constraint)
                # because the object is not grasped in the hand, no need to set the in-hand pose.

                motion_plan_result = move_group.plan()

                # if the planner uses gmm, then it will convert sampled data in the robot state format.
                # so we need to convert it back to the numpy format based on the active joints.
                if use_gmm:
                    for m in motion_plan_result[4].verified_motions:
                        m.sampled_state = [
                            m.sampled_state.joint_state.position[
                                m.sampled_state.joint_state.name.index(j)
                            ]
                            for j in move_group.get_active_joints()
                        ]

                task_planner.update(task.task_graph_info, motion_plan_result)

                if motion_plan_result[
                    0
                ]:  # if the motion planner find motion solution, then add it to the solution path
                    solution_path.append(motion_plan_result[1])
                    object_pose_in_solution_path.append(
                        (False, task.manifold_detail.object_pose)
                    )
                    # add the intersection motion to the solution path
                    if len(task.next_motion) > 1:
                        intersection_motion = convert_joint_values_to_robot_trajectory(
                            task.next_motion, move_group.get_active_joints()
                        )
                        solution_path.append(intersection_motion)
                        object_pose_in_solution_path.append(
                            (False, task.manifold_detail.object_pose)
                        )
                else:  # if the motion planner can't find a solution, then replan
                    found_solution = False

                # remove the object from the planning scene
                scene.remove_world_object(task.manifold_detail.object_name)

                # check whether the object is in the planning scene
                # if it is, wait for short time.
                while (
                    task.manifold_detail.object_name in scene.get_known_object_names()
                ):
                    rospy.sleep(0.0001)

                if not found_solution:
                    break

        if found_solution:  # found solution, then break the loop
            # save the solution_path into a file

            print("found solution")
            # try to execute the solution path

            # display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            # display_trajectory.trajectory_start = move_group.get_current_state()

            # attached a object into the robot state
            # display_trajectory.trajectory_start.attached_collision_objects.append(attached_object)

            # display_trajectory.trajectory = solution_path
            # display_trajectory.trajectory = [combine_robot_trajectory_list(solution_path)]

            need_to_break = False

            intermediate_object_marker = Marker()
            intermediate_object_marker.header.frame_id = "base_link"
            intermediate_object_marker.id = 0
            intermediate_object_marker.type = Marker.MESH_RESOURCE
            intermediate_object_marker.scale = Point(1, 1, 1)
            intermediate_object_marker.color = ColorRGBA(0, 1, 0, 1)
            intermediate_object_marker.mesh_resource = (
                "package://task_planner/mesh_dir/"
                + os.path.basename(
                    task_planner.manifold_info[
                        (experiment.start_foliation_id, experiment.start_manifold_id)
                    ].object_mesh
                )
            )
            print("press ctrl+c to display the trajectory")
            while not rospy.is_shutdown():
                for (
                    trajectory_in_solution_path,
                    each_object_pose_in_solution_path,
                ) in zip(solution_path, object_pose_in_solution_path):
                    for p in trajectory_in_solution_path.joint_trajectory.points:
                        current_robot_state_msg = moveit_msgs.msg.DisplayRobotState()
                        current_robot_state_msg.state = (
                            convert_joint_values_to_robot_state(
                                p.positions, move_group.get_active_joints(), robot
                            )
                        )
                        if each_object_pose_in_solution_path[
                            0
                        ]:  # if the object is in hand
                            attached_object.object.pose = msgify(
                                geometry_msgs.msg.Pose,
                                each_object_pose_in_solution_path[1],
                            )
                            current_robot_state_msg.state.attached_collision_objects.append(
                                attached_object
                            )
                            intermediate_object_marker.action = Marker.DELETE
                        else:
                            intermediate_object_marker.action = Marker.ADD
                            intermediate_object_marker.pose = msgify(
                                geometry_msgs.msg.Pose,
                                each_object_pose_in_solution_path[1],
                            )
                        intermediate_object_publisher.publish(
                            intermediate_object_marker
                        )

                        display_robot_state_publisher.publish(current_robot_state_msg)

                        # rospy sleep
                        rospy.sleep(0.01)

                        if rospy.is_shutdown():
                            need_to_break = True
                            break
                    if need_to_break:
                        break

                # display_trajectory_publisher.publish(display_trajectory)

            break

    if not found_solution:
        # wait for user to press enter to continue
        raw_input("Press any button to continue...")

    # make the marker thread stop
    running_flag = False
    marker_thread.join()

    # shutdown the moveit
    moveit_commander.roscpp_shutdown()
    moveit_commander.os._exit(0)
