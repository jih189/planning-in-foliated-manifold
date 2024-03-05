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
import numpy as np
import time
import json
import os
import tqdm

if __name__ == "__main__":
    """
    This is the main function for evaluation on all different task planners on the same experiment.
    """
    ##################################################################################

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("evaluation_node", anonymous=True)

    rospack = rospkg.RosPack()
    # Get the path of the desired package
    package_path = rospack.get_path("task_planner")

    # load the gmm
    gmm_dir_path = package_path + "/computed_gmms_dir/dpgmm/"
    gmm = GMM()
    gmm.load_distributions(gmm_dir_path)

    ################### parameters(you should modify here only) ###################

    experiment_name = rospy.get_param("~experiment_name", "maze")
    max_attempt_times = rospy.get_param("~max_attempt_times", 100)
    experiment_times = rospy.get_param("~experiment_times", 50)

    print("experiment_name: " + experiment_name)
    print("max_attempt_times: " + str(max_attempt_times))
    print("experiment_times: " + str(experiment_times))

    evaulated_task_planners = []  # you can add more task planners here
    evaulated_task_planners.append(MTGTaskPlanner())
    evaulated_task_planners.append(MDPTaskPlanner())
    evaulated_task_planners.append(MTGTaskPlannerWithGMM(gmm))
    evaulated_task_planners.append(MDPTaskPlannerWithGMM(gmm))

    #####################################################################

    # load the expierment
    experiment = Experiment()
    experiment.load(package_path + "/experiment_dir/" + experiment_name)

    #####################################################################################

    # # initialize the motion planner
    # moveit_commander.roscpp_initialize(sys.argv)
    # rospy.init_node('main_pipeline_node', anonymous=True)
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()

    rospy.sleep(0.5)  # wait for the planning scene to be ready
    scene.clear()
    move_group = moveit_commander.MoveGroupCommander("arm")

    move_group.set_planner_id("CDISTRIBUTIONRRTConfigDefault")

    move_group.set_planning_time(2.0)

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

    ##########################################################################################

    # load the obstacle into the planning scene.
    obstacle_pose_stamped = PoseStamped()
    obstacle_pose_stamped.header.frame_id = "base_link"
    obstacle_pose_stamped.pose = msgify(
        geometry_msgs.msg.Pose, experiment.obstacle_mesh_pose
    )
    scene.add_mesh(
        "obstacle", obstacle_pose_stamped, experiment.obstacle_mesh, size=(1, 1, 1)
    )

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

    ##############################################################################################

    evaluated_data = []

    # run the experiment on all task planners for evaluation
    for task_planner in evaulated_task_planners:
        print("evaluating task planner: " + task_planner.planner_name + " ...")
        for _ in tqdm.tqdm(range(experiment_times)):
            # reset the task planner
            task_planner.reset_task_planner()

            # load the experiment to all evalued task planners
            # add manifolds
            for manifold in experiment.manifolds:
                manifold_object_pose = (
                    manifold.in_hand_pose
                    if manifold.has_object_in_hand
                    else manifold.object_pose
                )

                manifold_constraint = (
                    construct_moveit_constraint(
                        manifold.in_hand_pose,
                        manifold.constraint_pose,
                        manifold.orientation_constraint,
                        manifold.position_constraint,
                    )
                    if manifold.has_object_in_hand
                    else get_no_constraint()
                )

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

            # add start and goal
            # set start and goal configurations with relative foliation and manifold id
            task_planner.set_start_and_goal(
                (
                    experiment.start_foliation_id,
                    experiment.start_manifold_id,
                ),  # start manifold id
                move_group.get_current_joint_values(),  # start configuration
                (
                    experiment.goal_foliation_id,
                    experiment.goal_manifold_id,
                ),  # goal manifold id
                move_group.get_current_joint_values(),  # goal configuration
            )

            start_time = time.time()
            found_solution = False
            total_distance = 0.0
            # run the main pipeline of the planner
            for _ in range(max_attempt_times):
                # generate task sequence
                task_sequence = task_planner.generate_task_sequence()

                if (
                    len(task_sequence) == 0
                ):  # if no task sequence found, then break the loop
                    print("no task sequence found")
                    break

                found_solution = True
                solution_path = []

                # motion planner tries to solve each task in the task sequence
                for task in task_sequence:
                    if task.has_solution:
                        solution_path.append(task.solution_trajectory)
                        if len(task.next_motion) > 1:
                            intersection_motion = (
                                convert_joint_values_to_robot_trajectory(
                                    task.next_motion, move_group.get_active_joints()
                                )
                            )
                            solution_path.append(intersection_motion)
                        continue

                    if task.manifold_detail.has_object_in_hand:  # has object in hand
                        # do the motion planning
                        move_group.clear_path_constraints()
                        move_group.clear_in_hand_pose()

                        # set start and goal congfiguration to motion planner.
                        start_moveit_robot_state = convert_joint_values_to_robot_state(
                            task.start_configuration,
                            move_group.get_active_joints(),
                            robot,
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

                        # In this project, we use C-Distribution RRT as the motion planner. It will return a set of
                        # sampled data in the robot state format with tag about whether the sampled data is valid or not.
                        # so we need to convert it back to the numpy format based on the active joints.
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
                            # add the intersection motion to the solution path
                            if len(task.next_motion) > 1:
                                intersection_motion = (
                                    convert_joint_values_to_robot_trajectory(
                                        task.next_motion, move_group.get_active_joints()
                                    )
                                )
                                solution_path.append(intersection_motion)
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
                            task.start_configuration,
                            move_group.get_active_joints(),
                            robot,
                        )
                        move_group.set_start_state(start_moveit_robot_state)
                        move_group.set_joint_value_target(task.goal_configuration)
                        move_group.set_path_constraints(task.manifold_detail.constraint)

                        motion_plan_result = move_group.plan()

                        # In this project, we use C-Distribution RRT as the motion planner. It will return a set of
                        # sampled data in the robot state format with tag about whether the sampled data is valid or not.
                        # so we need to convert it back to the numpy format based on the active joints.
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
                            # add the intersection motion to the solution path
                            if len(task.next_motion) > 1:
                                intersection_motion = (
                                    convert_joint_values_to_robot_trajectory(
                                        task.next_motion, move_group.get_active_joints()
                                    )
                                )
                                solution_path.append(intersection_motion)
                        else:  # if the motion planner can't find a solution, then replan
                            found_solution = False

                        # remove the object from the planning scene
                        scene.remove_world_object(task.manifold_detail.object_name)

                        # check whether the object is in the planning scene
                        # if it is, wait for short time.
                        while (
                            task.manifold_detail.object_name
                            in scene.get_known_object_names()
                        ):
                            rospy.sleep(0.0001)

                        if not found_solution:
                            break

                if found_solution:  # found solution, then break the
                    solution_trajectory = []
                    for s in solution_path:
                        # get the length of the solution path
                        for p in s.joint_trajectory.points:
                            solution_trajectory.append(p.positions)

                    solution_trajectory = np.array(solution_trajectory)
                    differences = np.diff(solution_trajectory, axis=0)
                    distances = np.linalg.norm(differences, axis=1)
                    total_distance = np.sum(distances)

                    break

            end_time = time.time()
            json_data = {
                "task_planner": task_planner.planner_name,
                "experiment_name": experiment_name,
            }
            if found_solution:
                json_data["found_solution"] = True
                json_data["time"] = end_time - start_time
                json_data["total_distance"] = total_distance
            else:
                json_data["found_solution"] = False

            evaluated_data.append(json_data)

    # create a evalued_data_dir
    if not os.path.exists(package_path + "/evaluated_data_dir"):
        os.makedirs(package_path + "/evaluated_data_dir")

    with open(package_path + "/evaluated_data_dir/evaluated_data.json", "w") as outfile:
        json.dump(evaluated_data, outfile, indent=4)

    # shutdown the moveit
    moveit_commander.roscpp_shutdown()
    moveit_commander.os._exit(0)
