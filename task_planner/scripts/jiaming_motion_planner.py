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
    SamplingDistribution,
)
from sensor_msgs.msg import JointState
from ros_numpy import numpify, msgify
from geometry_msgs.msg import Quaternion, Point, Pose, PoseStamped, Point32
import numpy as np

from jiaming_helper import (
    convert_joint_values_to_robot_trajectory,
    convert_joint_values_to_robot_state,
    get_no_constraint,
    construct_moveit_constraint,
    make_mesh,
    INIT_JOINT_NAMES, 
    INIT_JOINT_POSITIONS
)
from foliation_planning.foliated_base_class import BaseMotionPlanner
from custom_foliated_class import CustomTaskMotion


class MoveitMotionPlanner(BaseMotionPlanner):

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        rospy.sleep(0.5)  # wait for the planning scene to be ready
        self.scene.clear()

        self.move_group = moveit_commander.MoveGroupCommander("arm")
        self.move_group.set_planner_id("CDISTRIBUTIONRRTConfigDefault")
        # self.move_group.set_planner_id('RRTConnectkConfigDefault')
        self.move_group.set_planning_time(4.0)
        self.active_joints = self.move_group.get_active_joints()

        # set initial joint state
        joint_state_publisher = rospy.Publisher(
            "/move_group/fake_controller_joint_states", JointState, queue_size=1
        )

        # Create a JointState message
        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = INIT_JOINT_NAMES
        joint_state.position = INIT_JOINT_POSITIONS

        rate = rospy.Rate(10)
        while (
            joint_state_publisher.get_num_connections() < 1
        ):  # need to wait until the publisher is ready.
            rate.sleep()
        joint_state_publisher.publish(joint_state)

        rospy.sleep(0.5)

    def plan(
        self,
        start_configuration,
        goal_configurations_with_following_action,
        foliation_constraints,
        co_parameter,
        related_experience,
        use_atlas=False,
    ):

        goal_configurations = [i.get_intersection_motion()[0] for i in goal_configurations_with_following_action]
        motions_after_goal = [i.get_intersection_motion() for i in goal_configurations_with_following_action]
        actions_after_goal = [i.get_intersection_action() for i in goal_configurations_with_following_action]
        object_mesh_and_pose_after_goal = [i.get_object_mesh_and_pose() for i in goal_configurations_with_following_action]

        # reset the motion planner
        self.scene.clear()
        self.move_group.clear_path_constraints()
        self.move_group.clear_in_hand_pose()
        self.move_group.clear_distribution()
        
        obstacle_pose_stamped = PoseStamped()
        obstacle_pose_stamped.header.frame_id = "base_link"
        obstacle_pose_stamped.pose = foliation_constraints["obstacle_pose"]

        # add the obstacle into the planning scene.
        self.scene.add_mesh(
            "obstacle",
            obstacle_pose_stamped,
            foliation_constraints["obstacle_mesh"],
            size=(1, 1, 1)
        )

        while "obstacle" not in self.scene.get_known_object_names():
            rospy.sleep(0.0001)


        start_moveit_robot_state = convert_joint_values_to_robot_state(
            start_configuration, self.active_joints, self.robot
        )

        distribution_sequence = []

        self.move_group.set_distribution(distribution_sequence)
        self.move_group.set_clean_planning_context_flag(True)
        self.move_group.set_use_atlas_flag(use_atlas)

        # if you have object in hand, then you need to set the object in hand pose
        if "object_constraints" in foliation_constraints:

            # need to add the constraint
            manifold_constraint = construct_moveit_constraint(
                co_parameter,
                foliation_constraints["object_constraints"]["reference_pose"],
                foliation_constraints["object_constraints"]["orientation_tolerance"],
                foliation_constraints["object_constraints"]["position_tolerance"],
            )
            self.move_group.set_in_hand_pose(msgify(Pose, co_parameter))

            # add the object in the planning scene if the object is in hand.
            current_object_pose_stamped = PoseStamped()
            current_object_pose_stamped.header.frame_id = "wrist_roll_link"
            current_object_pose_stamped.pose = Pose()
            manipulated_object = make_mesh(
                "object",
                current_object_pose_stamped,
                foliation_constraints["object_mesh"]
            )
            attached_object = AttachedCollisionObject()
            attached_object.link_name = "wrist_roll_link"
            attached_object.object = manipulated_object
            attached_object.touch_links = [
                "l_gripper_finger_link",
                "r_gripper_finger_link",
                "gripper_link",
            ]
            attached_object.object.pose = msgify(Pose, co_parameter)
            start_moveit_robot_state.attached_collision_objects.append(attached_object)
        else:
            manifold_constraint = get_no_constraint()

            # add the object in the planning scene if the object is not in hand.
            object_pose_stamped = PoseStamped()
            object_pose_stamped.header.frame_id = "base_link"
            object_pose_stamped.pose = msgify(Pose, co_parameter)

            self.scene.add_mesh(
                "object",
                object_pose_stamped,
                foliation_constraints["object_mesh"],
                size=(1, 1, 1)
            )

            while "object" not in self.scene.get_known_object_names():
                rospy.sleep(0.0001)

        self.move_group.set_path_constraints(manifold_constraint)

        # set the start configuration
        self.move_group.set_start_state(start_moveit_robot_state)
        # set the goal configurations
        self.move_group.set_multi_target_robot_state(goal_configurations)

        motion_plan_result = self.move_group.plan()

        if not motion_plan_result[0]:
            return (
                False, # success flag
                None, # motion plan result
                None, # next motion
                None, # experience
                None, # manifold constraint
                None, # last configuration
            )

        last_configuration = motion_plan_result[1].joint_trajectory.points[-1].positions

        # find the index of goal_configurations with the same configuration
        goal_configuration_index = -1
        for i in range(len(goal_configurations)):
            # check if goal_configurations[i] and last_configuration are the same with for loop
            is_euqal = True
            for j in range(len(last_configuration)):
                angle1 = goal_configurations[i][j] % 6.28318530718
                angle2 = last_configuration[j] % 6.28318530718
                angle_diff = abs(angle1 - angle2)
                if min(angle_diff, 6.28318530718 - angle_diff) > 0.01:
                    is_euqal = False
                    break
            if is_euqal:
                goal_configuration_index = i
                break

        if goal_configuration_index == -1:
            raise ValueError("The last configuration is not in the goal_configurations.")

        # the section returned value should be a BaseTaskMotion
        generated_task_motion = CustomTaskMotion(
            motion_plan_result[1],
            "object_constraints" in foliation_constraints,
            msgify(Pose, co_parameter), # object pose
            foliation_constraints["object_mesh"], # object mesh path
            foliation_constraints["obstacle_pose"], # obstacle pose
            foliation_constraints["obstacle_mesh"], # obstacle mesh path
        )

        if len(motions_after_goal[goal_configuration_index]) == 0:
            next_task_motion = None
            return (
                motion_plan_result[0], # success flag
                generated_task_motion, # motion plan result
                next_task_motion, # next motion
                None, # experience
                manifold_constraint, # manifold constraint
                last_configuration # last configuration
            )
        else:
            object_mesh_during_action, object_pose_during_action = object_mesh_and_pose_after_goal[goal_configuration_index]
            next_task_motion = CustomTaskMotion(
                convert_joint_values_to_robot_trajectory(motions_after_goal[goal_configuration_index], self.active_joints),
                False,
                object_pose_during_action,
                object_mesh_during_action,
                foliation_constraints["obstacle_pose"],
                foliation_constraints["obstacle_mesh"]
            )

            return (
                motion_plan_result[0], # success flag
                generated_task_motion, # motion plan result
                next_task_motion, # next motion
                None, # experience
                manifold_constraint, # manifold constraint
                motions_after_goal[goal_configuration_index][-1] # last configuration
            )

    def shutdown_planner(self):
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)
