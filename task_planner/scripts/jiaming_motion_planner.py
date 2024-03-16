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
)
from foliation_planning.foliated_base_class import BaseMotionPlanner


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
        self.move_group.set_planning_time(1.0)

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
        joint_state.position = [0.1, -1.28, 1.52, 0.35, 1.81, 1.47, 0.04, 0.04]

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
        goal_configuration,
        foliation_constraints,
        co_parameter,
        related_experience,
        use_atlas=False,
    ):
        # reset the motion planner
        self.scene.clear()
        self.move_group.clear_path_constraints()
        self.move_group.clear_in_hand_pose()
        self.move_group.clear_distribution()

        obstacle_pose_stamped = PoseStamped()
        obstacle_pose_stamped.header.frame_id = "base_link"
        obstacle_pose_stamped.pose = msgify(
            Pose, foliation_constraints["obstacle_pose"]
        )

        # add the obstacle into the planning scene.
        self.scene.add_mesh(
            "obstacle",
            obstacle_pose_stamped,
            foliation_constraints["obstacle_mesh"],
            size=(1, 1, 1),
        )

        # wait for the obstacle to be added into the planning scene
        while "obstacle" not in self.scene.get_known_object_names():
            rospy.sleep(0.0001)

        start_moveit_robot_state = convert_joint_values_to_robot_state(
            start_configuration, self.move_group.get_active_joints(), self.robot
        )

        distribution_sequence = []

        for node_id, node_distribution, related_node_data in related_experience:
            distribution = SamplingDistribution()
            distribution.distribution_mean = node_distribution.mean.tolist()
            distribution.distribution_convariance = (
                node_distribution.covariance.flatten().tolist()
            )
            distribution.foliation_id = node_id[0]
            distribution.co_parameter_id = node_id[1]
            distribution.distribution_id = node_id[2]
            distribution.related_co_parameter_index = []
            distribution.related_beta = []
            distribution.related_similarity = []
            for (
                related_co_parameter_index,
                related_beta,
                related_similarity,
            ) in related_node_data:
                distribution.related_co_parameter_index.append(
                    related_co_parameter_index
                )
                distribution.related_beta.append(related_beta)
                distribution.related_similarity.append(related_similarity)

            distribution_sequence.append(distribution)

        self.move_group.set_distribution(distribution_sequence)
        self.move_group.set_clean_planning_context_flag(True)
        self.move_group.set_use_atlas_flag(use_atlas)

        # if you have object in hand, then you need to set the object in hand pose
        if foliation_constraints["is_object_in_hand"]:
            current_object_pose_stamped = PoseStamped()
            current_object_pose_stamped.header.frame_id = "wrist_roll_link"
            current_object_pose_stamped.pose = Pose()
            manipulated_object = make_mesh(
                "object",
                current_object_pose_stamped,
                foliation_constraints["object_mesh_path"],
            )

            attached_object = AttachedCollisionObject()
            attached_object.link_name = "wrist_roll_link"
            attached_object.object = manipulated_object
            attached_object.touch_links = [
                "l_gripper_finger_link",
                "r_gripper_finger_link",
                "gripper_link",
            ]
            attached_object.object.pose = msgify(Pose, np.linalg.inv(co_parameter))
            start_moveit_robot_state.attached_collision_objects.append(attached_object)

            # need to add the constraint
            manifold_constraint = construct_moveit_constraint(
                np.linalg.inv(co_parameter),
                foliation_constraints["reference_pose"],
                foliation_constraints["orientation_tolerance"],
                foliation_constraints["position_tolerance"],
            )
            self.move_group.set_in_hand_pose(msgify(Pose, np.linalg.inv(co_parameter)))
            # self.move_group.set_path_constraints(manifold_constraint)

        else:
            # becasuse object is not in hand, so we need to add the object into the planning scene
            current_object_pose_stamped = PoseStamped()
            current_object_pose_stamped.header.frame_id = "base_link"
            current_object_pose_stamped.pose = msgify(Pose, co_parameter)
            self.scene.add_mesh(
                "object",
                current_object_pose_stamped,
                foliation_constraints["object_mesh_path"],
                size=(1, 1, 1),
            )

            while "object" not in self.scene.get_known_object_names():
                rospy.sleep(0.0001)

            manifold_constraint = get_no_constraint()

        self.move_group.set_path_constraints(manifold_constraint)

        # set the start configuration
        self.move_group.set_start_state(start_moveit_robot_state)
        self.move_group.set_joint_value_target(goal_configuration)

        motion_plan_result = self.move_group.plan()

        # the section returned value should be a BaseTaskMotion
        return (
            motion_plan_result[0],
            None,
            motion_plan_result,
            manifold_constraint,
        )

    def shutdown_planner(self):
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)
