import os
import rospy
from foliated_base_class import BaseIntersection, BaseTaskMotion, BaseVisualizer
from geometry_msgs.msg import Point, Pose, PoseStamped
from std_msgs.msg import ColorRGBA
import moveit_msgs.msg
import trajectory_msgs.msg
from jiaming_helper import convert_joint_values_to_robot_state, make_mesh
from visualization_msgs.msg import Marker, MarkerArray
from ros_numpy import msgify, numpify
import numpy as np
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
import random


class ManipulationTaskMotion(BaseTaskMotion):
    def __init__(
        self,
        action_name,
        planned_motion,
        has_object_in_hand,
        object_pose,
        object_mesh_path,
        obstacle_pose,
        obstacle_mesh_path,
    ):
        # if planned_motion must be trajectory_msgs/JointTrajectory.
        if not isinstance(planned_motion, moveit_msgs.msg.RobotTrajectory):
            raise TypeError("planned_motion must be trajectory_msgs/JointTrajectory.")

        self.action_name = action_name
        self.planned_motion = planned_motion
        self.has_object_in_hand = has_object_in_hand  # if the object is in hand.
        self.object_pose = object_pose  # if the object is in hand, then this is the object pose in the hand frame. if not, this is the object pose in the base_link frame.
        self.object_mesh_path = object_mesh_path
        self.obstacle_pose = obstacle_pose
        self.obstacle_mesh_path = obstacle_mesh_path

    def get(self):
        return (
            self.action_name,
            self.planned_motion,
            self.has_object_in_hand,
            self.object_pose,
            self.object_mesh_path,
            self.obstacle_pose,
            self.obstacle_mesh_path,
        )

    def cost(self):
        solution_trajectory = []
        for p in self.planned_motion.joint_trajectory.points:
            solution_trajectory.append(p.positions)
        solution_trajectory = np.array(solution_trajectory)
        differences = np.diff(solution_trajectory, axis=0)
        distances = np.linalg.norm(differences, axis=1)
        return np.sum(distances)


class MoveitVisualizer(BaseVisualizer):
    def prepare_visualizer(self, active_joints, robot):
        """
        This function prepares the visualizer.
        """
        rospy.wait_for_service("/compute_fk")
        self.compute_fk_srv = rospy.ServiceProxy("/compute_fk", GetPositionFK)

        # initialize a ros marker array publisher
        self.sampled_robot_state_publisher = rospy.Publisher(
            "sampled_robot_state", MarkerArray, queue_size=5
        )

        self.sampled_marker_ids = []

        self.task_info_publisher = rospy.Publisher(
            "task_info", MarkerArray, queue_size=5
        )

        self.task_info_marker_ids = []

        self.object_publisher = rospy.Publisher(
            "/intermediate_object", MarkerArray, queue_size=10
        )
        # this is used to display the planned path in rviz
        self.display_robot_state_publisher = rospy.Publisher(
            "/move_group/result_display_robot_state",
            moveit_msgs.msg.DisplayRobotState,
            queue_size=5,
        )
        self.active_joints = active_joints
        self.robot = robot

        self.whole_scene_marker_array = MarkerArray()

        self.manipulated_object_marker = Marker()
        self.manipulated_object_marker.header.frame_id = "base_link"
        self.manipulated_object_marker.id = 0
        self.manipulated_object_marker.type = Marker.MESH_RESOURCE
        self.manipulated_object_marker.scale = Point(1, 1, 1)
        self.manipulated_object_marker.color = ColorRGBA(0, 1, 0, 1)
        self.whole_scene_marker_array.markers.append(self.manipulated_object_marker)

        self.obstacle_marker = Marker()
        self.obstacle_marker.header.frame_id = "base_link"
        self.obstacle_marker.id = 1
        self.obstacle_marker.type = Marker.MESH_RESOURCE
        self.obstacle_marker.scale = Point(1, 1, 1)
        self.obstacle_marker.color = ColorRGBA(1, 1, 1, 1)
        self.whole_scene_marker_array.markers.append(self.obstacle_marker)

        self.current_object_pose_stamped = PoseStamped()
        self.current_object_pose_stamped.header.frame_id = "wrist_roll_link"
        self.current_object_pose_stamped.pose = Pose()

        self.attached_object = moveit_msgs.msg.AttachedCollisionObject()
        self.attached_object.link_name = "wrist_roll_link"
        self.attached_object.touch_links = [
            "l_gripper_finger_link",
            "r_gripper_finger_link",
            "gripper_link",
        ]

    def visualize_for_debug(
        self,
        sampled_configurations,
        task_constraint_parameters,
        start_configuration,
        goal_configuration,
        action_name,
        co_parameter,
    ):
        self.visualize_sampled_configurations(sampled_configurations)

        self.visualize_task_information(
            task_constraint_parameters,
            start_configuration,
            goal_configuration,
            action_name,
            co_parameter,
        )

    def get_end_effector_pose(self, configuration):
        """
        This function will return the end effector pose of the given configuration.
        """
        # convert the sampled configuration into RobotState
        current_robot_state = convert_joint_values_to_robot_state(
            configuration, self.active_joints, self.robot
        )

        # pass current robot state to compute fk service
        fk_request = GetPositionFKRequest()
        fk_request.header.frame_id = "base_link"
        fk_request.fk_link_names = ["wrist_roll_link"]
        fk_request.robot_state = current_robot_state

        fk_response = self.compute_fk_srv(fk_request)

        return numpify(fk_response.pose_stamped[0].pose)

    def generate_configuration_marker(
        self,
        configuration,
        id,
        arm_color=ColorRGBA(1, 0, 0, 0.3),
        finger_color=ColorRGBA(0, 1, 0, 0.3),
    ):
        """ """
        # convert the sampled configuration into RobotState
        current_robot_state = convert_joint_values_to_robot_state(
            configuration, self.active_joints, self.robot
        )

        # pass current robot state to compute fk service
        fk_request = GetPositionFKRequest()
        fk_request.header.frame_id = "base_link"
        fk_request.fk_link_names = self.robot.get_link_names(group="arm")
        fk_request.robot_state = current_robot_state

        fk_response = self.compute_fk_srv(fk_request)

        arm_marker = Marker()
        arm_marker.header.frame_id = "base_link"
        arm_marker.id = id
        arm_marker.type = Marker.LINE_STRIP
        arm_marker.scale = Point(0.02, 0.02, 0.02)
        arm_marker.color = arm_color
        arm_marker.points = [
            Point(p.pose.position.x, p.pose.position.y, p.pose.position.z)
            for p in fk_response.pose_stamped
        ]

        l_finger_marker = Marker()
        l_finger_marker.header.frame_id = "base_link"
        l_finger_marker.id = id + 1
        l_finger_marker.type = Marker.CUBE
        l_finger_marker.scale = Point(0.1, 0.02, 0.02)
        l_finger_marker.color = finger_color
        l_finger_marker.pose = msgify(
            Pose,
            np.dot(
                numpify(fk_response.pose_stamped[-1].pose),
                np.array(
                    [[1, 0, 0, 0.15], [0, 1, 0, 0.045], [0, 0, 1, 0], [0, 0, 0, 1]]
                ),
            ),
        )

        r_finger_marker = Marker()
        r_finger_marker.header.frame_id = "base_link"
        r_finger_marker.id = id + 2
        r_finger_marker.type = Marker.CUBE
        r_finger_marker.scale = Point(0.1, 0.02, 0.02)
        r_finger_marker.color = finger_color
        r_finger_marker.pose = msgify(
            Pose,
            np.dot(
                numpify(fk_response.pose_stamped[-1].pose),
                np.array(
                    [[1, 0, 0, 0.15], [0, 1, 0, -0.045], [0, 0, 1, 0], [0, 0, 0, 1]]
                ),
            ),
        )

        return arm_marker, l_finger_marker, r_finger_marker

    def visualize_task_information(
        self,
        task_constraint_parameters,
        start_configuration,
        goal_configuration,
        action_name,
        co_parameter,
    ):
        # visualize the task information

        # clean previous marker
        delete_marker_arrary = MarkerArray()
        for i in self.task_info_marker_ids:
            delete_marker = Marker()
            delete_marker.header.frame_id = "base_link"
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            delete_marker_arrary.markers.append(delete_marker)
        self.task_info_publisher.publish(delete_marker_arrary)

        marker_array = MarkerArray()

        # visualize the start configuration
        (
            start_arm_marker,
            start_l_finger_marker,
            start_r_finger_marker,
        ) = self.generate_configuration_marker(
            start_configuration,
            0,
            arm_color=ColorRGBA(0, 0, 1, 1),
            finger_color=ColorRGBA(0, 0, 1, 1),
        )
        marker_array.markers.append(start_arm_marker)
        marker_array.markers.append(start_l_finger_marker)
        marker_array.markers.append(start_r_finger_marker)

        self.task_info_marker_ids.append(start_arm_marker.id)
        self.task_info_marker_ids.append(start_l_finger_marker.id)
        self.task_info_marker_ids.append(start_r_finger_marker.id)

        # visualize the goal configuration
        (
            goal_arm_marker,
            goal_l_finger_marker,
            goal_r_finger_marker,
        ) = self.generate_configuration_marker(
            goal_configuration,
            3,
            arm_color=ColorRGBA(0, 0, 1, 1),
            finger_color=ColorRGBA(0, 0, 1, 1),
        )
        marker_array.markers.append(goal_arm_marker)
        marker_array.markers.append(goal_l_finger_marker)
        marker_array.markers.append(goal_r_finger_marker)

        self.task_info_marker_ids.append(goal_arm_marker.id)
        self.task_info_marker_ids.append(goal_l_finger_marker.id)
        self.task_info_marker_ids.append(goal_r_finger_marker.id)

        # visualize the action name
        action_name_marker = Marker()
        action_name_marker.header.frame_id = "base_link"
        action_name_marker.id = 6
        action_name_marker.type = Marker.TEXT_VIEW_FACING
        action_name_marker.scale = Point(0.15, 0.15, 0.15)
        action_name_marker.color = ColorRGBA(1, 1, 1, 1)
        action_name_marker.text = action_name
        action_name_marker.pose.position = Point(0, 0, 2.0)
        marker_array.markers.append(action_name_marker)
        self.task_info_marker_ids.append(action_name_marker.id)

        # add the object marker based on the task_constraint_parameters
        if task_constraint_parameters["is_object_in_hand"]:
            # get end effector pose
            start_end_effector_pose = self.get_end_effector_pose(start_configuration)
            # add the object marker
            start_object_marker = Marker()
            start_object_marker.header.frame_id = "base_link"
            start_object_marker.id = 7
            start_object_marker.type = Marker.MESH_RESOURCE
            start_object_marker.scale = Point(1, 1, 1)
            start_object_marker.color = ColorRGBA(0, 1, 1, 1)
            start_object_marker.mesh_resource = (
                "package://task_planner/mesh_dir/"
                + os.path.basename(task_constraint_parameters["object_mesh_path"])
            )
            start_object_marker.pose = msgify(
                Pose, np.dot(start_end_effector_pose, np.linalg.inv(co_parameter))
            )
            marker_array.markers.append(start_object_marker)

            self.task_info_marker_ids.append(start_object_marker.id)

            # get end effector pose
            goal_end_effector_pose = self.get_end_effector_pose(goal_configuration)
            # add the object marker
            goal_object_marker = Marker()
            goal_object_marker.header.frame_id = "base_link"
            goal_object_marker.id = 8
            goal_object_marker.type = Marker.MESH_RESOURCE
            goal_object_marker.scale = Point(1, 1, 1)
            goal_object_marker.color = ColorRGBA(0, 1, 1, 1)
            goal_object_marker.mesh_resource = (
                "package://task_planner/mesh_dir/"
                + os.path.basename(task_constraint_parameters["object_mesh_path"])
            )
            goal_object_marker.pose = msgify(
                Pose, np.dot(goal_end_effector_pose, np.linalg.inv(co_parameter))
            )
            marker_array.markers.append(goal_object_marker)

            self.task_info_marker_ids.append(goal_object_marker.id)

        else:
            object_marker = Marker()
            object_marker.header.frame_id = "base_link"
            object_marker.id = 7
            object_marker.type = Marker.MESH_RESOURCE
            object_marker.scale = Point(1, 1, 1)
            object_marker.color = ColorRGBA(0, 1, 1, 1)
            object_marker.mesh_resource = (
                "package://task_planner/mesh_dir/"
                + os.path.basename(task_constraint_parameters["object_mesh_path"])
            )
            object_marker.pose = msgify(Pose, co_parameter)
            marker_array.markers.append(object_marker)

            self.task_info_marker_ids.append(object_marker.id)

        # add the obstacle marker based on the task_constraint_parameters
        obstacle_marker = Marker()
        obstacle_marker.header.frame_id = "base_link"
        obstacle_marker.id = 9
        obstacle_marker.type = Marker.MESH_RESOURCE
        obstacle_marker.scale = Point(1, 1, 1)
        obstacle_marker.color = ColorRGBA(1, 1, 1, 1)
        obstacle_marker.mesh_resource = (
            "package://task_planner/mesh_dir/"
            + os.path.basename(task_constraint_parameters["obstacle_mesh"])
        )
        obstacle_marker.pose = msgify(Pose, task_constraint_parameters["obstacle_pose"])
        marker_array.markers.append(obstacle_marker)

        self.task_info_marker_ids.append(obstacle_marker.id)

        self.task_info_publisher.publish(marker_array)

    def visualize_sampled_configurations(self, sampled_configurations):
        # use the moveit visualizer to visualize the sampled configurations
        # sampled_configurations is a list pair of configurations and status

        # if number of sampled configuration is more than 100, then only visualize 100 of them randomly
        if len(sampled_configurations) > 100:
            sampled_configurations = random.sample(sampled_configurations, 100)

        delete_marker_arrary = MarkerArray()
        for i in self.sampled_marker_ids:
            delete_marker = Marker()
            delete_marker.header.frame_id = "base_link"
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            delete_marker_arrary.markers.append(delete_marker)
        self.sampled_robot_state_publisher.publish(delete_marker_arrary)

        # create a marker array
        marker_array = MarkerArray()

        for t, (c, s) in enumerate(sampled_configurations):
            if s == 5:
                (
                    arm_marker,
                    l_finger_marker,
                    r_finger_marker,
                ) = self.generate_configuration_marker(
                    c,
                    3 * t,
                    arm_color=ColorRGBA(1, 1, 1, 0.3),
                    finger_color=ColorRGBA(1, 1, 1, 0.3),
                )
            elif s > 5:
                (
                    arm_marker,
                    l_finger_marker,
                    r_finger_marker,
                ) = self.generate_configuration_marker(
                    c,
                    3 * t,
                    arm_color=ColorRGBA(0, 0, 0, 0.3),
                    finger_color=ColorRGBA(0, 0, 0, 0.3),
                )
            elif s == 2:
                (
                    arm_marker,
                    l_finger_marker,
                    r_finger_marker,
                ) = self.generate_configuration_marker(
                    c,
                    3 * t,
                    arm_color=ColorRGBA(0.5, 0.5, 0, 0.3),
                    finger_color=ColorRGBA(0.5, 0.5, 0, 0.3),
                )
            elif s == 0:
                (
                    arm_marker,
                    l_finger_marker,
                    r_finger_marker,
                ) = self.generate_configuration_marker(
                    c,
                    3 * t,
                    arm_color=ColorRGBA(0, 1, 0, 0.3),
                    finger_color=ColorRGBA(0, 1, 0, 0.3),
                )
            else:
                (
                    arm_marker,
                    l_finger_marker,
                    r_finger_marker,
                ) = self.generate_configuration_marker(
                    c,
                    3 * t,
                    arm_color=ColorRGBA(1, 0, 0, 0.3),
                    finger_color=ColorRGBA(1, 0, 0, 0.3),
                )
            # arm_marker, l_finger_marker, r_finger_marker = self.generate_configuration_marker(c, 3*t)
            marker_array.markers.append(arm_marker)
            marker_array.markers.append(l_finger_marker)
            marker_array.markers.append(r_finger_marker)
            self.sampled_marker_ids.append(arm_marker.id)
            self.sampled_marker_ids.append(l_finger_marker.id)
            self.sampled_marker_ids.append(r_finger_marker.id)

        self.sampled_robot_state_publisher.publish(marker_array)

    def visualize_plan(self, list_of_motion_plan):
        """
        This function will receive a list of motion plan and visualize it.
        One thing must be considered is that this list contains both motion between intersections and
        intersection. Therefore, the visuliaztion must handle this situation properly.
        """
        print("visualize the plan")
        print("press ctrl+c to exit")

        need_to_break = False

        is_gripper_open = True 
        gripper_joint_names = ["l_gripper_finger_joint", "r_gripper_finger_joint"]

        while not rospy.is_shutdown():
            for motion_plan in list_of_motion_plan:
                (
                    action_name,
                    motion_trajectory,
                    has_object_in_hand,
                    object_pose,
                    object_mesh_path,
                    obstacle_pose,
                    obstacle_mesh_path,
                ) = motion_plan.get()

                # if the action is release, then we need to visualize the release motion
                if action_name == "release":
                    is_gripper_open = True

                if obstacle_pose is not None:
                    self.obstacle_marker.mesh_resource = (
                        "package://task_planner/mesh_dir/"
                        + os.path.basename(obstacle_mesh_path)
                    )
                    self.obstacle_marker.action = Marker.ADD
                    self.obstacle_marker.pose = msgify(Pose, obstacle_pose)

                for p in motion_trajectory.joint_trajectory.points:
                    current_robot_state_msg = moveit_msgs.msg.DisplayRobotState()
                    # current_robot_state_msg.state = convert_joint_values_to_robot_state(
                    #     p.positions, self.active_joints, self.robot
                    # )

                    # based on the is_gripper_open, we need to change the gripper states
                    if is_gripper_open:
                        gripper_positions = [0.04, 0.04]
                    else:
                        gripper_positions = [0.0, 0.0]
                    current_robot_state_msg.state = convert_joint_values_to_robot_state(
                        list(p.positions) + gripper_positions, self.active_joints + gripper_joint_names, self.robot
                    )

                    # if not manipulated object, then does not need to publish the object
                    if object_pose is not None:
                        if has_object_in_hand:
                            self.attached_object.object = make_mesh(
                                "object",
                                self.current_object_pose_stamped,
                                object_mesh_path,
                            )
                            self.attached_object.object.pose = msgify(
                                Pose, np.linalg.inv(object_pose)
                            )
                            current_robot_state_msg.state.attached_collision_objects.append(
                                self.attached_object
                            )
                            self.manipulated_object_marker.action = Marker.DELETE
                        else:
                            self.manipulated_object_marker.mesh_resource = (
                                "package://task_planner/mesh_dir/"
                                + os.path.basename(object_mesh_path)
                            )
                            self.manipulated_object_marker.action = Marker.ADD
                            self.manipulated_object_marker.pose = msgify(
                                Pose, object_pose
                            )

                        self.object_publisher.publish(self.whole_scene_marker_array)

                    self.display_robot_state_publisher.publish(current_robot_state_msg)

                    # rospy sleep
                    rospy.sleep(0.03)

                    if rospy.is_shutdown():
                        need_to_break = True
                        break
                if need_to_break:
                    break
                # if the action is grasp, then we need to visualize the grasp motion
                if action_name == "grasp":
                    is_gripper_open = False

            if need_to_break:
                break
