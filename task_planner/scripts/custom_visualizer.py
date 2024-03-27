import os
import rospy
from foliation_planning.foliated_base_class import BaseVisualizer
from geometry_msgs.msg import Point, Pose, PoseStamped
from std_msgs.msg import ColorRGBA
import moveit_msgs.msg
from visualization_msgs.msg import Marker, MarkerArray
from ros_numpy import msgify, numpify
import numpy as np

from jiaming_helper import convert_joint_values_to_robot_state



class MoveitVisualizer(BaseVisualizer):
    def prepare_visualizer(self, active_joints, robot):
        """
        This function prepares the visualizer.
        """
        self.sampled_marker_ids = []

        # initialize a ros marker array publisher
        self.sampled_robot_state_publisher = rospy.Publisher(
            "sampled_robot_state", MarkerArray, queue_size=5
        )

        self.obstacle_publisher = rospy.Publisher(
            "/obstacle_marker", MarkerArray, queue_size=5
        )

        # this is used to display the planned path in rviz
        self.display_robot_state_publisher = rospy.Publisher(
            "/move_group/result_display_robot_state",
            moveit_msgs.msg.DisplayRobotState,
            queue_size=5,
        )
        self.active_joints = active_joints
        self.robot = robot

    def visualize_plan(self, plan):
        need_to_break = False
        is_gripper_open = True
        gripper_joint_names = ["l_gripper_finger_joint", "r_gripper_finger_joint"]

        while not rospy.is_shutdown():
            for task_motion in plan:
                if task_motion is None:
                    continue
                (
                    motion_trajectory,
                    has_object_in_hand,
                    object_pose,
                    object_mesh_path,
                    obstacle_pose,
                    obstacle_mesh_path,
                ) = task_motion.get()

                obstacle_marker_array = MarkerArray()
                obstacle_marker = Marker()
                obstacle_marker.header.frame_id = "base_link"
                obstacle_marker.id = 0
                obstacle_marker.type = Marker.MESH_RESOURCE
                obstacle_marker.mesh_resource = (
                    "package://task_planner/mesh_dir/"
                    + os.path.basename(obstacle_mesh_path)
                )
                obstacle_marker.pose = obstacle_pose
                obstacle_marker.scale = Point(1, 1, 1)
                obstacle_marker.color = ColorRGBA(0.5, 0.5, 0.5, 1)
                obstacle_marker_array.markers.append(obstacle_marker)

                for p in motion_trajectory.joint_trajectory.points:
                    current_robot_state_msg = moveit_msgs.msg.DisplayRobotState()

                    if has_object_in_hand:
                        gripper_positions = [0.0, 0.0]
                    else:
                        gripper_positions = [0.04, 0.04]

                    current_robot_state_msg.state = convert_joint_values_to_robot_state(
                        list(p.positions) + gripper_positions, self.active_joints + gripper_joint_names, self.robot
                    )

                    self.display_robot_state_publisher.publish(current_robot_state_msg)
                    self.obstacle_publisher.publish(obstacle_marker_array)

                    rospy.sleep(0.03)

                    if rospy.is_shutdown():
                        need_to_break = True
                        break
                if need_to_break:
                    break
            if need_to_break:
                break

    def generate_configuration_marker(
        self,
        current_robot_state,
        id,
        arm_color=ColorRGBA(1, 0, 0, 0.3),
        finger_color=ColorRGBA(0, 1, 0, 0.3),
    ):

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

    def visualize_sampled_configurations(self, sampled_state_spaces):
        # use the moveit visualizer to visualize the sampled configurations
        # sampled_state_spaces is a list pair of configurations and status

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

        for t, (c, s) in enumerate(sampled_state_spaces):
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
