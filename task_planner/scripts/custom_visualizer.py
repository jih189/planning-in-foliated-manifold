import os
import rospy
from foliation_planning.foliated_base_class import BaseVisualizer
from geometry_msgs.msg import Point, Pose, PoseStamped
from std_msgs.msg import ColorRGBA
import moveit_msgs.msg
from visualization_msgs.msg import Marker, MarkerArray
from ros_numpy import msgify, numpify
import numpy as np

class MoveitVisualizer(BaseVisualizer):
    def prepare_visualizer(self):
        """
        This function prepares the visualizer.
        """
        self.sampled_marker_ids = []

        # initialize a ros marker array publisher
        self.sampled_robot_state_publisher = rospy.Publisher(
            "sampled_robot_state", MarkerArray, queue_size=5
        )

    def visualize_plan(self, plan):
        pass

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
