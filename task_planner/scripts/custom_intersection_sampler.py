import rospy
from foliation_planning.foliated_base_class import (
    BaseIntersectionSampler
)

from jiaming_helper import construct_moveit_constraint, get_joint_values_from_joint_state, END_EFFECTOR_LINK, PRE_GRASP_POSE
from custom_foliated_class import CustomIntersection
from moveit_msgs.srv import GetJointWithConstraints, GetJointWithConstraintsRequest, GetCartesianPath, GetCartesianPathRequest
import numpy as np
import geometry_msgs.msg
from ros_numpy import numpify, msgify
from geometry_msgs.msg import Quaternion, Point, Pose, PoseStamped, Point32

class CustomIntersectionSampler(BaseIntersectionSampler):
    def __init__(self, robot, scene):
        self.robot = robot
        self.scene = scene
        self.joint_names = self.robot.get_group("arm").get_active_joints()
        self.sample_joint_with_constraints_service = rospy.ServiceProxy('/sample_joint_with_constraints', GetJointWithConstraints)
        self.get_cartesian_path_service = rospy.ServiceProxy('/compute_cartesian_path', GetCartesianPath)

    def generate_final_configuration(self, foliation, co_parameter_index, goal_configuration):
        """
        This function will generate a base intersection for the goal detail.
        """
        return [CustomIntersection(
            foliation.foliation_name,
            co_parameter_index,
            foliation.foliation_name,
            co_parameter_index,
            "done",
            [goal_configuration], 
            None,
            None
        )]

    def generate_configurations_on_intersection(self, foliation1, co_parameter_1_index, foliation2, co_parameter_2_index, intersection_detail):
        """
        This function samples the intersection action from the foliated intersection.
        """
        moveit_constraint = None
        intersection_action = None

        if foliation1.co_parameter_type != foliation2.co_parameter_type:
            # if one co-parameter is grasp and the other is placement, then the object 
            # constraint will be the grasp over the object placement.
            foliation_with_grasp_co_parameter = None
            co_parameter_grasp_index = None
            foliation_with_placement_co_parameter = None
            co_parameter_placement_index = None
            if foliation1.co_parameter_type == "grasp" and foliation2.co_parameter_type == "placement":
                foliation_with_grasp_co_parameter = foliation1
                co_parameter_grasp_index = co_parameter_1_index
                foliation_with_placement_co_parameter = foliation2
                co_parameter_placement_index = co_parameter_2_index
                intersection_action = "release"
            elif foliation1.co_parameter_type == "placement" and foliation2.co_parameter_type == "grasp":
                foliation_with_grasp_co_parameter = foliation2
                co_parameter_grasp_index = co_parameter_2_index
                foliation_with_placement_co_parameter = foliation1
                co_parameter_placement_index = co_parameter_1_index
                intersection_action = "grasp"
            else:
                raise ValueError("The co-parameter type is not supported.")
            
            grasp = foliation_with_grasp_co_parameter.co_parameters[co_parameter_grasp_index]
            placement = foliation_with_placement_co_parameter.co_parameters[co_parameter_placement_index]
            moveit_constraint = construct_moveit_constraint(grasp, placement, [0.001, 0.001, 0.001], [0.0001, 0.0001, 0.0001])

        elif foliation1.co_parameter_type == "grasp" and foliation2.co_parameter_type == "grasp":
            # if both co-parameters are grasp, then the object constraint should be from intersection detail.
            grasp = foliation1.co_parameters[co_parameter_1_index]
            placement = intersection_detail["object_reference_pose"]
            moveit_constraint = construct_moveit_constraint(grasp, placement, intersection_detail["object_orientation_tolerance"], intersection_detail["object_position_tolerance"])
            intersection_action = "hold"
        else:
            raise ValueError("The co-parameter type is not supported.")
        
        # set the scene with obstacle.
        self.scene.clear()

        obstacle_pose_stamped = PoseStamped()
        obstacle_pose_stamped.header.frame_id = "base_link"
        obstacle_pose_stamped.pose = msgify(Pose, intersection_detail["obstacle_pose"])

        # add the obstacle into the planning scene.
        self.scene.add_mesh(
            "obstacle",
            obstacle_pose_stamped,
            intersection_detail["obstacle_mesh"],
            size=(1, 1, 1)
        )

        while "obstacle" not in self.scene.get_known_object_names():
            rospy.sleep(0.0001)

        if intersection_action == "release" or intersection_action == "grasp":
            # need to add manipulated object during the release or grasp action
            object_pose_stamped = PoseStamped()
            object_pose_stamped.header.frame_id = "base_link"
            object_pose_stamped.pose = msgify(geometry_msgs.msg.Pose, placement)

            self.scene.add_mesh(
                "object",
                object_pose_stamped,
                intersection_detail["object_mesh"],
                size=(1, 1, 1)
            ) 

            while "object" not in self.scene.get_known_object_names():
                rospy.sleep(0.0001)
        
        sample_request = GetJointWithConstraintsRequest()
        sample_request.constraints = moveit_constraint
        sample_request.group_name = "arm"
        sample_request.max_sampling_attempt = 20
                
        # send constraints to the service
        response = self.sample_joint_with_constraints_service(sample_request)

        sampled_configurations = np.array([get_joint_values_from_joint_state(s.joint_state, self.joint_names) for s in response.solutions])
        sampled_robot_states = []

        if len(sampled_configurations) > 0:
            # filter out if two configurations are too close
            filtered_sampled_configurations = [sampled_configurations[0]]
            sampled_robot_states.append(response.solutions[0])
            for i in range(1, len(sampled_configurations)):
                has_close_configuration = False
                for j in filtered_sampled_configurations:
                    if np.linalg.norm(sampled_configurations[i] - j) < 0.1:
                        has_close_configuration = True
                        break
                if not has_close_configuration:
                    filtered_sampled_configurations.append(sampled_configurations[i])
                    sampled_robot_states.append(response.solutions[i])

            sampled_intersection_set = []
            
            if intersection_action == "release" or intersection_action == "grasp":
                pre_grasp_pose_mat = np.dot(
                    np.dot(placement, np.linalg.inv(grasp)),
                    PRE_GRASP_POSE,
                )

                pre_grasp_pose = msgify(geometry_msgs.msg.Pose, pre_grasp_pose_mat)

                # evaluate the next motion action with cartesian path
                for i in range(len(filtered_sampled_configurations)):
                    cartesian_path_request = GetCartesianPathRequest()
                    cartesian_path_request.start_state = sampled_robot_states[i]
                    cartesian_path_request.group_name = "arm"
                    cartesian_path_request.waypoints = [pre_grasp_pose]
                    cartesian_path_request.max_step = 0.005
                    cartesian_path_request.jump_threshold = 5.0
                    cartesian_path_request.avoid_collisions = True
                    cartesian_path_request.link_name = END_EFFECTOR_LINK

                    # set header
                    cartesian_path_request.header.frame_id = "base_link"
                    cartesian_path_request.header.stamp = rospy.Time.now()

                    # send the request to the service
                    cartesian_path_response = self.get_cartesian_path_service(cartesian_path_request)

                    if cartesian_path_response.fraction > 0.9:

                        intersection_motion = np.array(
                            [p.positions for p in cartesian_path_response.solution.joint_trajectory.points]
                        ).tolist()

                        if intersection_action == "release":
                            sampled_intersection_set.append(
                                CustomIntersection(
                                    foliation1.foliation_name,
                                    co_parameter_1_index,
                                    foliation2.foliation_name,
                                    co_parameter_2_index,
                                    intersection_action,
                                    intersection_motion,
                                    intersection_detail["object_mesh"],
                                    msgify(geometry_msgs.msg.Pose, placement)
                                )
                            )
                        elif intersection_action == "grasp":
                            sampled_intersection_set.append(
                                CustomIntersection(
                                    foliation1.foliation_name,
                                    co_parameter_1_index,
                                    foliation2.foliation_name,
                                    co_parameter_2_index,
                                    intersection_action,
                                    intersection_motion[::-1],
                                    intersection_detail["object_mesh"],
                                    msgify(geometry_msgs.msg.Pose, placement)
                                )
                            )
            else:
                # when the intersection action is hold
                for i in range(len(filtered_sampled_configurations)):
                    sampled_intersection_set.append(
                        CustomIntersection(
                            foliation1.foliation_name,
                            co_parameter_1_index,
                            foliation2.foliation_name,
                            co_parameter_2_index,
                            intersection_action,
                            [filtered_sampled_configurations[i].tolist()],
                            intersection_detail["object_mesh"],
                            msgify(geometry_msgs.msg.Pose, placement)
                        )
                    )         

            return sampled_intersection_set
        else:
            return []
 