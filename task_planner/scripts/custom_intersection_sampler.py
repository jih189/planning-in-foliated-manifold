import rospy
from foliation_planning.foliated_base_class import (
    BaseIntersectionSampler
)

from jiaming_helper import construct_moveit_constraint, get_joint_values_from_joint_state
from custom_foliated_class import CustomIntersection
from moveit_msgs.srv import GetJointWithConstraints, GetJointWithConstraintsRequest
import numpy as np

class CustomIntersectionSampler(BaseIntersectionSampler):
    def __init__(self, robot):
        self.robot = robot
        self.joint_names = self.robot.get_group("arm").get_joints()
        self.sample_joint_with_constraints_service = rospy.ServiceProxy('/sample_joint_with_constraints', GetJointWithConstraints)

    def generate_final_configuration(self, foliation, co_parameter_index, goal_configuration):
        """
        This function will generate a base intersection for the goal detail.
        """
        return [CustomIntersection(
            foliation.foliation_name,
            co_parameter_index,
            foliation.foliation_name,
            co_parameter_index,
            (goal_configuration, "done")
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
            placement = intersection_detail["object_constraints"]["constraint_pose"]
            moveit_constraint = construct_moveit_constraint(grasp, placement, intersection_detail["object_constraints"]["orientation_constraint"], intersection_detail["object_constraints"]["position_constraint"])
            intersection_action = "hold"
        else:
            raise ValueError("The co-parameter type is not supported.")
        
        sample_request = GetJointWithConstraintsRequest()
        sample_request.constraints = moveit_constraint
        sample_request.group_name = "arm"
        sample_request.max_sampling_attempt = 5
                
        # send constraints to the service
        response = self.sample_joint_with_constraints_service(sample_request)

        sampled_configurations = np.array([get_joint_values_from_joint_state(s.joint_state, self.joint_names) for s in response.solutions])

        if len(sampled_configurations) > 0:
            # filter out if two configurations are too close
            filtered_sampled_configurations = [sampled_configurations[0]]
            sampled_intersection_set = [
                CustomIntersection(
                    foliation1.foliation_name,
                    co_parameter_1_index,
                    foliation2.foliation_name,
                    co_parameter_2_index,
                    (sampled_configurations[0].tolist(), intersection_action)
                )
            ]
            for i in sampled_configurations[1:]:
                has_close_configuration = False
                for j in filtered_sampled_configurations:
                    if np.linalg.norm(i - j) < 0.1:
                        has_close_configuration = True
                        break
                if not has_close_configuration:
                    filtered_sampled_configurations.append(i)
                    sampled_intersection_set.append(
                        CustomIntersection(
                            foliation1.foliation_name,
                            co_parameter_1_index,
                            foliation2.foliation_name,
                            co_parameter_2_index,
                            (i.tolist(), intersection_action)
                        )
                    )
            return sampled_intersection_set
        else:
            return []
