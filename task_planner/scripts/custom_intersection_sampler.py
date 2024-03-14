from foliation_planning.foliated_base_class import (
    BaseIntersectionSampler
)

from jiaming_helper import construct_moveit_constraint

class CustomIntersectionSampler(BaseIntersectionSampler):
    def __init__(self):
        pass

    def generate_configurations_on_intersection(self, foliation1, co_parameter_1_index, foliation2, co_parameter_2_index, intersection_detail):
        """
        This function samples the intersection action from the foliated intersection.
        """
        moveit_constraint = None

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
            elif foliation1.co_parameter_type == "placement" and foliation2.co_parameter_type == "grasp":
                foliation_with_grasp_co_parameter = foliation2
                co_parameter_grasp_index = co_parameter_2_index
                foliation_with_placement_co_parameter = foliation1
                co_parameter_placement_index = co_parameter_1_index
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
        else:
            raise ValueError("The co-parameter type is not supported.")
                