from foliation_planning.foliated_base_class import (
        FoliationConfig, 
        BaseIntersection, 
        BaseFoliation,
        FoliatedIntersection,
        BaseTaskMotion
    )

import moveit_msgs.msg

"""
CustomFoliation class
"""
class CustomFoliation(BaseFoliation):
    """
    In this custom class, we do not need to modify anything.
    """
    def __init__(self, foliation_name, constraint_parameters, co_parameters, similarity_matrix, co_parameter_type):
        super(CustomFoliation, self).__init__(foliation_name, constraint_parameters, co_parameters, similarity_matrix)
        self.co_parameter_type = co_parameter_type

"""
CustomIntersection class
"""
class CustomIntersection(BaseIntersection):
    """
    In this custom class, we do not need to modify anything.
    """
    def __init__(
            self, 
            foliation1_name, 
            co_parameter1_index, 
            foliation2_name, 
            co_parameter2_index, 
            intersection_action,
            intersection_motion, 
            object_mesh_during_action,
            object_pose_during_action
        ):
        self.object_mesh_during_action = object_mesh_during_action
        self.object_pose_during_action = object_pose_during_action
        super(CustomIntersection, self).__init__(foliation1_name, co_parameter1_index, foliation2_name, co_parameter2_index, intersection_action, intersection_motion)

    def get_intersection_action(self):
        return self.intersection_action

    def get_intersection_motion(self):
        return self.intersection_motion

    def get_object_mesh_and_pose(self):
        return self.object_mesh_during_action, self.object_pose_during_action

    def inverse_action(self):
        if self.intersection_action == "grasp":
            return "release"
        elif self.intersection_action[1] == "release":
            return "grasp"
        elif self.intersection_action[1] == "hold":
            return "hold"
        else:
            raise ValueError("The intersection action is not supported.")

    def inverse_motion(self):
        return self.intersection_motion[::-1]

"""
CustomFoliationConfig class
"""
class CustomFoliationConfig(FoliationConfig):
    def __init__(self, foliation_set, foliated_intersection_set):
        super(CustomFoliationConfig, self).__init__(foliation_set, foliated_intersection_set)

    def load_foliation(self, foliation):
        if foliation["co-parameter-type"] == "grasp":
            # if the co-parameter-type is grasp, the constraint parameters should
            # include the object mesh and the object constraints during action.
            constraint_parameters = {
                "object_mesh": foliation["object_mesh"],
                "object_constraints": foliation["object_constraints"],
                "obstacle_pose": foliation["obstacle_pose"],
                "obstacle_mesh": foliation["obstacle_mesh"]
            }
            return CustomFoliation(
                    foliation["name"],
                    constraint_parameters,
                    foliation["co-parameter-set"],
                    foliation["similarity-matrix"],
                    foliation["co-parameter-type"]
                )
        else:
            # if the co-parameter-type is placement, the constraint parameters should
            # include the object mesh.
            constraint_parameters = {
                "object_mesh": foliation["object_mesh"],
                "obstacle_pose": foliation["obstacle_pose"],
                "obstacle_mesh": foliation["obstacle_mesh"]
            }
            return CustomFoliation(
                    foliation["name"],
                    constraint_parameters,
                    foliation["co-parameter-set"],
                    foliation["similarity-matrix"],
                    foliation["co-parameter-type"]
                )

    def load_foliated_intersection(self, intersection):

        return FoliatedIntersection(
            intersection["name"],
            intersection["foliation1"],
            intersection["foliation2"],
            intersection["intersection_detail"]
        )

def custom_intersection_rule(foliation1, foliation2):
    """
    This function is used to find the connected co-parameter sets between two foliations.
    When the co-parameter type of the two foliations are the same, the structure between two foliations
    should be parallel(one to one). When the co-parameter type of the two foliations are different,
    the structure between two foliations should be cross(many to many).
    """
    if foliation1.co_parameter_type == foliation2.co_parameter_type:
        # check if they have the same co-parameter set
        if len(foliation1.co_parameters) != len(foliation2.co_parameters):
            raise ValueError("Foliations must have the same number of co-parameter sets")
        
        result = []
        for i in range(len(foliation1.co_parameters)):
            if not np.allclose(foliation1.co_parameters[i], foliation2.co_parameters[i]):
                raise ValueError("Foliations must have the same co-parameter sets")
            result.append((i, i))
        return result
    else:
        result = []
        # add edge from each manifold from foliation 1 to each manifold from foliation 2
        for i in range(len(foliation1.co_parameters)):
            for j in range(len(foliation2.co_parameters)):
                result.append((i, j))
        return result

class CustomTaskMotion(BaseTaskMotion):
    '''
    This class is used to store the motion plan result and passed to the visualizer.
    '''
    def __init__(
        self,
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

        self.planned_motion = planned_motion
        self.has_object_in_hand = has_object_in_hand  # if the object is in hand.
        self.object_pose = object_pose  # if the object is in hand, then this is the object pose in the hand frame. if not, this is the object pose in the base_link frame.
        self.object_mesh_path = object_mesh_path
        self.obstacle_pose = obstacle_pose
        self.obstacle_mesh_path = obstacle_mesh_path

    def get(self):
        return (
            self.planned_motion,
            self.has_object_in_hand,
            self.object_pose,
            self.object_mesh_path,
            self.obstacle_pose,
            self.obstacle_mesh_path,
        )