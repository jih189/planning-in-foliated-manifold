from foliation_planning.foliated_base_class import (
        FoliationConfig, 
        BaseIntersection, 
        BaseFoliation,
        FoliatedIntersection
    )

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
            intersection_action # the intersection action is a tuple (arm configuration, grasp or release)
        ):
        super(CustomIntersection, self).__init__(foliation1_name, co_parameter1_index, foliation2_name, co_parameter2_index, intersection_action)

    def get_intersection_action(self):
        return self.intersection_action[1]

    def get_edge_configurations(self):
        return [self.intersection_action[0], self.intersection_action[0]]

    def inverse_action(self):
        if self.intersection_action[1] == "grasp":
            return (self.intersection_action[0], "release")
        else:
            return (self.intersection_action[0], "grasp")

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
                "object_constraints": foliation["object_constraints"]
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
            }
            return CustomFoliation(
                    foliation["name"],
                    constraint_parameters,
                    foliation["co-parameter-set"],
                    foliation["similarity-matrix"],
                    foliation["co-parameter-type"]
                )

    def load_foliated_intersection(self, intersection):

        try:
            foliation1 = self.foliation_set[intersection["foliation1"]]
            foliation2 = self.foliation_set[intersection["foliation2"]]
        except: # catch dictionary key error
            print("Error: Foliation not found")

        return FoliatedIntersection(
            intersection["name"],
            foliation1,
            foliation2,
            intersection["intersection_region_constraints"]
        )