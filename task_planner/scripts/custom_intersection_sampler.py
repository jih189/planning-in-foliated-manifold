from foliation_planning.foliated_base_class import (
    BaseIntersectionSampler
)

class CustomIntersectionSampler(BaseIntersectionSampler):
    def __init__(self):
        pass

    def generate_configurations_on_intersection(self, foliation1, co_parameter_1_index, foliation2, co_parameter_2_index, intersection_detail):
        """
        This function samples the intersection action from the foliated intersection.
        """
        print "sample intersection"
        print "foliation1: ", foliation1.foliation_name
        print "co_parameter_1_index: ", co_parameter_1_index
        print "foliation2: ", foliation2.foliation_name
        print "co_parameter_2_index: ", co_parameter_2_index
        print "intersection_detail: ", intersection_detail
        return [0,0,0,0,0,0,0]