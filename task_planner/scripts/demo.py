#!/usr/bin/env python

import rospy
import rospkg
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
from lead_generation import LeadGeneration

import numpy as np

if __name__ == "__main__":
    rospy.init_node("demo_node", anonymous=True)

    rospack = rospkg.RosPack()

    # Get the path of the desired package
    package_path = rospack.get_path("task_planner")

    # try to slide the object on the table.

    foliation_approach_object = {
        "name": "approach_object",
        "co-parameter-type": "placement",
        "co-parameter-set": [
            np.array([[1,0,0,0.75],
                      [0,1,0,-0.55],
                      [0,0,1,0.78],
                      [0,0,0,1]]),
        ]
    }

    grasp_input = np.load(package_path + "/mesh_dir/cup.npz")
    grasp_set = [grasp_input[g] for g in grasp_input]

    foliation_slide_object = {
        "name": "slide_object",
        "co-parameter-type": "grasp",
        "co-parameter-set": grasp_set
    }

    foliation_reset_robot = {
        "name": "reset_robot",
        "co-parameter-type": "placement",
        "co-parameter-set": [
            np.array([[1,0,0,0.75],
                        [0,1,0,-0.15],
                        [0,0,1,0.78],
                        [0,0,0,1]]),
        ]
    }

    intersection_approach_object_slide_object = {
        "name": "approach_object_slide_object",
        "foliations": [foliation_approach_object, foliation_slide_object]
    }

    intersection_slide_object_reset_robot = {
        "name": "slide_object_reset_robot",
        "foliations": [foliation_slide_object, foliation_reset_robot]
    }

    lead_generator = LeadGeneration()
    lead_generator.add_foliation(foliation_approach_object)
    lead_generator.add_foliation(foliation_slide_object)
    lead_generator.add_foliation(foliation_reset_robot)
    lead_generator.add_intersection(intersection_approach_object_slide_object)