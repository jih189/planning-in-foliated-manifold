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
        "co-parameter-set": grasp_set[:10]
    }

    foliation_pour_object = {
        "name": "pour_object",
        "co-parameter-type": "grasp",
        "co-parameter-set": grasp_set[:10]
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

    intersecttion_slide_object_pour_object = {
        "name": "slide_object_pour_object",
        "foliations": [foliation_slide_object, foliation_pour_object]
    }

    intersection_pour_object_reset_robot = {
        "name": "pour_object_reset_robot",
        "foliations": [foliation_pour_object, foliation_reset_robot]
    }

    lead_generator = LeadGeneration()
    lead_generator.add_foliation(foliation_approach_object)
    lead_generator.add_foliation(foliation_slide_object)
    lead_generator.add_foliation(foliation_pour_object)
    lead_generator.add_foliation(foliation_reset_robot)

    lead_generator.add_intersection(intersection_approach_object_slide_object)
    lead_generator.add_intersection(intersecttion_slide_object_pour_object)
    lead_generator.add_intersection(intersection_pour_object_reset_robot)

    lead = lead_generator.get_lead("approach_object", 0, "reset_robot", 0)

    # add penalty to the edges
    lead_generator.add_penalty("approach_object", 0, "slide_object", 0, 1.0)

