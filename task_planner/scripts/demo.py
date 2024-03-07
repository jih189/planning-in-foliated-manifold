#!/usr/bin/env python

import rospy
import rospkg

import numpy as np

if __name__ == "__main__":
    rospy.init_node("demo_node", anonymous=True)

    rospack = rospkg.RosPack()

    # Get the path of the desired package
    package_path = rospack.get_path("task_planner")

    foliation_approach_object = {
        "name": "approach_object",
        "co-parameter-type": "placement",
        "co-parameter-set-path": "placement_path"
    }

    foliation_slide_object = {
        "name": "slide_object",
        "co-parameter-type": "grasp",
        "co-parameter-set-path": package_path + "/mesh_dir/cup.npz"
    }

    foliation_reset_robot = {
        "name": "reset_robot",
        "co-parameter-type": "placement",
        "co-parameter-set-path": "placement_path"
    }

    foliation_regrasp_object = {
        "name": "regrasp_object",
        "co-parameter-type": "placement",
        "co-parameter-set-path": "placement_path"
    }

    intersection_approach_object_slide_object = {
        "name": "approach_object_slide_object",
        "foliations": [foliation_approach_object, foliation_slide_object],
        "object_constraint": np.identity(4)
    }

    intersection_slide_object_reset_robot = {
        "name": "slide_object_reset_robot",
        "foliations": [foliation_slide_object, foliation_reset_robot],
        "object_constraint": np.identity(4)
    }

    intersection_slide_object_re_grasp_object = {
        "name": "slide_object_regrasp_object",
        "foliations": [foliation_slide_object, foliation_regrasp_object]
    }