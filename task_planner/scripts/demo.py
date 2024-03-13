#!/usr/bin/env python

import rospy
import rospkg

import numpy as np
from custom_foliated_class import CustomFoliationConfig
from foliation_planning.foliated_base_class import FoliatedProblem
from foliation_planning.foliated_planning_framework import FoliatedPlanningFramework

if __name__ == "__main__":
    rospy.init_node("demo_node", anonymous=True)
    # Get the path of the desired package
    package_path = rospkg.RosPack().get_path("task_planner")

    foliation_approach_object = {
        "name": "approach_object",
        "co-parameter-type": "placement",
        "object_mesh": "cup",
        "co-parameter-set": [
            np.array([[1,0,0,0.75],
                      [0,1,0,-0.55],
                      [0,0,1,0.78],
                      [0,0,0,1]]),
        ],
        "similarity-matrix": np.identity(1)
    }

    grasp_input = np.load(package_path + "/mesh_dir/cup.npz")
    grasp_set = [grasp_input[g] for g in grasp_input]

    foliation_slide_object = {
        "name": "slide_object",
        "co-parameter-type": "grasp",
        "object_mesh": "cup",
        "object_constraints": "slide",
        "co-parameter-set": grasp_set,
        "similarity-matrix": np.identity(len(grasp_set))
    }

    foliation_reset_robot = {
        "name": "reset_robot",
        "co-parameter-type": "placement",
        "object_mesh": "cup",
        "co-parameter-set": [
            np.array([[1,0,0,0.75],
                        [0,1,0,-0.15],
                        [0,0,1,0.78],
                        [0,0,0,1]]),
        ],
        "similarity-matrix": np.identity(1)
    }

    intersection_approach_object_slide_object = {
        "name": "approach_object_slide_object",
        "foliation1": "approach_object",
        "foliation2": "slide_object", 
        "intersection_region_constraints": "grasp_object_in_start_placement"
    }

    intersection_slide_object_reset_robot = {
        "name": "pour_object_reset_robot",
        "foliation1": "slide_object",
        "foliation2": "reset_robot",
        "intersection_region_constraints": "release_object_in_end_placement"
    }

    foliation_config = CustomFoliationConfig(
        [
            foliation_approach_object, 
            foliation_slide_object,
            foliation_reset_robot
        ],[
            intersection_approach_object_slide_object, 
            intersection_slide_object_reset_robot
        ]
    )

    foliation_problem = FoliatedProblem("sliding_cup_on_desk", foliation_config)
    
    foliated_planning_framework = FoliatedPlanningFramework()

    task_planner = MTGTaskPlanner()

    foliated_planning_framework.set_task_planner(task_planner)

    foliated_planning_framework.set_foliated_problem(foliation_problem)

    foliated_planning_framework.setStartAndGoal(
        start_foliation_index,
        start_co_parameter_index,
        start_configuration,
        goal_foliation_index,
        0,
        goal_configuration,
    )