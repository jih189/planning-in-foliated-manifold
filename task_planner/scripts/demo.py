#!/usr/bin/env python

import rospy
import rospkg

import numpy as np
from custom_foliated_class import CustomFoliationConfig, custom_intersection_rule
from foliation_planning.foliated_base_class import FoliatedProblem, IntersectionRule
from foliation_planning.foliated_planning_framework import FoliatedPlanningFramework
from MTG_task_planner import MTGTaskPlanner
from custom_intersection_sampler import CustomIntersectionSampler

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
    grasp_set = [grasp_input[g] for g in grasp_input][:20]

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
        "intersection_detail": "grasp_object_in_start_placement"
    }

    intersection_slide_object_reset_robot = {
        "name": "pour_object_reset_robot",
        "foliation1": "slide_object",
        "foliation2": "reset_robot",
        "intersection_detail": "release_object_in_end_placement"
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

    foliation_problem = FoliatedProblem(
        "sliding_cup_on_desk", 
        foliation_config, 
        IntersectionRule(custom_intersection_rule)
    )
    
    foliated_planning_framework = FoliatedPlanningFramework()

    task_planner = MTGTaskPlanner()
    foliated_planning_framework.setTaskPlanner(task_planner)

    intersection_sampler = CustomIntersectionSampler()
    foliated_planning_framework.setIntersectionSampler(intersection_sampler)

    foliated_planning_framework.setFoliatedProblem(foliation_problem)

    foliated_planning_framework.setStartAndGoal(
        "approach_object",
        0,
        [-1.28, 1.51, 0.35, 1.81, 0.0, 1.47, 0.0],
        "reset_robot",
        0,
        [-1.28, 1.51, 0.35, 1.81, 0.0, 1.47, 0.0],
    )

    foliated_planning_framework.solve()