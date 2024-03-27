#!/usr/bin/env python
import os

import rospy
import rospkg

import numpy as np
import random
from custom_foliated_class import CustomFoliationConfig, custom_intersection_rule
from foliation_planning.foliated_base_class import FoliatedProblem, IntersectionRule
from foliation_planning.foliated_planning_framework import FoliatedPlanningFramework
from MTG_task_planner import MTGTaskPlanner
from jiaming_motion_planner import MoveitMotionPlanner
from custom_intersection_sampler import CustomIntersectionSampler
from jiaming_helper import generate_similarity_matrix, FETCH_GRIPPER_ROTATION

from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
from sensor_msgs.msg import JointState
from custom_visualizer import MoveitVisualizer

from visualization_msgs.msg import Marker, MarkerArray
from ros_numpy import numpify, msgify
from geometry_msgs.msg import Quaternion, Point, Pose, PoseStamped, Point32
from std_msgs.msg import ColorRGBA

def get_position_difference_between_poses(pose_1_, pose_2_):
    """
    Get the position difference between two poses.
    pose_1_ and pose_2_ are both 4x4 numpy matrices.
    """
    return np.linalg.norm(pose_1_[:3, 3] - pose_2_[:3, 3])

if __name__ == "__main__":
    rospy.init_node("demo_node", anonymous=True)
    # Get the path of the desired package
    package_path = rospkg.RosPack().get_path("task_planner")

    obstacle_mesh_path = package_path + "/mesh_dir/desk.stl"

    obstacle_pose = Pose()
    obstacle_pose.position.x = 0.51
    obstacle_pose.position.y = 0.05
    obstacle_pose.position.z = -0.02
    obstacle_pose.orientation.x = 0
    obstacle_pose.orientation.y = 0
    obstacle_pose.orientation.z = 0.707
    obstacle_pose.orientation.w = 0.707

    manipulated_object_mesh_path = package_path + "/mesh_dir/cup.stl"

    # problem_publisher = rospy.Publisher(
    #     "/problem_visualization_marker_array", MarkerArray, queue_size=5
    # )

    start_object_pose = np.array([[1,0,0,0.65],
                                  [0,1,0,-0.55],
                                  [0,0,1,0.78],
                                  [0,0,0,1]])

    goal_object_pose = np.array([[1,0,0,0.65],
                                [0,1,0,-0.15],
                                [0,0,1,0.78],
                                [0,0,0,1]])


    # manipulated_object_mesh_path = package_path + "/mesh_dir/cup.stl"

    # # visualize both start and goal object placements
    # marker_array = MarkerArray()

    # object_marker = Marker()
    # object_marker.header.frame_id = "base_link"
    # object_marker.header.stamp = rospy.Time.now()
    # object_marker.ns = "placement"
    # object_marker.id = 0
    # object_marker.type = Marker.MESH_RESOURCE
    # object_marker.action = Marker.ADD
    # object_marker.pose = msgify(Pose, start_object_pose)
    # object_marker.scale = Point(1, 1, 1)
    # object_marker.color = ColorRGBA(1.0, 0.5, 0.5, 1)
    # object_marker.mesh_resource = (
    #     "package://task_planner/mesh_dir/"
    #     + os.path.basename(manipulated_object_mesh_path)
    # )
    # marker_array.markers.append(object_marker)

    # object_marker.id = 1
    # object_marker.pose = msgify(Pose, goal_object_pose)
    # object_marker.color = ColorRGBA(0.5, 1.0, 0.5, 1)
    # marker_array.markers.append(object_marker)


    ########################################################################

    table_top_pose = np.array(
        [[1, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 0.78], [0, 0, 0, 1]]
    )

    foliation_approach_object = {
        "name": "approach_object",
        "co-parameter-type": "placement",
        "object_mesh": "cup",
        "co-parameter-set": [
            start_object_pose
        ],
        "similarity-matrix": np.identity(1),
        "obstacle_pose": obstacle_pose, 
        "obstacle_mesh": obstacle_mesh_path
    }

    loaded_array = np.load(package_path + "/mesh_dir/cup.npz")
    grasp_set = [np.dot(loaded_array[loaded_array.files[ind]], FETCH_GRIPPER_ROTATION) for ind in random.sample(list(range(len(loaded_array.files))), 100)]

    foliation_slide_object_similarity_matrix = generate_similarity_matrix(grasp_set, get_position_difference_between_poses)
    grasp_inv_set = [np.linalg.inv(g) for g in grasp_set]

    foliation_slide_object = {
        "name": "slide_object",
        "co-parameter-type": "grasp",
        "object_mesh": "cup",
        "object_constraints": {
            "frame_id": "base_link",
            "reference_pose": table_top_pose,
            "orientation_tolerance": [0.001, 0.001, 0.001],
            "position_tolerance": np.array([2000, 2000, 0.0008]),
        },
        "co-parameter-set": grasp_inv_set,
        "similarity-matrix": foliation_slide_object_similarity_matrix,
        "obstacle_pose": obstacle_pose, 
        "obstacle_mesh": obstacle_mesh_path
    }

    foliation_reset_robot = {
        "name": "reset_robot",
        "co-parameter-type": "placement",
        "object_mesh": "cup",
        "co-parameter-set": [goal_object_pose],
        "similarity-matrix": np.identity(1),
        "obstacle_pose": obstacle_pose, 
        "obstacle_mesh": obstacle_mesh_path
    }

    intersection_approach_object_slide_object = {
        "name": "approach_object_slide_object",
        "foliation1": "approach_object",
        "foliation2": "slide_object", 
        "intersection_detail": {},
        "obstacle_pose": obstacle_pose, 
        "obstacle_mesh": obstacle_mesh_path
    }

    intersection_slide_object_reset_robot = {
        "name": "pour_object_reset_robot",
        "foliation1": "slide_object",
        "foliation2": "reset_robot",
        "intersection_detail": {},
        "obstacle_pose": obstacle_pose, 
        "obstacle_mesh": obstacle_mesh_path
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

    motion_planner = MoveitMotionPlanner()

    intersection_sampler = CustomIntersectionSampler(motion_planner.robot)
    foliated_planning_framework.setIntersectionSampler(intersection_sampler)

    foliated_planning_framework.setMotionPlanner(motion_planner)

    foliated_planning_framework.setFoliatedProblem(foliation_problem)

    foliated_planning_framework.setStartAndGoal(
        "approach_object",
        0,
        [-1.28, 1.51, 0.35, 1.81, 0.0, 1.47, 0.0],
        "reset_robot",
        0,
        [-1.28, 1.51, 0.35, 1.81, 0.0, 1.47, 0.0],
    )

    planned_solution = foliated_planning_framework.solve()

    print("length of planned solution")
    print(len(planned_solution))

    if len(planned_solution) > 0:
        print("Planned solution is found.")
        visualizer = MoveitVisualizer()
        visualizer.prepare_visualizer(
            motion_planner.active_joints,
            motion_planner.robot
        )
        visualizer.visualize_plan(planned_solution)
    else:
        print("No solution is found.")