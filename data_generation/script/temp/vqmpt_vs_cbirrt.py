#!/usr/bin/env python

import sys
import rospy
import moveit_commander
from trajectory_generation import TrajectoryGenerator

def main():
    rospy.init_node('deep_learning_based_path_planner_test')

    # Initialize MoveIt
    moveit_commander.roscpp_initialize(sys.argv)
    path_planner_tester = TrajectoryGenerator(moveit_commander)

    success_time_planner1 = 0.0
    success_time_planner2 = 0.0
    success_time_planner3 = 0.0
    total_time_planner1 = 0.0
    total_time_planner2 = 0.0
    total_time_planner3 = 0.0
    total_length_planner1 = 0.0
    total_length_planner2 = 0.0
    total_length_planner3 = 0.0

    test_number = 100

    for i in range(test_number):
        env_num = i

        obstacle_meshes = path_planner_tester.generate_random_mesh(env_num)
        pointcloud = path_planner_tester.setObstaclesInScene(obstacle_meshes)

        hasTask, start_joint, target_joint, task_constraints = path_planner_tester.getRandomTaskWithConstraints()

        if not hasTask:
             continue

        print("env ", env_num)
        print("start joint ", start_joint)
        print("target joint ", target_joint)

        path_planner_tester.set_path_planner_id('CVQMPTRRTConfigDefault')
        success, planning_time, path_length = path_planner_tester.measurePlanningWithConstraints(start_joint, target_joint, task_constraints, pointcloud)
        if success:
            success_time_planner1 += 1
            total_time_planner1 += planning_time
            total_length_planner1 += path_length

        path_planner_tester.set_path_planner_id('CBIRRTConfigDefault')
        success, planning_time, path_length = path_planner_tester.measurePlanningWithConstraints(start_joint, target_joint, task_constraints, pointcloud)
        if success:
            success_time_planner2 += 1
            total_time_planner2 += planning_time
            total_length_planner2 += path_length

        # set the pointcloud to zeros
        pointcloud.fill(0)
        path_planner_tester.set_path_planner_id('CVQMPTRRTConfigDefault')
        success, planning_time, path_length = path_planner_tester.measurePlanningWithConstraints(start_joint, target_joint, task_constraints, pointcloud)
        if success:
            success_time_planner3 += 1
            total_time_planner3 += planning_time
            total_length_planner3 += path_length

        # clear the planning scene.
        path_planner_tester.cleanPlanningScene()

    print("planner1 = CVQMPT with pointcloud")
    print("planner2 = CBIRRT")
    print("planner3 = CVQMPT with zero pointcloud")

    print("test done")
    print("success rate of planner1 ", success_time_planner1 / test_number)
    print("success rate of planner2 ", success_time_planner2 / test_number)
    print("success rate of planner3 ", success_time_planner3 / test_number)
    print("average planning time of planner1 ", total_time_planner1 /  success_time_planner1)
    print("average planning time of planner2 ", total_time_planner2 /  success_time_planner2)
    print("average planning time of planner3 ", total_time_planner3 /  success_time_planner3)
    print("average path length of planner1 ", total_length_planner1 / success_time_planner1)
    print("average path length of planner2 ", total_length_planner2 / success_time_planner2)
    print("average path length of planner3 ", total_length_planner3 / success_time_planner3)

if __name__ == '__main__':
    main()
