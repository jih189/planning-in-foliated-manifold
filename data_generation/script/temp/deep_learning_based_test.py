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

    success_time_deep_learning = 0.0
    success_time_non_deep_learning = 0.0
    total_time_deep_learning = 0.0
    total_time_non_deep_learning = 0.0
    total_length_deep_learning = 0.0
    total_length_non_deep_learning = 0.0

    test_number = 1

    for i in range(test_number):
        env_num = i

        obstacle_meshes = path_planner_tester.generate_random_mesh(env_num)
        pointcloud = path_planner_tester.setObstaclesInScene(obstacle_meshes)

        hasTask, start_joint, target_joint, task_constraints = path_planner_tester.getRandomTaskWithConstraints()
        #hasTask, start_joint, target_joint = path_planner_tester.getRandomTask()

        if not hasTask:
             continue

        print("env ", env_num)
        print("start joint ", start_joint)
        print("target joint ", target_joint)

        path_planner_tester.set_path_planner_id('CMPNETRRTConfigDefault')
        success, planning_time, path_length = path_planner_tester.measurePlanningWithConstraints(start_joint, target_joint, task_constraints, pointcloud)
        if success:
            success_time_deep_learning += 1
            total_time_deep_learning += planning_time
            total_length_deep_learning += path_length

        path_planner_tester.set_path_planner_id('CBIRRTConfigDefault')
        success, planning_time, path_length = path_planner_tester.measurePlanningWithConstraints(start_joint, target_joint, task_constraints, pointcloud)
        if success:
            success_time_non_deep_learning += 1
            total_time_non_deep_learning += planning_time
            total_length_non_deep_learning += path_length

        # clear the planning scene.
        path_planner_tester.cleanPlanningScene()

    # print("test done")
    # print("success rate of deep learning ", success_time_deep_learning / test_number)
    # print("success rate of non deep learning ", success_time_non_deep_learning / test_number)
    # print("average planning time of deep learning ", total_time_deep_learning /  success_time_deep_learning)
    # print("average planning time of non deep learning ", total_time_non_deep_learning /  success_time_non_deep_learning)
    # print("average path length of deep learning ", total_length_deep_learning / success_time_deep_learning)
    # print("average path length of non deep learning ", total_length_non_deep_learning / success_time_non_deep_learning)

if __name__ == '__main__':
    main()
