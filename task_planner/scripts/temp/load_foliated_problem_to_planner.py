#!/usr/bin/env python
from foliated_base_class import FoliatedProblem, FoliatedIntersection
from manipulation_foliations_and_intersections import (
    ManipulationFoliation,
    ManipulationIntersection,
)
from foliated_planning_framework import FoliatedPlanningFramework
from jiaming_GMM import GMM
from jiaming_task_planner import (
    MTGTaskPlanner,
    MTGTaskPlannerWithGMM,
    MTGTaskPlannerWithAtlas,
)
from jiaming_motion_planner import MoveitMotionPlanner
from jiaming_visualizer import MoveitVisualizer

import rospy
import rospkg

# from moveit_msgs.msg import RobotState


if __name__ == "__main__":
    rospy.init_node("main_pipeline_node", anonymous=True)

    rospack = rospkg.RosPack()

    # Get the path of the desired package
    package_path = rospack.get_path("task_planner")

    # load the foliated problem
    loaded_foliated_problem = FoliatedProblem.load(
        ManipulationFoliation, ManipulationIntersection, package_path + "/check"
    )

    # load the gmm
    gmm_dir_path = package_path + "/computed_gmms_dir/dpgmm/"
    # gmm_dir_path = package_path + '/computed_gmms_dir/gmm/'
    gmm = GMM()
    gmm.load_distributions(gmm_dir_path)

    # initialize the motion planner
    motion_planner = MoveitMotionPlanner()

    # # get the current robot state
    # robot_state = RobotState()
    # robot_state.joint_state.name = ['torso_lift_joint', 'shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint', 'wrist_flex_joint', 'l_gripper_finger_joint', 'r_gripper_finger_joint']
    # robot_state.joint_state.position = [0.38, -1.28, 1.52, 0.35, 1.81, 1.47, 0.04, 0.04]

    # initialize the foliated planning framework, and set the task planner and motion planner
    foliated_planning_framework = FoliatedPlanningFramework()
    foliated_planning_framework.setMotionPlanner(motion_planner)

    # load it into the task planner.
    # task_planner = MTGTaskPlanner()
    # task_planner = MTGTaskPlannerWithGMM(gmm)
    task_planner = MTGTaskPlannerWithAtlas(
          gmm, motion_planner.move_group.get_current_state()
    )  # need to provide the current robot state as the default robot state.
    foliated_planning_framework.setTaskPlanner(task_planner)
    foliated_planning_framework.setMaxAttemptTime(30)

    # initialize the visualizer
    visualizer = MoveitVisualizer()
    visualizer.prepare_visualizer(
        motion_planner.move_group.get_active_joints(), motion_planner.robot
    )

    # set the visualizer
    foliated_planning_framework.setVisualizer(visualizer)

    # set the foliated problem
    foliated_planning_framework.setFoliatedProblem(loaded_foliated_problem)

    # set the start and goal
    foliated_planning_framework.setStartAndGoal(
        0,
    0,
        ManipulationIntersection(
            action="start",
            motion=[[-1.28, 1.51, 0.35, 1.81, 0.0, 1.47, 0.0]],
            active_joints=motion_planner.move_group.get_active_joints(),
        ),
        0,
        5,
        ManipulationIntersection(
            action="goal",
            motion=[[-1.28, 1.51, 0.35, 1.81, 0.0, 1.47, 0.0]],
            active_joints=motion_planner.move_group.get_active_joints(),
        ),
    )

    # foliated_planning_framework.setStartAndGoal(
    #     0, 10,
    #     ManipulationIntersection(action='start', motion=[[ 0.38, -1.28, 1.51, 0.35, 1.81, 1.47, 0.0]], active_joints=motion_planner.move_group.get_active_joints()),
    #     0, 11,
    #     ManipulationIntersection(action='goal', motion=[[ 0.38, -1.28, 1.51, 0.35, 1.81, 1.47, 0.0]], active_joints=motion_planner.move_group.get_active_joints())
    # )

    # solve the problem
    found_solution, solution_trajectory = foliated_planning_framework.solve()

    if found_solution:
        print("found solution")
        # visualize the solution
        foliated_planning_framework.visualizeSolutionTrajectory(solution_trajectory)
    else:
        print("no solution found")

    # shutdown the planning framework
    foliated_planning_framework.shutdown()
