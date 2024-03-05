#!/usr/bin/env python
import sys
import copy
import rospy
import rospkg
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from ros_numpy import numpify, msgify
import numpy as np
import os

def convert_joint_values_to_robot_state(joint_values_list_, joint_names_, robot_):
    '''
    convert a list of joint values to robotState
    joint_values_list_: a list of joint values
    joint_names_: a list of joint names
    robot_: a robotCommander
    '''
    moveit_robot_state = robot_.get_current_state()
    position_list = list(moveit_robot_state.joint_state.position)
    for joint_name, joint_value in zip(joint_names_, joint_values_list_):
        position_list[moveit_robot_state.joint_state.name.index(joint_name)] = joint_value
    moveit_robot_state.joint_state.position = tuple(position_list)
    return moveit_robot_state

if __name__ == "__main__":

    rospack = rospkg.RosPack()
    
    # Get the path of the desired package
    package_path = rospack.get_path('data_generation')

    # check whether the dir exists
    if not os.path.exists(package_path + '/gmm_data'):
        os.makedirs(package_path + '/gmm_data')

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('random_joint_state_generation_node', anonymous=True)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    scene.clear()
    move_group = moveit_commander.MoveGroupCommander("arm")
    state_validity_service = rospy.ServiceProxy('/check_state_validity', GetStateValidity)

    # set initial joint state
    joint_state_publisher = rospy.Publisher('/move_group/fake_controller_joint_states', JointState, queue_size=1)

    # Create a JointState message
    joint_state = JointState()
    joint_state.header.stamp = rospy.Time.now()
    joint_state.name = ['torso_lift_joint', 'shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint', 'wrist_flex_joint', 'l_gripper_finger_joint', 'r_gripper_finger_joint']
    joint_state.position = [0.38, -1.28, 1.52, 0.35, 1.81, 1.47, 0.04, 0.04]

    # publish the initial joint state
    rate = rospy.Rate(10)
    while(joint_state_publisher.get_num_connections() < 1): # need to wait until the publisher is ready.
        rate.sleep()
    joint_state_publisher.publish(joint_state)

    moveit_robot_state = robot.get_current_state()
    active_joint_names = move_group.get_active_joints()

    # save the joint names
    np.save(package_path + '/gmm_data/joint_names.npy', moveit_robot_state.joint_state.name)

    valid_robot_states = []

    for _ in range(100):
        random_joint_value = move_group.get_random_joint_values()

        request = GetStateValidityRequest()
        request.robot_state.joint_state.name = active_joint_names
        request.robot_state.joint_state.position = random_joint_value
        result = state_validity_service(request)

        if result.valid:
            random_robot_state = convert_joint_values_to_robot_state(random_joint_value, active_joint_names, robot)        
            valid_robot_states.append(random_robot_state.joint_state.position)

    # save the valid robot states
    np.save(package_path + '/gmm_data/valid_robot_states.npy', valid_robot_states)