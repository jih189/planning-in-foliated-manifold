#!/usr/bin/env python
import sys
import rospy
import rospkg
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from geometry_msgs.msg import Point
from moveit_msgs.msg import PlanningScene, CollisionObject
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from ros_numpy import numpify, msgify
from shape_msgs.msg import Mesh
from shape_msgs.msg import MeshTriangle
import numpy as np
import os

import trimesh

class MeshCollisionChecker:
    def __init__(self, mc):
        self.robot = mc.RobotCommander()
        self.scene = mc.PlanningSceneInterface()
        self.scene.clear()
        rospy.sleep(0.5)
        self.move_group = mc.MoveGroupCommander("arm")
        self.joint_names = self.move_group.get_active_joints()
        self.state_validity_service = rospy.ServiceProxy('/check_state_validity', GetStateValidity)

        # # set initial joint state
        joint_state_publisher = rospy.Publisher('/move_group/fake_controller_joint_states', JointState, queue_size=1)

        # Create a JointState message
        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = ['torso_lift_joint', 'shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint', 'wrist_flex_joint', 'l_gripper_finger_joint', 'r_gripper_finger_joint']
        joint_state.position = [0.38, -1.28, 1.52, 0.35, 1.81, 1.47, 0.04, 0.04]

        rate = rospy.Rate(10)
        while(joint_state_publisher.get_num_connections() < 1): # need to wait until the publisher is ready.
            rate.sleep()
        joint_state_publisher.publish(joint_state)

    def trimesh_to_shape_msgs_mesh(self, object_id, tri_mesh, tri_pose):
        '''
        Generate collisionObject from a trimesh object mesh.
        '''
        co = CollisionObject()
        co.operation = CollisionObject.ADD
        co.id = object_id
        co.header.frame_id = "base_link"
        co.pose = tri_pose

        mesh_msg = Mesh()

        for vertex in tri_mesh.vertices:
            mesh_msg.vertices.append(Point(x=vertex[0], y=vertex[1], z=vertex[2]))

        for face in tri_mesh.faces:
            mesh_msg.triangles.append(MeshTriangle(vertex_indices=[face[0], face[1], face[2]]))

        co.meshes = [mesh_msg]

        return co

    def cleanPlanningScene(self):
        '''
        clean the planning scene.
        '''
        self.scene.clear()
        rospy.sleep(0.5)

    def checkInitCollision(self):
        '''
        Check the collsion happening in the scene.
        '''
        joint_values = self.move_group.get_current_joint_values()
        # Create a GetStateValidityRequest object
        request = GetStateValidityRequest()
        request.robot_state.joint_state.name = self.joint_names
        request.robot_state.joint_state.position = joint_values

        result = []
        for contact in self.state_validity_service(request).contacts:
            result.append(contact.contact_body_1)
        return list(set(result))

    def checkValid(self, joint_values, joint_names):
        '''
        Check the collsion happening in the scene.
        '''
        # Create a GetStateValidityRequest object
        request = GetStateValidityRequest()
        request.robot_state.joint_state.name = joint_names
        request.robot_state.joint_state.position = joint_values

        return self.state_validity_service(request).valid

    def setObstaclesInScene(self, mesh_list, pose_list):
        '''
        Set the obstacle into the planning scene.
        '''
        for i in range(len(mesh_list)):
            mesh_msg = self.trimesh_to_shape_msgs_mesh("obstacle"+str(i), mesh_list[i], pose_list[i])
            self.scene.add_object(mesh_msg)
        rospy.sleep(1.0)


if __name__ == "__main__":

    rospack = rospkg.RosPack()
    
    # Get the path of the desired package
    package_path = rospack.get_path('data_generation')

    # check whether the dir exists
    if not os.path.exists(package_path + '/gmm_data'):
        print "The directory does not exist. Please run the random_joint_state_generation.py first."
        sys.exit()

    # check whether the file exists
    if (not os.path.exists(package_path + '/gmm_data/joint_names.npy')) or \
        (not os.path.exists(package_path + '/gmm_data/valid_robot_states.npy')):
        print "The file does not exist. Please run the random_joint_state_generation.py first."
        sys.exit()

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('mesh_collision_checker_node', anonymous=True)

    mesh_path = rospy.get_param('~mesh_path', '') # the mesh should be in the base_link frame

    mesh_collision_checker = MeshCollisionChecker(moveit_commander)

    # load the joint names
    joint_names = np.load(package_path + '/gmm_data/joint_names.npy')

    # load the valid robot states
    valid_robot_states = np.load(package_path + '/gmm_data/valid_robot_states.npy')

    # initialize the mesh pose
    mesh_pose = geometry_msgs.msg.Pose()
    mesh_pose.position.x = 0.0
    mesh_pose.position.y = 0.0
    mesh_pose.position.z = 0.0
    mesh_pose.orientation.x = 0.0
    mesh_pose.orientation.y = 0.0
    mesh_pose.orientation.z = 0.0
    mesh_pose.orientation.w = 1.0

    if mesh_path == '':
        # print "Please specify the mesh path."
        # sys.exit()
        task_package_path = rospack.get_path('task_planner')
        mesh_path = task_package_path + '/mesh_dir/table.stl'
        mesh_pose = msgify(geometry_msgs.msg.Pose, np.array([[0.0, 1.0, 0.0, 0.28], [-1.0, 0.0, 0.0, 0.92], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]))

    # load the mesh
    mesh = trimesh.load_mesh(mesh_path)

    # set the obstacle in the scene
    mesh_collision_checker.setObstaclesInScene([mesh], [mesh_pose])

    for j in range(valid_robot_states.shape[0]):
        if mesh_collision_checker.checkValid(valid_robot_states[j], joint_names):
            print "valid robot state: ", valid_robot_states[j]
        else:
            print "invalid robot state: ", valid_robot_states[j]

    # clean the planning scene
    mesh_collision_checker.cleanPlanningScene()

    # shutdown the moveit
    moveit_commander.roscpp_shutdown()
    moveit_commander.os._exit(0)