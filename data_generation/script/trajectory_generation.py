#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import random
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
from moveit_msgs.msg import PlanningScene, CollisionObject, AttachedCollisionObject
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Point, Point32
from shape_msgs.msg import Mesh
from shape_msgs.msg import MeshTriangle

from moveit_msgs.srv import GetJointWithConstraints, GetJointWithConstraintsRequest
from moveit_msgs.msg import Constraints, OrientationConstraint, MoveItErrorCodes, RobotState
from geometry_msgs.msg import Quaternion, Pose

from moveit_msgs.msg import RobotTrajectory, DisplayTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

import numpy as np
import pickle
from os import path as osp
import os
import shutil

import trimesh
from trimesh_util import sample_points_on_mesh, filter_points_inside_mesh, write_ply

from sensor_msgs.msg import PointCloud2, PointCloud, PointField
import std_msgs.msg
import struct

import time

class TrajectoryGenerator:
    def __init__(self, mc):
        self.robot = mc.RobotCommander()
        self.scene = mc.PlanningSceneInterface()
        self.scene.clear()
        self.move_group = mc.MoveGroupCommander("arm")
        self.state_validity_service = rospy.ServiceProxy('/check_state_validity', GetStateValidity)
        self.sample_joint_with_constraints_service = rospy.ServiceProxy('/sample_joint_with_constraints', GetJointWithConstraints)
        self.joint_names = self.move_group.get_active_joints()

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

        # set the planner to rrt star
        #self.move_group.set_planner_id('RRTstarkConfigDefault')
        self.move_group.set_planner_id('RRTConnectkConfigDefault')
        self.move_group.set_planning_time(10.0)

        self.pointcloud_pub = rospy.Publisher("/obstacle_point_cloud", PointCloud2, queue_size=1)

        self.start_configuration = []
        self.goal_configuration = []
        self.planning_time_cost = 0.0

    def show_point_cloud(self, pointcloud):
        '''
        publish the pointcloud to the rviz
        '''
        point_cloud_msg = self.numpy_to_pointcloud2(pointcloud, frame_id="base_link")
        self.pointcloud_pub.publish(point_cloud_msg)

    def set_path_planner_id(self, planner_id):
        self.move_group.set_planner_id(planner_id)
        
    def numpy_to_pointcloud(self, points, frame_id="base_link"):
        pc_msg = PointCloud()
        pc_msg.header.stamp = rospy.Time.now()
        pc_msg.header.frame_id = frame_id

        for point in points:
            p = Point32()
            p.x = point[0]
            p.y = point[1]
            p.z = point[2]
            pc_msg.points.append(p)

        return pc_msg


    def numpy_to_pointcloud2(self, points, frame_id="base_link"):
        '''
        convert pointcloud from numpy format to PointCloud2 in the base_link frame.
        '''
        pc2_msg = PointCloud2()
        pc2_msg.header.stamp = rospy.Time.now()
        pc2_msg.header.frame_id = frame_id
        pc2_msg.height = 1
        pc2_msg.width = len(points)
        pc2_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        pc2_msg.is_bigendian = False
        pc2_msg.point_step = 12
        pc2_msg.row_step = pc2_msg.point_step * pc2_msg.width
        pc2_msg.is_dense = True

        buffer = []

        for point in points:
            float_bytes = [struct.pack('f', coord) for coord in point]
            buffer.append(b''.join(float_bytes))

        pc2_msg.data = b''.join(buffer)

        return pc2_msg

    def trimesh_to_shape_msgs_mesh(self, object_id, tri_mesh):
        '''
        Generate collisionObject from a trimesh object mesh.
        '''
        co = CollisionObject()
        co.operation = CollisionObject.ADD
        co.id = object_id
        co.header.frame_id = "base_link"
        co.pose.position.x = 0.0
        co.pose.position.y = 0.0
        co.pose.position.z = 0.0
        co.pose.orientation.w = 1.0

        mesh_msg = Mesh()

        for vertex in tri_mesh.vertices:
            mesh_msg.vertices.append(Point(x=vertex[0], y=vertex[1], z=vertex[2]))

        for face in tri_mesh.faces:
            mesh_msg.triangles.append(MeshTriangle(vertex_indices=[face[0], face[1], face[2]]))

        co.meshes = [mesh_msg]

        return co

    def setObstaclesInScene(self, mesh_list):
        '''
        Set the obstacle into the planning scene as long as it does not collide with the robot, and return the pointcloud in numpy format.
        '''

        for i in range(len(mesh_list)):
            mesh_msg = self.trimesh_to_shape_msgs_mesh("obstacle"+str(i), mesh_list[i])
            self.scene.add_object(mesh_msg)
        rospy.sleep(1)

        # need to remove the obstacle if it collides with the robot.
        bad_obstacle_ids = self.checkCollision()
        for bad_obstacle_id in bad_obstacle_ids:
            self.scene.remove_world_object(bad_obstacle_id)
        rospy.sleep(1)

        meshes = trimesh.util.concatenate([mesh_list[i] for i in range(len(mesh_list)) if "obstacle"+str(i) not in bad_obstacle_ids])

        pointcloud = sample_points_on_mesh(meshes, 5000)

        filtered_pointcloud = filter_points_inside_mesh(meshes, pointcloud)

        return filtered_pointcloud

    def cleanPlanningScene(self):
        '''
        clean the planning scene.
        '''
        self.scene.clear()

    def checkCollision(self):
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

    def getValidJoints(self):
        '''
        Return a valid joint values of the Fetch. If something stuck here, it can be
        caused by too little joint values are valid.
        output: success, joint value
        '''
        count = 0
        while count < 10:
            joint_values = self.move_group.get_random_joint_values()
            # Create a GetStateValidityRequest object
            request = GetStateValidityRequest()
            request.robot_state.joint_state.name = self.joint_names
            request.robot_state.joint_state.position = joint_values
            result = self.state_validity_service(request)
            if result.valid:
                return True, joint_values
            count += 1
        return False, None

    def getRandomTask(self):
        count = 0
        while count < 100:
            start_joint_success, start_joint = self.getValidJoints()
            if not start_joint_success:
                count += 1
                continue
            target_joint_success, target_joint = self.getValidJoints()
            if not target_joint_success:
                count += 1
                continue

            return True, start_joint, target_joint
        return False, None, None

    def getProperJointState(self, all_name, all_joint, selected_name):
        result = []
        for sn in selected_name:
            result.append(all_joint[all_name.index(sn)])
        return result

    def getRandomTaskWithConstraints(self):
        horizontal_constraint = Constraints()
        horizontal_constraint.name = "use_equality_constraints"

        oc = OrientationConstraint()

        oc.parameterization = OrientationConstraint.ROTATION_VECTOR;
        oc.header.frame_id = "base_link";
        oc.header.stamp = rospy.Time(0)
        oc.link_name = "wrist_roll_link";
        constrained_quaternion = Quaternion();
        constrained_quaternion.x = 0.0
        constrained_quaternion.y = 0.0
        constrained_quaternion.z = 0.0
        constrained_quaternion.w = 1.0
        oc.orientation = constrained_quaternion
        oc.weight = 1.0

        oc.absolute_x_axis_tolerance = 0.1
        oc.absolute_y_axis_tolerance = 0.1
        oc.absolute_z_axis_tolerance = 2 * 3.1415
        horizontal_constraint.orientation_constraints.append(oc)

        # need to set in-hand pose
        in_hand_pose = Pose()
        in_hand_pose.position.x = 0.0
        in_hand_pose.position.y = 0.0
        in_hand_pose.position.z = 0.0
        in_hand_pose.orientation.x = 0.0
        in_hand_pose.orientation.y = 0.0
        in_hand_pose.orientation.z = 0.0
        in_hand_pose.orientation.w = 1.0
        horizontal_constraint.in_hand_pose = in_hand_pose

        sample_request = GetJointWithConstraintsRequest()
        sample_request.constraints = horizontal_constraint
        sample_request.group_name = "arm"
        sample_request.max_sampling_attempt = 20

        count = 0
        while count < 100:
            start_config_val = self.sample_joint_with_constraints_service(sample_request)
            if start_config_val.error_code.val != MoveItErrorCodes.SUCCESS:
                count += 1
                continue

            start_joint = self.getProperJointState(start_config_val.solution.joint_state.name, start_config_val.solution.joint_state.position, self.joint_names)
            
            goal_config_val = self.sample_joint_with_constraints_service(sample_request)
            if goal_config_val.error_code.val != MoveItErrorCodes.SUCCESS:
                count += 1
                continue

            target_joint = self.getProperJointState(goal_config_val.solution.joint_state.name, goal_config_val.solution.joint_state.position, self.joint_names)

            return True, start_joint, target_joint, horizontal_constraint
        return False, None, None, None

    def measurePlanningWithConstraints(self, start_joint, target_joint, task_constraints, pointcloud):
        moveit_robot_state = RobotState()
        moveit_robot_state.joint_state.name = self.joint_names
        moveit_robot_state.joint_state.position = start_joint

        self.move_group.set_start_state(moveit_robot_state)
        self.move_group.set_joint_value_target(target_joint)
        self.move_group.set_path_constraints(task_constraints)
        self.move_group.set_in_hand_pose(task_constraints.in_hand_pose)
        self.move_group.set_obstacle_point_cloud(self.numpy_to_pointcloud(pointcloud))
        
        start_time = time.time()
        result = self.move_group.plan()
        planning_time = time.time() - start_time
        if result[0]:
            sampled_trajectory = np.array([j.positions for j in result[1].joint_trajectory.points])
            path_length = np.linalg.norm(np.diff(sampled_trajectory, axis=0), axis=1).sum()
            return True, planning_time, path_length
        else:
            return False, 0.0, 0.0

    def measurePlanning(self, start_joint, target_joint):
        moveit_robot_state = RobotState()
        moveit_robot_state.joint_state.name = self.joint_names
        moveit_robot_state.joint_state.position = start_joint

        self.move_group.set_start_state(moveit_robot_state)
        self.move_group.set_joint_value_target(target_joint)
        
        start_time = time.time()
        result = self.move_group.plan()
        planning_time = time.time() - start_time
        if result[0]:
            sampled_trajectory = np.array([j.positions for j in result[1].joint_trajectory.points])
            path_length = np.linalg.norm(np.diff(sampled_trajectory, axis=0), axis=1).sum()
            return True, planning_time, path_length
        else:
            return False, 0.0, 0.0

    def motionTaskPlanning(self):
        '''
        It first samples two valid joint values, then plan for the trajectory between them.
        output: success, trajectory
        '''
        count = 0
        while count < 100:
            start_joint_success, start_joint = self.getValidJoints()
            if not start_joint_success:
                count += 1
                continue
            target_joint_success, target_joint = self.getValidJoints()
            if not target_joint_success:
                count += 1
                continue

            moveit_robot_state = RobotState()
            moveit_robot_state.joint_state.name = self.joint_names
            moveit_robot_state.joint_state.position = start_joint

            self.start_configuration = start_joint
            self.goal_configuration = target_joint

            self.move_group.set_start_state(moveit_robot_state)
            self.move_group.set_joint_value_target(target_joint)
            start_time = time.time()
            result = self.move_group.plan()
            self.planning_time_cost = time.time() - start_time
            if result[0]:
                sampled_trajectory = np.array([j.positions for j in result[1].joint_trajectory.points])
                path_length = np.linalg.norm(np.diff(sampled_trajectory, axis=0), axis=1).sum()
                return True, path_length
            else:
                return False, 0.0
        print("there are no valid task in this env")
        return False, 0.0
        #         sampled_trajectory = [j.positions for j in result[1].joint_trajectory.points]
        #         return True, np.array(sampled_trajectory)
        #     else:
        #         count += 1
        # return False, None

    def generateValidTrajectory(self):
        '''
        It first samples two valid joint values, then plan for the trajectory between them.
        output: success, trajectory
        '''
        count = 0
        while count < 100:
            start_joint_success, start_joint = self.getValidJoints()
            if not start_joint_success:
                count += 1
                continue
            target_joint_success, target_joint = self.getValidJoints()
            if not target_joint_success:
                count += 1
                continue

            moveit_robot_state = RobotState()
            moveit_robot_state.joint_state.name = self.joint_names
            moveit_robot_state.joint_state.position = start_joint

            self.start_configuration = start_joint
            self.goal_configuration = target_joint

            self.move_group.set_start_state(moveit_robot_state)
            self.move_group.set_joint_value_target(target_joint)
            start_time = time.time()
            result = self.move_group.plan()
            self.planning_time_cost = time.time() - start_time
            if result[0]:
                sampled_trajectory = [j.positions for j in result[1].joint_trajectory.points]
                return True, np.array(sampled_trajectory)
            else:
                count += 1
        return False, None

    def print_task(self):
        print("joint names")
        print(self.joint_names)
        print("start configuration")
        print(self.start_configuration)
        print("target configuration")
        print(self.goal_configuration)

    def get_planning_time(self):
        return self.planning_time_cost

    def visualizeTrajectory(self, trajectory_data):
        '''
        Visualize the trajectory in robot trajectory format.
        '''
        robot_trajectory = RobotTrajectory()
        robot_trajectory.joint_trajectory.joint_names = self.joint_names
        currenttime = rospy.Duration(0.0)

        for point in trajectory_data.tolist():
            joint_trajectory_point = JointTrajectoryPoint()
            joint_trajectory_point.positions = point
            joint_trajectory_point.time_from_start = currenttime
            robot_trajectory.joint_trajectory.points.append(joint_trajectory_point)
            currenttime += rospy.Duration(0.5)

        # Create a DisplayTrajectory message
        display_trajectory = DisplayTrajectory()
        display_trajectory.trajectory.append(robot_trajectory)

        display_trajectory_publisher = rospy.Publisher('/move_group/planned_path', DisplayTrajectory, queue_size=1)
        display_trajectory_publisher.publish(display_trajectory)

    def random_point_in_bounding_box(self, min_point, max_point):
        # Generate a random point within the bounding box
        random_point = np.random.rand(3) * (max_point - min_point) + min_point
        return random_point

    def generate_random_mesh(self, seed_value, obstacle_size=0.3, min_point=[-0.5, -1.0, 0.0], max_point=[1.0,1.0,2.0], obstacle_num=20, num_points=20):
        np.random.seed(seed_value)
        seed_number_list = np.random.randint(0, 10000, size=obstacle_num)
        result = []
        for sn in seed_number_list:
            np.random.seed(sn)
            points = (np.random.rand(num_points, 3) - 0.5) * obstacle_size
            # Create a convex hull from the random points
            hull = trimesh.convex.convex_hull(points)

            position = self.random_point_in_bounding_box(np.array(min_point), np.array(max_point))

            hull.apply_translation(position.tolist())
            result.append(hull)

        return result

def main():
    ###################
    scene_count = 200
    trajectory_count_per_scene = 20
    rospy.init_node('data_trajectory_generation')

    # Initialize MoveIt
    moveit_commander.roscpp_initialize(sys.argv)
    trajectory_generator = TrajectoryGenerator(moveit_commander)

    fileDir = 'trajectory_data/'

    # remove the directory for data if it exists.
    if os.path.exists(fileDir):
        shutil.rmtree(fileDir)

    os.mkdir(fileDir)

    scene_count += 1
    start_scene = 0

    for env_num in range(start_scene, start_scene + scene_count):
        print "process ", env_num - start_scene, " / ", (scene_count - 1)
        os.mkdir(fileDir + "env_%06d/" % env_num)

        # use the env_num as the seed
        obstacle_meshes = trajectory_generator.generate_random_mesh(env_num)
        pointcloud = trajectory_generator.setObstaclesInScene(obstacle_meshes)

        write_ply(fileDir + "env_%06d/map_" % env_num + "%d.ply" % env_num, pointcloud)

        i = 0
        while i < trajectory_count_per_scene:
            plan_result, sampled_trajectory = trajectory_generator.generateValidTrajectory()
            if not plan_result:
                continue
            trajData = {'path': sampled_trajectory}
            with open(fileDir + "env_%06d/" % env_num + "path_%d.p" % i, 'wb') as f:
                pickle.dump(trajData, f)
            i += 1

        trajectory_generator.cleanPlanningScene()

        # # Load the saved numpy array using pickle
        # with open(fileDir + "path_%d.p" % i, 'rb') as f:
        #     loaded_array = pickle.load(f)
        # print loaded_array['path']




if __name__ == '__main__':
    main()
