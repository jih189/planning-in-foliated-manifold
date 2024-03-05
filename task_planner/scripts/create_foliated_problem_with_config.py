#!/usr/bin/env python
import os
import sys
import yaml
import trimesh
import numpy as np
import random
import rospkg
import rospy
import moveit_commander
import geometry_msgs.msg
import tf.transformations as tf_trans

from sensor_msgs.msg import JointState
from manipulation_foliations_and_intersections import ManipulationFoliation
from foliated_base_class import FoliatedIntersection, FoliatedProblem
from jiaming_helper import create_pose_stamped, get_position_difference_between_poses, gaussian_similarity, \
    create_pose_stamped_from_raw, collision_check, convert_pose_stamped_to_matrix, create_rotation_matrix_from_euler
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_msgs.msg import MoveItErrorCodes
from manipulation_foliations_and_intersections import ManipulationIntersection
from ros_numpy import numpify, msgify
from geometry_msgs.msg import Quaternion, Point, Pose, PoseStamped, Point32
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


class Config(object):
    def __init__(self, package_name):
        self.config_data = {}
        self.package_path = None

        self.load_config(package_name)

    def load_config(self, package_name):
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('task_planner')
        self.package_path = package_path

        try:
            with open(self.package_path + '/problems/config.yaml', 'r') as yaml_file:
                config_base = yaml.safe_load(yaml_file)

            problem_path = config_base["path"]

            with open(self.package_path + problem_path, 'r') as yaml_file:
                self.config_data = yaml.safe_load(yaml_file)

            random_seed = self.config_data.get('task_parameters', {}).get('random_seed', None)
            if random_seed is not None:
                random.seed(random_seed)
                print("Random seed has been set to: " + str(random_seed))
            else:
                print("No random seed detected")

        except Exception as e:
            print("Unable to load config" + str(e))
            self.config_data = None

    def get(self, section, key=None, default=None):
        config_target = None
        if key:
            config_target = self.config_data.get(section, {}).get(key, default)
        else:
            config_target = self.config_data.get(section, {})
        return config_target

    def set(self, section, key, value):
        if section not in self.config_data:
            self.config_data[section] = {}
        self.config_data[section][key] = value


class FoliatedBuilder(object):
    def __init__(self, config):
        self.package_path = config.package_path
        self.env_mesh_path = self.package_path + config.get('environment', 'env_mesh_path')
        self.manipulated_object_mesh_path = self.package_path + config.get('environment',
                                                                           'manipulated_object_mesh_path')
        self.grasp_poses_file = self.package_path + config.get('environment', 'grasp_poses_file')
        self.env_pose = create_pose_stamped(config.get('environment', 'env_pose'))
        self.ref_pose = np.array(config.get('environment', 'ref_pose'))

        self.foliation_regrasp = None
        self.foliation_dict = {}
        self.sliding_similarity_matrix = None
        self.similarity_sigma = config.get('placement', 'similarity_sigma')
        self.env_mesh = None
        self.collision_manager = None
        self.feasible_placements = []
        self.feasible_grasps = []

        self.placement_parameters = config.get('placement')
        self.grasp_parameters = config.get('grasp')

        self.sample_task_queue = []
        self.foliation_placement_group = []

        self.initialize()

    def initialize(self):
        self._load_environment()
        self._handle_grasps(**self.grasp_parameters)
        self._calc_similarity_matrix(self.feasible_grasps)
        self._handle_placements(self.placement_parameters)
        self._create_grasp_foliation()

    def _load_environment(self):
        self.env_mesh = trimesh.load_mesh(self.env_mesh_path)
        self.env_mesh.apply_transform(convert_pose_stamped_to_matrix(self.env_pose))
        self.collision_manager = trimesh.collision.CollisionManager()
        self.collision_manager.add_object('env', self.env_mesh)

    def _placement_rectangular(self, foliation):
        size_row = foliation.get("size_row")
        size_col = foliation.get("size_col")
        position = np.array(foliation.get("placement_position"))

        if size_row and size_col:
            x_shift = position[0]
            y_shift = position[1]
            z_shift = position[2]

            for i in range(size_row):
                for j in range(size_col):
                    obj_pose = create_pose_stamped_from_raw("base_link",
                                                            i * 0.1 - size_row * 0.1 / 2 + x_shift + 0.05,
                                                            j * 0.1 - size_col * 0.1 / 2 + y_shift + 0.05,
                                                            z_shift,
                                                            0, 0, 0, 1)

                    if collision_check(self.collision_manager, self.manipulated_object_mesh_path, obj_pose):
                        self.feasible_placements.append(convert_pose_stamped_to_matrix(obj_pose))

    def _placement_point(self, foliation):
        position = np.array(foliation.get("placement_position"))
        euler = np.array(foliation.get("placement_orientation"))

        orientation = tf_trans.quaternion_from_euler(euler[0], euler[1], euler[2])
        obj_pose = create_pose_stamped_from_raw("base_link", position[0], position[1], position[2], orientation[0],
                                                orientation[1], orientation[2], orientation[3])

        if collision_check(self.collision_manager, self.manipulated_object_mesh_path, obj_pose):
            self.feasible_placements.append(convert_pose_stamped_to_matrix(obj_pose))
        else:
            foliation_name = foliation.get("name")
            print "no placement added for: " + foliation_name

    def _placement_circular(self, foliation):
        center_position = np.array(foliation.get("placement_position"))
        radius = foliation.get("radius")
        steps = foliation.get("steps")

        start_angle_z = foliation.get("start_angle_z", 0)
        end_angle_z = foliation.get("end_angle_z", 0)
        start_angle_x = foliation.get("start_angle_x", 0)
        end_angle_x = foliation.get("end_angle_x", 0)
        start_angle_y = foliation.get("start_angle_y", 0)
        end_angle_y = foliation.get("end_angle_y", 0)

        if steps:
            angles_z = np.linspace(start_angle_z, end_angle_z, steps)
            angles_x = np.linspace(start_angle_x, end_angle_x, steps)
            angles_y = np.linspace(start_angle_y, end_angle_y, steps)

            for angle_z, angle_x, angle_y in zip(angles_z, angles_x, angles_y):
                x = center_position[0] + radius * np.cos(angle_z)
                y = center_position[1] + radius * np.sin(angle_z)
                z = center_position[2]
                orientation = tf_trans.quaternion_from_euler(angle_x, angle_y, angle_z)
                obj_pose = create_pose_stamped_from_raw("base_link", x, y, z,
                                                        orientation[0], orientation[1], orientation[2], orientation[3])

                if collision_check(self.collision_manager, self.manipulated_object_mesh_path, obj_pose):
                    self.feasible_placements.append(convert_pose_stamped_to_matrix(obj_pose))

    def _placement_linear(self, foliation):
        start_position = np.array(foliation.get("start_position"))
        end_position = np.array(foliation.get("end_position"))
        num_steps = foliation.get("steps")

        if num_steps:
            positions_to_place = [start_position]
            for step in range(1, num_steps):
                current_position = start_position + (end_position - start_position) * (float(step) / num_steps)
                positions_to_place.append(current_position)
            positions_to_place.append(end_position)
            for position in positions_to_place:
                obj_pose = create_pose_stamped_from_raw("base_link", position[0], position[1], position[2],
                                                        0, 0, 0, 1)

                if collision_check(self.collision_manager, self.manipulated_object_mesh_path, obj_pose):
                    self.feasible_placements.append(convert_pose_stamped_to_matrix(obj_pose))

    def _handle_placements(self, params):
        foliations = params.get("foliations")
        for foliation in foliations:
            foliation_name = foliation.get("name")

            reference_pose = None
            if foliation.get("reference_pose"):
                reference_pose = np.array(foliation.get("reference_pose"))
            elif foliation.get("reference_pose_position"):
                reference_pose = np.array(create_rotation_matrix_from_euler(foliation.get("reference_pose_orientation"),
                                                                            foliation.get("reference_pose_position")))

            orientation_tolerance = foliation.get("orientation_tolerance")
            position_tolerance = foliation.get("position_tolerance")
            foliation_placement = self._create_foliation(foliation_name, "base_link", reference_pose,
                                                         orientation_tolerance,
                                                         position_tolerance, self.feasible_grasps,
                                                         self.sliding_similarity_matrix)

            self.foliation_dict[foliation_name] = foliation_placement

            intersect_with = foliation.get("intersect_with")
            if not intersect_with:
                self.sample_task_queue.append([foliation_placement, "default"])
            else:
                for intersect in intersect_with:
                    self.sample_task_queue.append([foliation_placement, intersect])

            placement_type = foliation.get("type")
            if not placement_type:
                placement_type = params.get("type")
            if not placement_type:
                print "no valid placement type detected"

            if placement_type == "rectangular":
                self._placement_rectangular(foliation)
            elif placement_type == "linear":
                self._placement_linear(foliation)
            elif placement_type == "circular":
                self._placement_circular(foliation)
            elif placement_type == "point":
                self._placement_point(foliation)
            elif placement_type == "none":
                continue
            else:
                print("invalid placement type, check config: " + foliation_name)

    def _handle_grasps(self, num_samples, rotated_matrix):
        loaded_array = np.load(self.grasp_poses_file)
        if num_samples == 0:
            num_samples = len(loaded_array.files)
        for ind in random.sample(list(range(len(loaded_array.files))), num_samples):
            self.feasible_grasps.append(np.dot(loaded_array[loaded_array.files[ind]], rotated_matrix))
        random.shuffle(self.feasible_grasps)

    def _calc_similarity_matrix(self, feasible_grasps):
        # calculate sliding similarity matrix
        different_matrix = np.zeros((len(feasible_grasps), len(feasible_grasps)))
        for i, grasp in enumerate(feasible_grasps):
            for j, grasp in enumerate(feasible_grasps):
                if i == j:
                    different_matrix[i, j] = 0
                different_matrix[i, j] = get_position_difference_between_poses(feasible_grasps[i],
                                                                               feasible_grasps[j])

        sliding_similarity_matrix = np.zeros((len(feasible_grasps), len(feasible_grasps)))
        max_distance = np.max(different_matrix)
        for i, grasp in enumerate(feasible_grasps):
            for j, grasp in enumerate(feasible_grasps):
                sliding_similarity_matrix[i, j] = gaussian_similarity(different_matrix[i, j], max_distance,
                                                                      sigma=self.similarity_sigma)

        self.sliding_similarity_matrix = sliding_similarity_matrix

    def _create_grasp_foliation(self, name='regrasp', frame_id="base_link"):
        foliation_regrasp = ManipulationFoliation(foliation_name=name,
                                                constraint_parameters={
                                                    "frame_id": frame_id,
                                                    "is_object_in_hand": False,
                                                    "object_mesh_path": self.manipulated_object_mesh_path,
                                                    "obstacle_mesh": self.env_mesh_path,
                                                    "obstacle_pose": convert_pose_stamped_to_matrix(self.env_pose)
                                                },
                                                co_parameters=self.feasible_placements,
                                                similarity_matrix=np.identity(len(self.feasible_placements)))
        self.foliation_dict["regrasp"] = foliation_regrasp
        self.foliation_dict["default"] = foliation_regrasp
        self.foliation_regrasp = foliation_regrasp

    def _create_foliation(self, name, frame_id, reference_pose, orientation_tolerance, position_tolerance,
                          co_parameters, similarity_matrix):

        foliation_placement = ManipulationFoliation(foliation_name=name,
                                                    constraint_parameters={
                                                        "frame_id": frame_id,
                                                        "is_object_in_hand": True,
                                                        'object_mesh_path': self.manipulated_object_mesh_path,
                                                        "obstacle_mesh": self.env_mesh_path,
                                                        "obstacle_pose": convert_pose_stamped_to_matrix(self.env_pose),
                                                        "reference_pose": reference_pose,
                                                        "orientation_tolerance": orientation_tolerance,
                                                        "position_tolerance": position_tolerance
                                                    },
                                                    co_parameters=co_parameters,
                                                    similarity_matrix=similarity_matrix)
        self.foliation_placement_group.append(foliation_placement)
        return foliation_placement


class RobotScene(object):
    def __init__(self, config):
        self.joint_state_dict = config.get('initial_joint_state')
        self.robot = None
        self.scene = None
        self.move_group = None
        self.compute_ik_srv = None

        self.initialize()

    def initialize(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('create_experiment_node', anonymous=True)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        rospy.sleep(0.5)
        self.scene.clear()

        self.move_group = moveit_commander.MoveGroupCommander("arm")
        rospy.wait_for_service("/compute_ik")
        self.compute_ik_srv = rospy.ServiceProxy("/compute_ik", GetPositionIK)

        self._set_initial_joint_state()

    def _set_initial_joint_state(self):
        joint_state_publisher = rospy.Publisher('/move_group/fake_controller_joint_states', JointState, queue_size=1)
        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = list(self.joint_state_dict.keys())
        joint_state.position = list(self.joint_state_dict.values())

        rate = rospy.Rate(10)
        while joint_state_publisher.get_num_connections() < 1:
            rate.sleep()
        joint_state_publisher.publish(joint_state)


class Sampler:
    def __init__(self, config, robot_scene, placements):
        self.robot = robot_scene.robot
        self.move_group = robot_scene.move_group
        self.compute_ik_srv = robot_scene.compute_ik_srv
        self.manipulated_object_mesh_path = config.package_path + config.get('environment',
                                                                             'manipulated_object_mesh_path')
        self.env_pose = create_pose_stamped(config.get('environment', 'env_pose'))
        self.env_mesh_path = config.package_path + config.get('environment', 'env_mesh_path')
        self.scene = robot_scene.scene
        self.fraction = 0.97
        self.target_frame_id = "base_link"
        self.target_group_name = "arm"
        self.grasp_pose_mat = [[1, 0, 0, -0.05], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        self.placements = placements

        self.selected_co_param_grasp = {}

    def _check_tolerance(self, reference_pose, position_tolerance, orientation_tolerance, position):
        ref_translation = reference_pose[:3, 3]
        pos_translation = position[:3, 3]
        translation_diff = np.abs(ref_translation - pos_translation)
        within_position_tolerance = np.all(translation_diff <= position_tolerance)

        ref_rotation_matrix = reference_pose[:3, :3]
        pos_rotation_matrix = position[:3, :3]
        relative_rotation_matrix = np.dot(ref_rotation_matrix.T, pos_rotation_matrix)

        relative_transform = np.identity(4)
        relative_transform[:3, :3] = relative_rotation_matrix

        relative_quaternion = tf_trans.quaternion_from_matrix(relative_transform)
        relative_euler_angles = tf_trans.euler_from_quaternion(relative_quaternion)
        within_orientation_tolerance = np.all(np.abs(relative_euler_angles) <= orientation_tolerance)

        return within_position_tolerance and within_orientation_tolerance

    def _check_grasp_feasibility(self, placement, grasp):
        # calc grasp pose based on base_link
        grasp_pose_mat = np.dot(placement, grasp)
        pre_grasp_pose_mat = np.dot(grasp_pose_mat, np.array(self.grasp_pose_mat))

        # set IK target pose
        ik_target_pose = PoseStamped()
        ik_target_pose.header.stamp = rospy.Time.now()
        ik_target_pose.header.frame_id = self.target_frame_id
        ik_target_pose.pose = msgify(geometry_msgs.msg.Pose, grasp_pose_mat)

        ik_req = GetPositionIKRequest()
        ik_req.ik_request.group_name = self.target_group_name
        ik_req.ik_request.avoid_collisions = True
        ik_req.ik_request.pose_stamped = ik_target_pose

        # random set robot pose
        random_moveit_robot_state = self.robot.get_current_state()
        random_position_list = list(random_moveit_robot_state.joint_state.position)
        for joint_name, joint_value in zip(self.move_group.get_joints(), self.move_group.get_random_joint_values()):
            random_position_list[random_moveit_robot_state.joint_state.name.index(joint_name)] = joint_value
        random_moveit_robot_state.joint_state.position = tuple(random_position_list)
        ik_req.ik_request.robot_state = random_moveit_robot_state

        ik_res = self.compute_ik_srv(ik_req)

        if not ik_res.error_code.val == MoveItErrorCodes.SUCCESS:
            return False, None

        # check motion
        moveit_robot_state = self.robot.get_current_state()
        moveit_robot_state.joint_state.position = ik_res.solution.joint_state.position

        self.move_group.set_start_state(moveit_robot_state)
        (planned_motion, fraction) = self.move_group.compute_cartesian_path(
            [msgify(geometry_msgs.msg.Pose, pre_grasp_pose_mat)], 0.01, 0.0)

        if fraction < self.fraction:
            return False, None

        intersection_motion = np.array([p.positions for p in planned_motion.joint_trajectory.points])
        return True, intersection_motion

    def _sample_co_param(self, foliation_co_param_grasp, foliation_co_param_placement,
                         selected_co_parameters1_index=None, selected_co_parameters2_index=None):

        co_parameters1 = foliation_co_param_grasp.co_parameters

        if isinstance(foliation_co_param_placement, list):
            co_parameters2 = foliation_co_param_placement
        else:
            co_parameters2 = foliation_co_param_placement.co_parameters

        # random choose param1 and param2
        if not selected_co_parameters1_index:
            selected_co_parameters1_index = random.randint(0, len(co_parameters1) - 1)
        grasp = co_parameters1[selected_co_parameters1_index]

        # random select placement and check if valid
        if not selected_co_parameters2_index:
            original_indices = list(range(len(co_parameters2)))

            random.shuffle(original_indices)

            selected_co_parameters2_index = None
            placement = None
            found_valid_sample = False

            reference_pose = foliation_co_param_grasp.constraint_parameters.get("reference_pose")
            position_tolerance = foliation_co_param_grasp.constraint_parameters.get("position_tolerance")
            orientation_tolerance = foliation_co_param_grasp.constraint_parameters.get("orientation_tolerance")

            for index in original_indices:
                placement = co_parameters2[index]
                tolerance = self._check_tolerance(reference_pose, position_tolerance, orientation_tolerance, placement)

                selected_co_parameters2_index = index

                if tolerance:
                    found_valid_sample = True
                    break

            if not found_valid_sample:
                print "can't sample any valid co_parameter, skip..."
                return False, None, None, selected_co_parameters1_index, selected_co_parameters2_index
        else:
            placement = co_parameters2[selected_co_parameters2_index]

        check_result, intersection_motion = self._check_grasp_feasibility(placement, grasp)

        return check_result, intersection_motion, placement, selected_co_parameters1_index, selected_co_parameters2_index

    def sampling_func_placement_grasp(self, foliation_1, foliation_2):
        check_result, intersection_motion, placement, selected_co_parameters1_index, selected_co_parameters2_index = (
            self._sample_co_param(foliation_1, foliation_2))

        if check_result:
            return True, selected_co_parameters1_index, selected_co_parameters2_index, ManipulationIntersection(
                'release',
                intersection_motion,
                self.move_group.get_active_joints(),
                placement,
                self.manipulated_object_mesh_path,
                convert_pose_stamped_to_matrix(self.env_pose),
                self.env_mesh_path
            )
        else:
            return False, selected_co_parameters1_index, selected_co_parameters2_index, None

    def sampling_func_placement_placement(self, foliation_1, foliation_2):
        successful_placements = {}

        for foliation_grasp in [foliation_1, foliation_2]:
            successful_placements[foliation_grasp.foliation_name] = []
            for placement_index, placement in enumerate(self.placements):
                if self._check_tolerance(foliation_grasp.constraint_parameters.get("reference_pose"),
                                         foliation_grasp.constraint_parameters.get("position_tolerance"),
                                         foliation_grasp.constraint_parameters.get("orientation_tolerance"),
                                         placement):
                    successful_placements[foliation_grasp.foliation_name].append(placement)

        common_placements = []
        
        # print successful_placements

        successful_placements_values = successful_placements.values()
        for array1 in successful_placements_values[0]:
            for array2 in successful_placements_values[1]:
                if np.allclose(array1, array2, atol=1e-6):
                    common_placements.append(array1)

        selected_common_placement = common_placements[random.randint(0, len(common_placements) - 1)]
        
        placement_index = None
        for i, placement in enumerate(self.placements):
            # print i, placement
            if np.allclose(placement, selected_common_placement, atol=1e-6):
                placement_index = i
                break

        if self.selected_co_param_grasp.get(foliation_1.foliation_name) and self.selected_co_param_grasp.get(foliation_2.foliation_name):
            if len(self.selected_co_param_grasp.get(foliation_1.foliation_name)) == len(self.selected_co_param_grasp.get(foliation_2.foliation_name)):
                selected_co_parameters1_index = random.randint(0, len(foliation_1.co_parameters) - 1)
            else:
                temp_list = [item for item in self.selected_co_param_grasp.get(foliation_1.foliation_name) if item != -1]
                selected_co_parameters1_index = random.choice(temp_list)
        else:
            selected_co_parameters1_index = random.randint(0, len(foliation_1.co_parameters) - 1)

        check_result, intersection_motion, placement, selected_co_parameters1_index, selected_co_parameters2_index = (
            self._sample_co_param(foliation_1, self.placements,
                                  selected_co_parameters1_index, placement_index))

        if check_result:
            # storage prev grasp pose
            if foliation_1.foliation_name not in self.selected_co_param_grasp:
                self.selected_co_param_grasp[foliation_1.foliation_name] = []
            if foliation_2.foliation_name not in self.selected_co_param_grasp:
                self.selected_co_param_grasp[foliation_2.foliation_name] = []

            if selected_co_parameters1_index not in self.selected_co_param_grasp[foliation_1.foliation_name]:
                self.selected_co_param_grasp[foliation_1.foliation_name].append(selected_co_parameters1_index)
            else:
                self.selected_co_param_grasp[foliation_1.foliation_name].append(-1)

            if selected_co_parameters1_index not in self.selected_co_param_grasp[foliation_2.foliation_name]:
                self.selected_co_param_grasp[foliation_2.foliation_name].append(selected_co_parameters1_index)
            else:
                self.selected_co_param_grasp[foliation_2.foliation_name].append(-1)

            return True, selected_co_parameters1_index, selected_co_parameters1_index, ManipulationIntersection(
                'release',
                [intersection_motion[0]],
                self.move_group.get_active_joints(),
                placement,
                self.manipulated_object_mesh_path,
                convert_pose_stamped_to_matrix(self.env_pose),
                self.env_mesh_path
            )
        else:
            return False, selected_co_parameters1_index, selected_co_parameters1_index, None

    def prepare_sampling_func(self):
        self.scene.add_mesh('env_obstacle', self.env_pose, self.env_mesh_path)

    def warp_sampling_func(self):
        self.scene.clear()


class ProblemVisualizer:
    def __init__(self, config, foliated_builder):
        self.env_mesh_path = "package://task_planner/" + config.get('environment', 'env_mesh_path')
        self.manipulated_object_mesh_path = "package://task_planner/" + config.get("environment",
                                                                                   "manipulated_object_mesh_path")
        self.env_pose = create_pose_stamped(config.get('environment', 'env_pose'))
        self.feasible_placements = foliated_builder.feasible_placements

    def visualize_problem(self):
        problem_publisher = rospy.Publisher('/problem_visualization_marker_array', MarkerArray, queue_size=5)
        rospy.sleep(1)

        marker_array = MarkerArray()

        # visualize the obstacle
        obstacle_marker = self.create_marker(self.env_pose.pose, self.env_mesh_path, "obstacle", 0)
        marker_array.markers.append(obstacle_marker)

        # visualize feasible placements
        for i, placement in enumerate(self.feasible_placements):
            placement = msgify(geometry_msgs.msg.Pose, placement)
            object_marker = self.create_marker(placement, self.manipulated_object_mesh_path, "placement", i + 1)
            marker_array.markers.append(object_marker)

        problem_publisher.publish(marker_array)

    @staticmethod
    def create_marker(pose, mesh_path, namespace, marker_id):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.pose = pose
        marker.scale = Point(1, 1, 1)
        marker.color = ColorRGBA(0.5, 0.5, 0.5, 1)
        marker.mesh_resource = mesh_path
        return marker


class Pipeline:

    def __init__(self, package_name):
        self.package_name = package_name
        self.config = None
        self.robot_scene = None
        self.foliated_builder = None
        self.sampler = None
        self.problem_visualizer = None

        self.foliated_intersections = []
        self.foliations = []

    def _init(self):
        self.config = Config(self.package_name)
        self.robot_scene = RobotScene(self.config)
        self.foliated_builder = FoliatedBuilder(self.config)
        self.sampler = Sampler(self.config, self.robot_scene, self.foliated_builder.feasible_placements)
        self.problem_visualizer = ProblemVisualizer(self.config, self.foliated_builder)
        self._print_info(self.foliated_builder)

    @staticmethod
    def _print_info(foliated_builder):
        print "Foliation Dictionary"
        for key, foliation in foliated_builder.foliation_dict.iteritems():
            print "{}: Foliation Name - {}".format(key, foliation.foliation_name)
        print "--------------------"
        print "Sample Task Queue"
        for task in foliated_builder.sample_task_queue:
            foliation, name = task
            print "Foliation Name - {}, Target - {}".format(foliation.foliation_name, name)

    def _build_foliations(self):
        # TODO: check why can't just use dict
        self.foliations = [self.foliated_builder.foliation_regrasp] + self.foliated_builder.foliation_placement_group

    @staticmethod
    def _get_foliation_co_param_type(foliation):
        return "grasp" if foliation.constraint_parameters.get("is_object_in_hand") else "placement"

    def _build_intersections(self):
        for task in self.foliated_builder.sample_task_queue:
            current_foliation = task[0]
            target_foliation = self.foliated_builder.foliation_dict.get(task[1])

            current_type = self._get_foliation_co_param_type(current_foliation)
            target_type = self._get_foliation_co_param_type(target_foliation)

            if (current_type, target_type) in [("grasp", "placement")]:
                foliated_intersection = FoliatedIntersection(
                    current_foliation, target_foliation,
                    self.sampler.sampling_func_placement_grasp,
                    self.sampler.prepare_sampling_func,
                    self.sampler.warp_sampling_func
                )
            elif current_type == target_type == "grasp":
                foliated_intersection = FoliatedIntersection(
                    current_foliation, target_foliation,
                    self.sampler.sampling_func_placement_placement,
                    self.sampler.prepare_sampling_func,
                    self.sampler.warp_sampling_func
                )
            else:
                print current_type
                print target_type
                raise ValueError("Invalid intersection settings")

            self.foliated_intersections.append(foliated_intersection)

    def _generate_problem(self):
        foliated_problem = FoliatedProblem(self.config.get("task_parameters", 'task_name'))
        foliated_problem.set_foliation_n_foliated_intersection(self.foliations, self.foliated_intersections)
        self.problem_visualizer.visualize_problem()
        foliated_problem.sample_intersections(self.config.get('task_parameters', 'num_samples'))

        # set the start and goal candidates
        start_candidates = []
        goal_candidates = []
        for p in range(len(self.foliated_builder.feasible_placements)):
            start_candidates.append((0, p))
            goal_candidates.append((0, p))

        foliated_problem.set_start_manifold_candidates(start_candidates)
        foliated_problem.set_goal_manifold_candidates(goal_candidates)

        foliated_problem.save(self.config.package_path + self.config.get('task_parameters', 'save_path'))

        FoliatedProblem.load(ManipulationFoliation, ManipulationIntersection,
                             self.config.package_path + self.config.get('task_parameters', 'save_path'))

    def main_pipeline(self):

        self._init()
        self._build_foliations()
        self._build_intersections()
        self._generate_problem()
        self.problem_visualizer.visualize_problem()

        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)


if __name__ == "__main__":
    pipline = Pipeline("task_planner")
    pipline.main_pipeline()
