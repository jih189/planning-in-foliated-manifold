from foliated_base_class import BaseFoliation, BaseIntersection
import copy
import json
import numpy as np
from jiaming_visualizer import ManipulationTaskMotion
from jiaming_helper import convert_joint_values_to_robot_trajectory
from ros_numpy import numpify, msgify
from geometry_msgs.msg import Pose


# define the intersection class
class ManipulationIntersection(BaseIntersection):
    def __init__(
        self,
        action,
        motion,
        active_joints,
        object_pose=None,
        object_mesh_path=None,
        obstacle_pose=None,
        obstacle_mesh_path=None,
    ):
        self.action = action
        self.motion = motion
        self.active_joints = active_joints
        self.object_pose = object_pose
        self.object_mesh_path = object_mesh_path
        self.obstacle_pose = obstacle_pose
        self.obstacle_mesh_path = obstacle_mesh_path

    def inverse(self):
        if self.action == "grasp":
            return ManipulationIntersection(
                action="release",
                motion=self.motion[::-1],
                active_joints=self.active_joints,
                object_pose=self.object_pose,
                object_mesh_path=self.object_mesh_path,
                obstacle_pose=self.obstacle_pose,
                obstacle_mesh_path=self.obstacle_mesh_path,
            )
        else:
            return ManipulationIntersection(
                action="grasp",
                motion=self.motion[::-1],
                active_joints=self.active_joints,
                object_pose=self.object_pose,
                object_mesh_path=self.object_mesh_path,
                obstacle_pose=self.obstacle_pose,
                obstacle_mesh_path=self.obstacle_mesh_path,
            )

    def get_edge_configurations(self):
        return self.motion[0], self.motion[-1]

    def save(self, file_path):
        # need to save the foliation name, co_parameter_index, action, motion
        (
            foliation1_name,
            co_parameter1_index,
            foliation2_name,
            co_parameter2_index,
        ) = self.get_foliation_names_and_co_parameter_indexes()

        intersection_data = {
            "foliation1_name": foliation1_name,
            "co_parameter1_index": co_parameter1_index,
            "foliation2_name": foliation2_name,
            "co_parameter2_index": co_parameter2_index,
            "action": self.action,
            "motion": [m.tolist() for m in self.motion],
            "active_joints": self.active_joints,
            "object_pose": self.object_pose.tolist(),  # convert numpy array to list
            "object_mesh_path": self.object_mesh_path,
            "obstacle_pose": self.obstacle_pose.tolist(),  # convert numpy array to list
            "obstacle_mesh_path": self.obstacle_mesh_path,
        }

        # create a json file
        with open(file_path, "w") as json_file:
            json.dump(intersection_data, json_file)

    def get_task_motion(self):
        return ManipulationTaskMotion(
            self.action,
            planned_motion=convert_joint_values_to_robot_trajectory(
                self.motion, self.active_joints
            ),
            has_object_in_hand=False,
            object_pose=self.object_pose,
            object_mesh_path=self.object_mesh_path,
            obstacle_pose=self.obstacle_pose,
            obstacle_mesh_path=self.obstacle_mesh_path,
        )

    @staticmethod
    def load(file_path):
        with open(file_path, "r") as json_file:
            intersection_data = json.load(json_file)

        loaded_intersection = ManipulationIntersection(
            action=intersection_data.get("action"),
            motion=[np.array(m) for m in intersection_data.get("motion")],
            active_joints=intersection_data.get("active_joints"),
            object_pose=np.array(intersection_data.get("object_pose")),
            object_mesh_path=intersection_data.get("object_mesh_path"),
            obstacle_pose=np.array(intersection_data.get("obstacle_pose")),
            obstacle_mesh_path=intersection_data.get("obstacle_mesh_path"),
        )

        foliation1_name = intersection_data.get("foliation1_name")
        co_parameter1_index = intersection_data.get("co_parameter1_index")
        foliation2_name = intersection_data.get("foliation2_name")
        co_parameter2_index = intersection_data.get("co_parameter2_index")

        loaded_intersection.set_foliation_names_and_co_parameter_indexes(
            foliation1_name, co_parameter1_index, foliation2_name, co_parameter2_index
        )

        return loaded_intersection


# define the foliation class
class ManipulationFoliation(BaseFoliation):
    def save(self, dir_path):
        # save foliation name, constraint_parameters, co_parameters

        # if a value in constraint_parameters is a numpy array, convert it to list
        copy_constraint_parameters = copy.deepcopy(self.constraint_parameters)
        for key, value in copy_constraint_parameters.items():
            if isinstance(value, np.ndarray):
                copy_constraint_parameters[key] = value.tolist()

        foliation_data = {
            "foliation_name": self.foliation_name,
            "constraint_parameters": copy_constraint_parameters,
            "co_parameters": [
                c.tolist() for c in self.co_parameters
            ],  # convert numpy array to list
            "similarity_matrix": self.similarity_matrix.tolist(),  # convert numpy array to list
        }

        # create a json file
        with open(dir_path + "/" + self.foliation_name + ".json", "w") as json_file:
            json.dump(foliation_data, json_file)

    @staticmethod
    def load(file_path):
        with open(file_path, "r") as json_file:
            foliation_data = json.load(json_file)

        # if a value in constraint_parameters is a list with 4x4 size, convert it to numpy array
        copy_constraint_parameters = copy.deepcopy(
            foliation_data.get("constraint_parameters")
        )
        for key, value in copy_constraint_parameters.items():
            if isinstance(value, list):
                # check if the list can be converted to numpy array
                try:
                    m = np.array(value)
                    # check if the matrix m is 4x4
                    if m.shape == (4, 4):
                        copy_constraint_parameters[key] = m
                except:
                    pass

        return ManipulationFoliation(
            foliation_name=foliation_data.get("foliation_name"),
            constraint_parameters=copy_constraint_parameters,
            co_parameters=[
                np.array(c) for c in foliation_data.get("co_parameters")
            ],  # convert list to numpy array
            similarity_matrix=np.array(
                foliation_data.get("similarity_matrix")
            ),  # convert list to numpy array
        )
