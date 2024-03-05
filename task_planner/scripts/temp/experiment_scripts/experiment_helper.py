#!/usr/bin/env python3
"""
experiment helper
------------------
Here are the helper classes to save and load experiments. Thus, later the task planner
can use this code to load and construct the task graph.
"""
import os
import numpy as np
import sys
import rospkg
import json


class Manifold:
    """
    Manifold is the class describe all information of a constraint manifold in foliations.
    """

    def __init__(
        self,
        foliation_id_,
        manifold_id_,
        object_name_,
        object_mesh_,
        has_object_in_hand_,
    ):
        """
        Initialize the class instance.
        """
        self.foliation_id = foliation_id_
        self.manifold_id = manifold_id_
        self.object_name = object_name_
        self.object_mesh = object_mesh_
        self.has_object_in_hand = has_object_in_hand_

    def add_constraint(
        self,
        in_hand_pose_,
        constraint_pose_,
        orientation_constraint_,
        position_constraint_,
    ):
        """
        Add a constraint to the manifold.
        """
        if not self.has_object_in_hand:
            raise ValueError(
                "The object is not in the hand, thus the constraint cannot be added."
            )
        self.in_hand_pose = in_hand_pose_
        self.constraint_pose = constraint_pose_
        self.orientation_constraint = orientation_constraint_
        self.position_constraint = position_constraint_

    def add_object_placement(self, object_pose_):
        """
        Add the object placement constraint to the manifold.
        """
        if self.has_object_in_hand:
            raise ValueError(
                "The object is in the hand, thus the object placement constraint cannot be added."
            )
        self.object_pose = object_pose_


class Intersection:
    """
    Intersection describes the motion across two different manifolds.
    """

    def __init__(
        self,
        foliation_id_1_,
        manifold_id_1_,
        foliation_id_2_,
        manifold_id_2_,
        has_object_in_hand_,
        trajectory_motion_,
        in_hand_pose_,
        object_mesh_,
        object_name_,
    ):
        self.foliation_id_1 = foliation_id_1_
        self.manifold_id_1 = manifold_id_1_
        self.foliation_id_2 = foliation_id_2_
        self.manifold_id_2 = manifold_id_2_
        self.has_object_in_hand = has_object_in_hand_
        self.trajectory_motion = trajectory_motion_
        self.in_hand_pose = in_hand_pose_  # if the object is not in hand, then in_hand_pose is the object placement pose.
        self.object_mesh = object_mesh_
        self.object_name = object_name_


class Experiment:
    """
    Experiment contains information of all manifolds and intersections.
    """

    def __init__(self):
        self.has_setup = False
        self.has_set_start_and_goal_foliation_and_manifold_id = False

    def setup(
        self,
        experiment_name_,
        manipulated_object_mesh_,
        obstacle_mesh_,
        obstacle_mesh_pose_,
        initial_robot_state_,
        joint_names_,
        active_joint_names_,
    ):
        self.experiment_name = experiment_name_
        self.manifolds = []
        self.intersections = []
        self.obstacle_mesh = obstacle_mesh_
        self.obstacle_mesh_pose = obstacle_mesh_pose_
        self.initial_robot_state = initial_robot_state_  # the initial robot state
        self.joint_names = joint_names_  # the joint names of the robot
        self.active_joint_names = active_joint_names_  # the joint names of arm group.
        self.has_setup = True
        self.manipulated_object_mesh = manipulated_object_mesh_

    def add_manifold(self, manifold):
        if not self.has_setup:
            raise ValueError("The experiment has not been setup yet.")

        self.manifolds.append(manifold)

    def add_intersection(self, intersection):
        if not self.has_setup:
            raise ValueError("The experiment has not been setup yet.")

        self.intersections.append(intersection)

    def save(self, dir_name):
        """
        Save the experiment to a file.
        """
        if not self.has_setup:
            raise ValueError("The experiment has not been setup yet.")

        if not self.has_set_start_and_goal_foliation_and_manifold_id:
            raise ValueError(
                "The start and goal foliation, manifold id have not been set yet."
            )

        # check if the directory exists
        # if so, then delete it
        if os.path.exists(dir_name):
            # delete the directory
            os.system("rm -rf " + dir_name)

        os.makedirs(dir_name)

        # Data to be saved
        experiment_data = {
            "experiment_name": self.experiment_name,
            "obstacle_mesh": self.obstacle_mesh,
            "obstacle_mesh_pose": self.obstacle_mesh_pose.tolist(),
            "initial_robot_state": self.initial_robot_state,
            "joint_names": self.joint_names,
            "active_joint_names": self.active_joint_names,
            "manipulated_object_mesh": self.manipulated_object_mesh,
            "manifolds": [],
        }

        # add manifolds into experiment data
        for manifold in self.manifolds:
            manifold_data = {
                "foliation_id": manifold.foliation_id,
                "manifold_id": manifold.manifold_id,
                "object_name": manifold.object_name,
                "object_mesh": manifold.object_mesh,
                "has_object_in_hand": manifold.has_object_in_hand,
            }
            if manifold.has_object_in_hand:
                manifold_data["in_hand_pose"] = manifold.in_hand_pose.tolist()
                manifold_data["constraint_pose"] = manifold.constraint_pose.tolist()
                manifold_data[
                    "orientation_constraint"
                ] = manifold.orientation_constraint.tolist()
                manifold_data[
                    "position_constraint"
                ] = manifold.position_constraint.tolist()
            else:
                manifold_data["object_pose"] = manifold.object_pose.tolist()
            experiment_data["manifolds"].append(manifold_data)

        # Save data to a JSON file
        with open(dir_name + "/manifolds.json", "w") as file:
            json.dump(experiment_data, file)

        # Save the interface into anther file
        if len(self.intersections) > 0:
            for intersection_id in range(len(self.intersections)):
                intersection_data = {
                    "foliation_id_1": self.intersections[
                        intersection_id
                    ].foliation_id_1,
                    "manifold_id_1": self.intersections[intersection_id].manifold_id_1,
                    "foliation_id_2": self.intersections[
                        intersection_id
                    ].foliation_id_2,
                    "manifold_id_2": self.intersections[intersection_id].manifold_id_2,
                    "has_object_in_hand": self.intersections[
                        intersection_id
                    ].has_object_in_hand,
                    "trajectory_motion": self.intersections[
                        intersection_id
                    ].trajectory_motion.tolist(),
                    "in_hand_pose": self.intersections[
                        intersection_id
                    ].in_hand_pose.tolist(),
                    "object_mesh": self.intersections[intersection_id].object_mesh,
                    "object_name": self.intersections[intersection_id].object_name,
                }

                with open(
                    dir_name + "/intersection_" + str(intersection_id) + ".json", "w"
                ) as file:
                    json.dump(intersection_data, file)

        # save the start and goal's foliation and manifold id
        start_and_goal_data = {
            "start_foliation_id": self.start_foliation_id,
            "start_manifold_id": self.start_manifold_id,
            "goal_foliation_id": self.goal_foliation_id,
            "goal_manifold_id": self.goal_manifold_id,
        }

        with open(dir_name + "/start_and_goal.json", "w") as file:
            json.dump(start_and_goal_data, file)

    def load(self, dir_name):
        """
        Load the experiment from a file.
        """
        print("Loading experiment from " + dir_name)

        # check if the directory exists
        # if not, then raise error
        if not os.path.exists(dir_name):
            raise ValueError("The experiment directory does not exist.")

        # check if the manifolds file exists
        # if not, then raise error
        if not os.path.exists(dir_name + "/manifolds.json"):
            raise ValueError("The manifolds file does not exist.")

        # load the manifolds file
        with open(dir_name + "/manifolds.json", "r") as file:
            experiment_data = json.load(file)
            self.experiment_name = experiment_data["experiment_name"]
            self.obstacle_mesh = experiment_data["obstacle_mesh"]
            self.obstacle_mesh_pose = np.array(experiment_data["obstacle_mesh_pose"])
            self.initial_robot_state = experiment_data["initial_robot_state"]
            self.joint_names = experiment_data["joint_names"]
            self.active_joint_names = experiment_data["active_joint_names"]
            self.manipulated_object_mesh = experiment_data["manipulated_object_mesh"]

            # load manifolds
            self.manifolds = []
            for manifold_data in experiment_data["manifolds"]:
                current_manifold = Manifold(
                    manifold_data["foliation_id"],
                    manifold_data["manifold_id"],
                    manifold_data["object_name"],
                    manifold_data["object_mesh"],
                    manifold_data["has_object_in_hand"],
                )

                if manifold_data["has_object_in_hand"]:
                    current_manifold.add_constraint(
                        np.array(manifold_data["in_hand_pose"]),
                        np.array(manifold_data["constraint_pose"]),
                        np.array(manifold_data["orientation_constraint"]),
                        np.array(manifold_data["position_constraint"]),
                    )
                else:
                    current_manifold.add_object_placement(
                        np.array(manifold_data["object_pose"])
                    )

                self.manifolds.append(current_manifold)

        # check the number of files start with "intersection_" in the directory
        intersection_files = [
            f
            for f in os.listdir(dir_name)
            if os.path.isfile(os.path.join(dir_name, f))
            and f.startswith("intersection_")
        ]
        print("Number of intersection files: " + str(len(intersection_files)))

        # load the intersection files
        self.intersections = []
        for intersection_file in intersection_files:
            with open(dir_name + "/" + intersection_file, "r") as file:
                intersection_data = json.load(file)
                current_intersection = Intersection(
                    intersection_data["foliation_id_1"],
                    intersection_data["manifold_id_1"],
                    intersection_data["foliation_id_2"],
                    intersection_data["manifold_id_2"],
                    intersection_data["has_object_in_hand"],
                    np.array(intersection_data["trajectory_motion"]),
                    np.array(intersection_data["in_hand_pose"]),
                    intersection_data["object_mesh"],
                    intersection_data["object_name"],
                )
                self.intersections.append(current_intersection)

        # load the start and goal data
        with open(dir_name + "/start_and_goal.json", "r") as file:
            start_and_goal_data = json.load(file)
            self.start_foliation_id = start_and_goal_data["start_foliation_id"]
            self.start_manifold_id = start_and_goal_data["start_manifold_id"]
            self.goal_foliation_id = start_and_goal_data["goal_foliation_id"]
            self.goal_manifold_id = start_and_goal_data["goal_manifold_id"]

        self.has_setup = True
        self.has_set_start_and_goal_foliation_and_manifold_id = True

    def print_experiment_data(self):
        """
        Print the experiment data.
        """
        if not self.has_setup:
            raise ValueError("The experiment has not been setup yet.")

        print("Experiment name: " + self.experiment_name)

        print("Manifolds:")
        for manifold in self.manifolds:
            print("Foliation id: " + str(manifold.foliation_id))
            print("Manifold id: " + str(manifold.manifold_id))
            print("Object name: " + str(manifold.object_name))
            print("Object mesh: " + str(manifold.object_mesh))
            print("Has object in hand: " + str(manifold.has_object_in_hand))
            if manifold.has_object_in_hand:
                print("In hand pose: " + str(manifold.in_hand_pose))
                print("Constraint pose: " + str(manifold.constraint_pose))
                print("Orientation constraint: " + str(manifold.orientation_constraint))
                print("Position constraint: " + str(manifold.position_constraint))
            else:
                print("Object pose: " + str(manifold.object_pose))

    def set_start_and_goal_foliation_manifold_id(
        self,
        start_foliation_id_,
        start_manifold_id_,
        goal_foliation_id_,
        goal_manifold_id_,
    ):
        """
        Set the start and goal manifold id.
        We suggest you run this function after you set all manifolds.
        """
        # need to check if the start and goal manifold id are valid
        found_start_manifold = False
        found_goal_manifold = False
        for manifold in self.manifolds:
            if (
                manifold.foliation_id == start_foliation_id_
                and manifold.manifold_id == start_manifold_id_
            ):
                found_start_manifold = True
            if (
                manifold.foliation_id == goal_foliation_id_
                and manifold.manifold_id == goal_manifold_id_
            ):
                found_goal_manifold = True

        if not found_start_manifold:
            raise ValueError("The start manifold id is invalid.")

        if not found_goal_manifold:
            raise ValueError("The goal manifold id is invalid.")

        self.start_manifold_id = start_manifold_id_
        self.goal_manifold_id = goal_manifold_id_
        self.start_foliation_id = start_foliation_id_
        self.goal_foliation_id = goal_foliation_id_

        self.has_set_start_and_goal_foliation_and_manifold_id = True


if __name__ == "__main__":
    # Create a experiment and save it to a file
    print("Create a experiment and save it to a file")
    rospack = rospkg.RosPack()
    # Get the path of the desired package
    package_path = rospack.get_path("task_planner")

    # check python version
    # experiment = Experiment("pick_and_place")

    # for pick and place manipulation, there are three foliations.
    # foliation 1(pre-grasp): the object is on the table and the robot hand is empty
    pre_grasp_manifold = Manifold(
        0,  # foliation id
        0,  # manifold id
        "mug",  # object name
        "mug",  # object mesh
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # in hand pose
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # constraint pose
        [0.0, np.pi, 0.0],  # orientation constraint
        [0.0, 0.0, 0.0],
    )  # position constraint
    # experiment.add_manifold(pre_grasp_manifold)
    # foliation 2(grasp): the object is grasped by the robot hand
    # foliation 3(post-grasp): the object is placed on the table and the robot hand is empty

    # experiment.save(package_path + "/experiment_dir/pick_and_place")
