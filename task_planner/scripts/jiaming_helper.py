import rospy
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import CollisionObject, RobotState, Constraints, OrientationConstraint, PositionConstraint, BoundingVolume, MoveItErrorCodes
import tf.transformations as tf_trans
from shape_msgs.msg import SolidPrimitive, Mesh, MeshTriangle
import trimesh
from trimesh import transformations
import numpy as np
from geometry_msgs.msg import Quaternion, Point, Pose, PoseStamped, Point32

try:
    from pyassimp import pyassimp
except:
    # support pyassimp > 3.0
    try:
        import pyassimp
    except:
        pyassimp = False
        print(
            "Failed to import pyassimp, see https://github.com/ros-planning/moveit/issues/86 for more info"
        )

############################# CONSTANTS #############################
# Fetch robot constants
GRIPPER_ROTATION = np.array([[1, 0, 0, -0.17], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
INIT_JOINT_NAMES = [
        "torso_lift_joint",
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "upperarm_roll_joint",
        "elbow_flex_joint",
        "wrist_flex_joint",
        "l_gripper_finger_joint",
        "r_gripper_finger_joint",
    ]
INIT_JOINT_POSITIONS = [0.3, -1.28, 1.52, 0.35, 1.81, 1.47, 0.04, 0.04]
END_EFFECTOR_LINK = "wrist_roll_link"
TOUCH_LINKS = ["l_gripper_finger_link", "r_gripper_finger_link", "gripper_link"]
#####################################################################

# convert a list of joint values to robotTrajectory
def convert_joint_values_to_robot_trajectory(joint_values_list_, joint_names_):
    robot_trajectory = RobotTrajectory()
    robot_trajectory.joint_trajectory = JointTrajectory()
    robot_trajectory.joint_trajectory.joint_names = joint_names_

    for i in range(len(joint_values_list_)):
        robot_trajectory.joint_trajectory.points.append(JointTrajectoryPoint())
        robot_trajectory.joint_trajectory.points[i].positions = joint_values_list_[i]
        robot_trajectory.joint_trajectory.points[i].velocities = [0.0] * len(
            joint_values_list_[i]
        )
        robot_trajectory.joint_trajectory.points[i].accelerations = [0.0] * len(
            joint_values_list_[i]
        )
        robot_trajectory.joint_trajectory.points[i].time_from_start = rospy.Duration(
            0.1 * i
        )

    return robot_trajectory

def convert_joint_values_to_robot_state(joint_values_list_, joint_names_, robot_):
    """
    convert a list of joint values to robotState
    joint_values_list_: a list of joint values
    joint_names_: a list of joint names
    robot_: a robotCommander
    """
    moveit_robot_state = robot_.get_current_state()
    position_list = list(moveit_robot_state.joint_state.position)
    for joint_name, joint_value in zip(joint_names_, joint_values_list_):
        position_list[
            moveit_robot_state.joint_state.name.index(joint_name)
        ] = joint_value
    moveit_robot_state.joint_state.position = tuple(position_list)
    return moveit_robot_state

def get_joint_values_from_joint_state(joint_state_, joint_names_):
    """
    get joint values from joint state
    joint_state_: a joint state
    joint_names_: a list of joint names
    """
    joint_values = []
    for joint_name in joint_names_:
        joint_values.append(
            joint_state_.position[joint_state_.name.index(joint_name)]
        )
    return joint_values

def get_no_constraint():
    no_constraint = Constraints()
    no_constraint.name = "use_equality_constraints"

    oc = OrientationConstraint()

    oc.parameterization = OrientationConstraint.ROTATION_VECTOR
    oc.header.frame_id = "base_link"
    oc.header.stamp = rospy.Time(0)
    oc.link_name = END_EFFECTOR_LINK
    constrained_quaternion = Quaternion()
    constrained_quaternion.x = 0.0
    constrained_quaternion.y = 0.0
    constrained_quaternion.z = 0.0
    constrained_quaternion.w = 1.0
    oc.orientation = constrained_quaternion
    oc.weight = 1.0

    oc.absolute_x_axis_tolerance = 2 * 3.1415
    oc.absolute_y_axis_tolerance = 2 * 3.1415
    oc.absolute_z_axis_tolerance = 2 * 3.1415
    no_constraint.orientation_constraints.append(oc)

    # need to set in-hand pose
    in_hand_pose = Pose()
    in_hand_pose.position.x = 0.0
    in_hand_pose.position.y = 0.0
    in_hand_pose.position.z = 0.0
    in_hand_pose.orientation.x = 0.0
    in_hand_pose.orientation.y = 0.0
    in_hand_pose.orientation.z = 0.0
    in_hand_pose.orientation.w = 1.0
    no_constraint.in_hand_pose = in_hand_pose
    return no_constraint

def construct_moveit_constraint(in_hand_pose_, constraint_pose_, orientation_constraint_, position_constraint_):
    moveit_quaternion = tf_trans.quaternion_from_matrix(
        constraint_pose_
    )  # return x, y z, w
    moveit_translation = tf_trans.translation_from_matrix(
        constraint_pose_
    )  # return x, y, z

    moveit_constraint = Constraints()
    moveit_constraint.name = "use_equality_constraints"

    oc = OrientationConstraint()

    oc.parameterization = OrientationConstraint.ROTATION_VECTOR
    oc.header.frame_id = "base_link"
    oc.header.stamp = rospy.Time(0)
    oc.link_name = END_EFFECTOR_LINK
    constrained_quaternion = Quaternion()
    constrained_quaternion.x = moveit_quaternion[0]
    constrained_quaternion.y = moveit_quaternion[1]
    constrained_quaternion.z = moveit_quaternion[2]
    constrained_quaternion.w = moveit_quaternion[3]
    oc.orientation = constrained_quaternion
    oc.weight = 1.0

    oc.absolute_x_axis_tolerance = orientation_constraint_[0] * 2.0
    oc.absolute_y_axis_tolerance = orientation_constraint_[1] * 2.0
    oc.absolute_z_axis_tolerance = orientation_constraint_[2] * 2.0

    moveit_constraint.orientation_constraints.append(oc)

    pc = PositionConstraint()
    pc.header.frame_id = "base_link"
    pc.header.stamp = rospy.Time(0)
    pc.link_name = END_EFFECTOR_LINK
    pc.target_point_offset.x = moveit_translation[0]
    pc.target_point_offset.y = moveit_translation[1]
    pc.target_point_offset.z = moveit_translation[2]

    pc.target_point_offset.x = 0.0
    pc.target_point_offset.y = 0.0
    pc.target_point_offset.z = 0.0

    pc.weight = 1.0

    solid_primitive = SolidPrimitive()
    solid_primitive.type = SolidPrimitive.BOX
    solid_primitive.dimensions = [
        position_constraint_[0],
        position_constraint_[1],
        position_constraint_[2],
    ]

    bounding_volume = BoundingVolume()
    bounding_volume.primitives.append(solid_primitive)

    bounding_volume_pose = Pose()
    bounding_volume_pose.position.x = moveit_translation[0]
    bounding_volume_pose.position.y = moveit_translation[1]
    bounding_volume_pose.position.z = moveit_translation[2]

    bounding_volume_pose.orientation.x = moveit_quaternion[0]
    bounding_volume_pose.orientation.y = moveit_quaternion[1]
    bounding_volume_pose.orientation.z = moveit_quaternion[2]
    bounding_volume_pose.orientation.w = moveit_quaternion[3]

    bounding_volume.primitive_poses.append(bounding_volume_pose)

    pc.constraint_region = bounding_volume

    moveit_constraint.position_constraints.append(pc)

    # convert in_hand_pose_ from matrix to Pose
    in_hand_quaternion = tf_trans.quaternion_from_matrix(
        in_hand_pose_
    )  # return x, y z, w
    in_hand_translation = tf_trans.translation_from_matrix(
        in_hand_pose_
    )  # return x, y, z

    in_hand_pose = Pose()
    in_hand_pose.position.x = in_hand_translation[0]
    in_hand_pose.position.y = in_hand_translation[1]
    in_hand_pose.position.z = in_hand_translation[2]

    in_hand_pose.orientation.x = in_hand_quaternion[0]
    in_hand_pose.orientation.y = in_hand_quaternion[1]
    in_hand_pose.orientation.z = in_hand_quaternion[2]
    in_hand_pose.orientation.w = in_hand_quaternion[3]

    moveit_constraint.in_hand_pose = in_hand_pose

    return moveit_constraint

def make_mesh(name, pose, filename, scale=(1, 1, 1)):
    co = CollisionObject()
    if pyassimp is False:
        raise MoveItCommanderException(
            "Pyassimp needs patch https://launchpadlibrarian.net/319496602/patchPyassim.txt"
        )
    scene = pyassimp.load(filename)
    if not scene.meshes or len(scene.meshes) == 0:
        raise MoveItCommanderException("There are no meshes in the file")
    if len(scene.meshes[0].faces) == 0:
        raise MoveItCommanderException("There are no faces in the mesh")
    co.operation = CollisionObject.ADD
    co.id = name
    co.header = pose.header
    co.pose = pose.pose

    mesh = Mesh()
    first_face = scene.meshes[0].faces[0]
    if hasattr(first_face, "__len__"):
        for face in scene.meshes[0].faces:
            if len(face) == 3:
                triangle = MeshTriangle()
                triangle.vertex_indices = [face[0], face[1], face[2]]
                mesh.triangles.append(triangle)
    elif hasattr(first_face, "indices"):
        for face in scene.meshes[0].faces:
            if len(face.indices) == 3:
                triangle = MeshTriangle()
                triangle.vertex_indices = [
                    face.indices[0],
                    face.indices[1],
                    face.indices[2],
                ]
                mesh.triangles.append(triangle)
    else:
        raise MoveItCommanderException(
            "Unable to build triangles from mesh due to mesh object structure"
        )
    for vertex in scene.meshes[0].vertices:
        point = Point()
        point.x = vertex[0] * scale[0]
        point.y = vertex[1] * scale[1]
        point.z = vertex[2] * scale[2]
        mesh.vertices.append(point)
    co.meshes = [mesh]
    identity_pose = Pose()
    identity_pose.orientation.w = 1.0
    identity_pose.position.x = 0.0
    identity_pose.position.y = 0.0
    identity_pose.position.z = 0.0

    co.mesh_poses = [identity_pose]
    pyassimp.release(scene)
    return co

def convert_pose_stamped_to_matrix(pose_stamped):
    pose_matrix = transformations.quaternion_matrix([pose_stamped.pose.orientation.w,
                                                     pose_stamped.pose.orientation.x,
                                                     pose_stamped.pose.orientation.y,
                                                     pose_stamped.pose.orientation.z])
    pose_matrix[0, 3] = pose_stamped.pose.position.x
    pose_matrix[1, 3] = pose_stamped.pose.position.y
    pose_matrix[2, 3] = pose_stamped.pose.position.z
    return pose_matrix

def create_pose_stamped(pose_data):
    pose = create_pose_stamped_from_raw(pose_data['frame_id'], pose_data['position'][0], pose_data['position'][1],
                                        pose_data['position'][2], pose_data['orientation'][0],
                                        pose_data['orientation'][1], pose_data['orientation'][2],
                                        pose_data['orientation'][3])
    return pose

def create_pose_stamped_from_raw(frame_id, x, y, z, o_x, o_y, o_z, o_w):
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z
    pose.pose.orientation.x = o_x
    pose.pose.orientation.y = o_y
    pose.pose.orientation.z = o_z
    pose.pose.orientation.w = o_w
    return pose

def gaussian_similarity(distance, max_distance, sigma=0.01):
    """
    Calculate the similarity score using Gaussian function.
    distance: the distance between two configurations
    sigma: the sigma of the Gaussian function
    max_distance: the maximum distance between two configurations
    The score is between 0 and 1. The larger the score, the more similar the two configurations are.
    If sigma is heigher, the scope of the Gaussian function is wider.
    """
    if distance == 0:  # when the distance is 0, the score should be 1
        return 1.0

    # Calculate the similarity score using Gaussian function
    score = np.exp(-(distance ** 2) / (2 * sigma ** 2))
    max_score = np.exp(-(max_distance ** 2) / (2 * sigma ** 2))
    score = (score - max_score) / (1 - max_score)

    if score < 0.001:
        score = 0.0

    return score

def generate_similarity_matrix(configuration_list, difference_function):
    '''
    based on the difference function to generate the similarity matrix for the configuration list
    '''
    different_matrix = np.zeros((len(configuration_list), len(configuration_list)))
    for i in range(len(configuration_list)):
        for j in range(len(configuration_list)):
            if i == j:
                different_matrix[i, j] = 0
            different_matrix[i, j] = difference_function(
                configuration_list[i], configuration_list[j]
            )

    similarity_matrix = np.zeros((len(configuration_list), len(configuration_list)))
    max_distance = np.max(different_matrix)
    for i in range(len(configuration_list)):
        for j in range(len(configuration_list)):
            similarity_matrix[i, j] = gaussian_similarity(
                different_matrix[i, j], max_distance, sigma=0.01
            )
    return similarity_matrix

def collision_check(collision_manager, obj_mesh, obj_pose):
    obj_mesh = trimesh.load_mesh(obj_mesh)
    obj_mesh.apply_transform(convert_pose_stamped_to_matrix(obj_pose))
    collision_manager.add_object('obj', obj_mesh)

    if not collision_manager.in_collision_internal():
        collision_manager.remove_object('obj')
        return True

    collision_manager.remove_object('obj')
    return False

def create_rotation_matrix_from_euler(orientation, position):
    rotation_matrix = tf_trans.euler_matrix(orientation[0], orientation[1], orientation[2])[:3, :3]

    reference_pose = np.identity(4)
    reference_pose[:3, :3] = rotation_matrix
    reference_pose[:3, 3] = position

    return reference_pose