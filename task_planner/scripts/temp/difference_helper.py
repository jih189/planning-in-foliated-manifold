import numpy as np


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
    score = np.exp(-(distance**2) / (2 * sigma**2))
    max_score = np.exp(-(max_distance**2) / (2 * sigma**2))
    score = (score - max_score) / (1 - max_score)

    if score < 0.001:
        score = 0.0

    return score


def get_difference_between_poses(pose_1_, pose_2_):
    """
    Get the difference score between two poses.
    pose_1_ and pose_2_ are both 4x4 numpy matrices.
    """
    position_difference = np.linalg.norm(pose_1_[:3, 3] - pose_2_[:3, 3])

    R = np.dot(pose_1_[:3, :3], pose_2_[:3, :3].T)

    # Calculate the angle of rotation (in radians)
    angle_difference = np.arccos((np.trace(R) - 1) / 2)

    return position_difference + angle_difference


def get_position_difference_between_poses(pose_1_, pose_2_):
    """
    Get the position difference between two poses.
    pose_1_ and pose_2_ are both 4x4 numpy matrices.
    """
    return np.linalg.norm(pose_1_[:3, 3] - pose_2_[:3, 3])
