from sklearn import mixture
import pickle
import glob
import numpy as np
import os
from tqdm import tqdm

# import matplotlib.pyplot as plt

# edit the following
generated_configs_dir_path = (
    "/root/catkin_ws/src/jiaming_manipulation/task_planner/empty_world_trajectory_data/"
)
use_dirichlet = False


def _get_all_paths_from_environment(directory):
    """
    Returns a list of all paths from a given environment
    Args:
        directory (str): path to the directory containing all generated paths for an environment.
                   The paths should be stored as .p files.
    Returns:
        list of paths List[str]
    """
    return sorted(glob.glob("%s/*.p" % directory))


def _get_paths_from_environments_dict(directory):
    """
    Returns a dictionary of all paths from all environments
    Args:
        directory (str): path to the directory containing all generated paths for all environments.
                   The directory is of the structure env_000000, env_000001, etc.
                   And each env_XXXXXX directory contains all the paths for that environment.
    Returns:
        dictionary of all paths flattened (Dict[str, List[str]])
    """
    env_names = sorted(glob.glob(os.path.join(directory, "env*")))
    return {
        env_name: _get_all_paths_from_environment(env_name) for env_name in env_names
    }


def _get_all_paths_from_all_environments(directory):
    """
    Returns a list of all paths from all environments
    Args:
        directory (str): path to the directory containing all generated paths for all environments.
                   The directory is of the structure env_000000, env_000001, etc.
                   And each env_XXXXXX directory contains all the paths for that environment.
    Returns:
        list of all paths flattened (List[str])
    """
    env_names = sorted(glob.glob(os.path.join(directory, "env*")))
    return [
        path
        for env_name in env_names
        for path in _get_all_paths_from_environment(env_name)
    ]


def _return_motion_from_path(path):
    """
    Returns the motion from a given path
    Args:
        path (str): path to the .p file containing the motion
    Returns:
        motion (List[float]) - a list of Nx7 elements, where N is the number of waypoints in the motion
    """
    with open(path, "rb") as f:
        return pickle.load(f)["path"]


def read_all_motions_from_all_environments(directory):
    """
    Returns 2-level list of all motions from directory
    Args:
        directory (str): path to the directory containing all generated paths for all environments.
                   The directory is of the structure env_000000, env_000001, etc.
                   And each env_XXXXXX directory contains all the paths for that environment.
    Returns:
        list of all motions (List[List[float]])
    """
    paths = _get_all_paths_from_all_environments(directory)
    return [_return_motion_from_path(path) for path in tqdm(paths)]


def find_number_of_components(motions):
    """
    Returns the maximum number of components from all the paths
    Args:
        paths (List[List[float]]): list of all paths
    Returns:
        maximum number of components from all the paths (int)
    """
    return max([len(motion) for motion in motions])


def convert_all_trajectories_to_numpy(motions):
    """
    Returns a list of all paths converted to numpy arrays
    Args:
        motions (List[List[float]]): list of all motions
    Returns:
        numpy array of (N, 7) which contains all valid configurations
    """
    return np.array([config for motion in motions for config in motion])


def fit_distribution(X, num_components, use_dirichlet=False):
    """
    Returns a GMM or DPGMM fit to the given paths
    Args:
        X (np.array of (N, 7)): list of all paths
        num_components (int): number of components to fit the GMM to
        use_dirichlet (bool): whether to use a DPGMM or not
    Returns:
        GMM fit to the given paths
        Uses either a DPGMM or a GMM depending on the value of use_dirichlet
        (Union[mixture.GaussianMixture, mixture.BayesianGaussianMixture])
    """
    if use_dirichlet:
        return mixture.BayesianGaussianMixture(
            n_components=num_components * 10, random_state=1, verbose=2, n_init=4
        ).fit(X)
    else:
        return mixture.GaussianMixture(
            n_components=num_components * 10, random_state=0, n_init=4, verbose=2
        ).fit(X)


def compute_distribution_edge_information(motions, gmm):
    """
    Returns the edge information for the given paths and GMM fit
    Args:
        motions List[List[float]]: list of all paths
        gmm: GMM fit to the given paths
    Returns:
        dictionary containing the edge information
        (Dict[str, np.ndarray]) - {"edges": np.ndarray of (N, 2), "probabilities": np.ndarray of (N,)}
    """
    edges = []
    for trajectory in motions:
        predicted_distributions = gmm.predict(trajectory)
        path_len = len(trajectory)
        for idx1, idx2 in zip(range(path_len), range(1, path_len)):
            dist1, dist2 = predicted_distributions[idx1], predicted_distributions[idx2]
            if dist1 != dist2:
                edges.append([dist1, dist2])
    edges = np.array(edges)
    unique, counts = np.unique(edges, return_counts=True, axis=0)
    probabilities = counts / np.sum(
        counts
    )  # the probability of the edge e_i occuring amongst all edges
    return {"edges": unique, "probabilities": probabilities}


def save_distribution_to_file(gmm, edge_information, directory):
    """
    Saves the GMM fit to a file
    Args:
        gmm: GMM fit to the given paths
        edge_information: dictionary containing the edge information
        directory: path to the directory where the GMM fit should be saved
    Returns:
        None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    np.save(os.path.join(directory, "weights.npy"), gmm.weights_, allow_pickle=False)
    np.save(os.path.join(directory, "means.npy"), gmm.means_, allow_pickle=False)
    np.save(
        os.path.join(directory, "covariances.npy"), gmm.covariances_, allow_pickle=False
    )
    np.save(
        os.path.join(directory, "edges.npy"),
        edge_information["edges"],
        allow_pickle=False,
    )
    np.save(
        os.path.join(directory, "probabilities.npy"),
        edge_information["probabilities"],
        allow_pickle=False,
    )


def main():
    """
    Main function
    """
    # motions is a multi-level list of all configurations from all environments
    motions = read_all_motions_from_all_environments(generated_configs_dir_path)
    num_components = find_number_of_components(motions)
    all_configurations = convert_all_trajectories_to_numpy(motions)

    print(all_configurations.shape)

    gmm = fit_distribution(all_configurations, int(num_components * 10), use_dirichlet)
    edge_information = compute_distribution_edge_information(motions, gmm)
    save_distribution_to_file(
        gmm,
        edge_information,
        "/root/catkin_ws/src/jiaming_manipulation/task_planner/gmm_related/computed_gmm_directory/",
    )


if __name__ == "__main__":
    main()
