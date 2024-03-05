from sklearn import mixture
import numpy as np


class GaussianDistribution:
    def __init__(
        self,
        mean_,
        covariance_,
    ):
        # Constructor
        self.mean = mean_
        self.covariance = covariance_


class GMM:
    def __init__(self):
        # Constructor
        self.distributions = []
        self.edge_of_distribution = []
        self.edge_probabilities = []
        self._sklearn_gmm = None

        self.collision_free_rates = []

    def get_distribution_index(self, configuration_):
        # find which distribution the configuration belongs to
        # then return the distribution
        # configuration_ is a (d,) element array : (d = 7)
        dist_num = self._sklearn_gmm.predict(configuration_.reshape((1, -1))).squeeze()
        return dist_num.item()

    def get_distribution_indexs(self, configurations_):
        # find which distribution the configuration belongs to
        # then return the distribution
        # configuration_ is a (d,) element array : (d = 7)
        dist_nums = self._sklearn_gmm.predict(configurations_)
        return dist_nums

    def get_distribution(self, configuration_):
        # find which distribution the configuration belongs to
        # then return the distribution
        # configuration_ is a (d,) element array : (d = 7)
        dist_num = self._sklearn_gmm.predict(configuration_.reshape((1, -1))).squeeze()
        # return GaussianDistribution(self._sklearn_gmm.means_[dist_num], self._sklearn_gmm.covariances_[dist_num])
        return self.distributions[dist_num]

    def load_distributions(self, dir_name="../gmm/"):
        means = np.load(dir_name + "means.npy")
        covariances = np.load(dir_name + "covariances.npy")

        # Create an sklearn Gaussian Mixture Model
        self._sklearn_gmm = mixture.GaussianMixture(
            n_components=len(means), covariance_type="full"
        )
        self._sklearn_gmm.precisions_cholesky_ = np.linalg.cholesky(
            np.linalg.inv(covariances)
        )
        self._sklearn_gmm.weights_ = np.load(
            dir_name + "weights.npy"
        )  # how common this distribution is.
        self._sklearn_gmm.means_ = means
        self._sklearn_gmm.covariances_ = covariances

        for mean, covariance in zip(means, covariances):
            self.distributions.append(GaussianDistribution(mean, covariance))
            self.collision_free_rates.append(0.5)
        print("Loaded %d distributions " % len(means), dir_name)
        self.edge_of_distribution = np.load(dir_name + "edges.npy")
        self.edge_probabilities = np.load(dir_name + "edge_probabilities.npy")

    def update_collision_free_rates(self, pointcloud_):
        """
        update the collision-free rate of each distribution.
        """
        pass
