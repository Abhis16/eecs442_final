# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg

class UKF(object):
    """
    Unscented Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim = 5
        self._ndim = ndim
        self._no_sigma_points = 2 * ndim + 1
        self._lamda = 3 - (ndim + 2)
        self._update_mat = np.eye(2, ndim)
        self._std_weight_position = 1. / 80
        self._std_weight_velocity = 1. / 600
        self._std_weight_acceleration = 1. / 800
        self.height = 0
        self.sigma_points = np.zeros((self._no_sigma_points + 4, self._ndim + 2))
        self.predicted_sigma_points = np.zeros((self._no_sigma_points + 4, self._ndim + 2))

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement[:2]
        mean_vel = 0
        mean_yaw = np.zeros(2)
        # we wanna use r_ instead of concat or hstack because mean_vel is not an np array
        mean = np.r_[mean_pos, mean_vel, mean_yaw]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_velocity * measurement[3],
            1e-6,
            1e-9,
        ]
        self.height = measurement[3]
        covariance = np.diag(np.square(std))
        return mean, covariance
    
    def generate_sigma_point(self, mean, covariance):
        sigma_points = np.zeros((self._no_sigma_points + 4, self._ndim + 2))
        sigma_points[0] = mean
        L = np.linalg.cholesky(covariance)
        for i in range(0, self._ndim + 2):
            sigma_points[i + 1] = mean + np.sqrt(self._ndim + 2 + self._lamda) * L[i]
            sigma_points[i + 1 + self._ndim + 2] = mean - np.sqrt(self._ndim + 2 + self._lamda) * L[i]
        return sigma_points


    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        
        mean_aug = np.zeros(7)
        mean_aug[:5] = mean
        covariance_aug = np.zeros((7, 7))
        covariance_aug[:5, :5] = covariance
        std = [
            self._std_weight_acceleration * self.height,
            1e-6
        ]
        covariance_aug[5:, 5:] = np.diag(np.square(std))
        
        self.sigma_points = self.generate_sigma_point(mean_aug, covariance_aug)
        predicted_sigma_points = np.zeros((self._no_sigma_points + 4, self._ndim))
        
        for i in range(self._no_sigma_points + 4):
            x = self.sigma_points[i]
            if x[4] < 1e-20:
                x[0] += x[2] * np.cos(x[3]) + 0.5 * x[5] * np.cos(x[3])
                x[1] += x[2] * np.sin(x[3]) + 0.5 * x[5] * np.sin(x[3])
                x[3] += x[4] + 0.5 * x[6]
            else:
                x[0] += x[2] / x[4] * (np.sin(x[3] + x[4]) - np.sin(x[3])) + 0.5 * x[5] * np.cos(x[3])
                x[1] += x[2] / x[4] * (-np.cos(x[3] + x[4]) + np.cos(x[3])) + 0.5 * x[5] * np.sin(x[3])
                x[3] += x[4] + 0.5 * x[6]
            x[4] += x[6]
            x[2] += x[5]
            predicted_sigma_points[i] = x[:5]
        

        weights = np.zeros(self._no_sigma_points + 4)
        weights[0] = self._lamda / (self._ndim + 2 + self._lamda)
        weights[1:] = 0.5 / (self._ndim + 2 + self._lamda)
        mean_aug = np.dot(weights, predicted_sigma_points)
        
        self.predicted_sigma_points = predicted_sigma_points

        weights = np.diag(weights)
        covariance_aug = np.linalg.multi_dot(
            ((mean_aug.T - self.predicted_sigma_points).T, weights, (mean_aug.T - self.predicted_sigma_points)))
        return mean_aug, covariance_aug

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        projected_sigma_points = self.predicted_sigma_points[:, :2].copy()

        weights = np.zeros(self._no_sigma_points + 4)
        weights[0] = self._lamda / (self._ndim+2 + self._lamda)
        weights[1:] = 0.5 / (self._ndim + 2 + self._lamda)
        projected_mean = np.dot(weights, projected_sigma_points)

        weights = np.diag(weights)
        covariance = np.linalg.multi_dot(
            ((projected_mean.T - projected_sigma_points).T, weights, (projected_mean.T - projected_sigma_points)))
        std = [
            self._std_weight_position * height,
            self._std_weight_position * height,
        ]
        innovation_cov = np.diag(np.square(std))
        correlation = np.linalg.multi_dot(
            ((mean.T - predicted_sigma_points).T, weights, (projected_mean.T - projected_sigma_points)))
        return projected_mean, covariance + innovation_cov, correlation

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_covariance, correlation = self.project(mean, covariance,
                                                                         measurement[3], self.predicted_sigma_points)
        kalman_gain = np.linalg.multi_dot((correlation, np.linalg.inv(projected_covariance)))

        innovation = measurement[:2] - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_covariance, kalman_gain.T))

        if (np.abs(new_mean[0] - measurement[0]) > 1 / 2 * np.abs(innovation[0])
                or np.abs(new_mean[1] - measurement[1]) > 2 / 3 * np.abs(innovation[1])):
            new_mean[0] = 1 / 3 * new_mean[0] + 2 / 3 * measurement[0]
            new_mean[1] = 1 / 3 * new_mean[1] + 2 / 3 * measurement[1]
        self.height = measurement[3]
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance, _ = self.project(mean, covariance, height, self.predicted_sigma_points)
        measurements = measurements[:, :2]
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        qr_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            qr_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
    
    