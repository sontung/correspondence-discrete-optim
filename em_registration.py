import numpy as np
import numbers
import sys
import cv2
from warnings import warn
from PIL import Image


def initialize_sigma2(X, Y):
    (N, D) = X.shape
    (M, _) = Y.shape
    diff = X[None, :, :] - Y[:, None, :]
    err = diff ** 2
    return np.sum(err) / (D * M * N)


def is_positive_semi_definite(R):
    if not isinstance(R, (np.ndarray, np.generic)):
        raise ValueError('Encountered an error while checking if the matrix is positive semi definite. \
            Expected a numpy array, instead got : {}'.format(R))
    return np.all(np.linalg.eigvals(R) > 0)


def color_based_posterior_prob(im, x_mat, ty_mat):
    ty_mat = ty_mat.astype(int)
    x_mat = x_mat.astype(int)
    ind1 = np.repeat(ty_mat, x_mat.shape[0], axis=0)
    ind2 = np.tile(x_mat, (ty_mat.shape[0], 1))
    color_diff = np.sum((im[ind1[:, 0], ind1[:, 1]] - im[ind2[:, 0], ind2[:, 1]])**2, axis=1)
    prob = color_diff.reshape((ty_mat.shape[0], x_mat.shape[0])).astype(np.float32)
    return prob


class EMRegistration(object):
    def __init__(self, X, Y, X_color, Y_color, X_full, Y_full, image,
                 zncc=None, sigma2=None, max_iterations=None, tolerance=None, w=None, *args, **kwargs):
        if type(X) is not np.ndarray or X.ndim != 2:
            raise ValueError(
                "The target point cloud (X) must be at a 2D numpy array.")

        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError(
                "The source point cloud (Y) must be a 2D numpy array.")

        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                "Both point clouds need to have the same number of dimensions.")

        if sigma2 is not None and (not isinstance(sigma2, numbers.Number) or sigma2 <= 0):
            raise ValueError(
                "Expected a positive value for sigma2 instead got: {}".format(sigma2))

        if max_iterations is not None and (not isinstance(max_iterations, numbers.Number) or max_iterations < 0):
            raise ValueError(
                "Expected a positive integer for max_iterations instead got: {}".format(max_iterations))
        elif isinstance(max_iterations, numbers.Number) and not isinstance(max_iterations, int):
            warn("Received a non-integer value for max_iterations: {}. Casting to integer.".format(max_iterations))
            max_iterations = int(max_iterations)

        if tolerance is not None and (not isinstance(tolerance, numbers.Number) or tolerance < 0):
            raise ValueError(
                "Expected a positive float for tolerance instead got: {}".format(tolerance))

        if w is not None and (not isinstance(w, numbers.Number) or w < 0 or w >= 1):
            raise ValueError(
                "Expected a value between 0 (inclusive) and 1 (exclusive) for w instead got: {}".format(w))

        self.image = image
        self.X = X
        self.Y = Y
        self.X_full = X_full
        self.Y_full = Y_full
        # self.X_color = X_color
        # self.Y_color = Y_color
        self.TY = Y
        self.TY_full = Y_full
        self.sigma2 = initialize_sigma2(X, Y) if sigma2 is None else sigma2
        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape
        self.tolerance = 0.001 if tolerance is None else tolerance
        self.w = 0.0 if w is None else w
        self.max_iterations = 100 if max_iterations is None else max_iterations
        self.iteration = 0
        self.diff = np.inf
        self.q = np.inf
        self.P = np.zeros((self.M, self.N))
        self.Pt1 = np.zeros((self.N, ))
        self.P1 = np.zeros((self.M, ))
        self.Np = 0
        self.error = 10000
        self.zncc = zncc

    def register(self, callback=lambda **kwargs: None):
        self.transform_point_cloud()
        # while self.iteration < self.max_iterations:
        while self.iteration < 5:
            self.iterate()
            if callable(callback):
                kwargs = {'iteration': self.iteration,
                          'error': self.error, 'X': self.X_full, 'Y': self.TY_full,
                          "X_color": self.X_color, "Y_color": self.Y_color}
                callback(**kwargs)

        return self.TY, self.get_registration_parameters()

    def get_registration_parameters(self):
        raise NotImplementedError(
            "Registration parameters should be defined in child classes.")

    def update_transform(self):
        raise NotImplementedError(
            "Updating transform parameters should be defined in child classes.")

    def transform_point_cloud(self):
        raise NotImplementedError(
            "Updating the source point cloud should be defined in child classes.")

    def update_variance(self):
        raise NotImplementedError(
            "Updating the Gaussian variance for the mixture model should be defined in child classes.")

    def iterate(self):
        self.expectation()
        self.maximization()
        self.iteration += 1

    def reload_inputs(self, X, Y, X_color, Y_color):
        self.X = X
        self.Y = Y
        self.X_color = X_color
        self.Y_color = Y_color
        (self.M, _) = self.Y.shape
        self.transform_point_cloud()
        self.iteration = 0
        self.diff = np.inf
        self.q = np.inf
        self.P = np.zeros((self.M, self.N))
        self.Pt1 = np.zeros((self.N, ))
        self.P1 = np.zeros((self.M, ))
        self.Np = 0

    def expectation(self):
        if self.iteration > 0:
            # P = color_based_posterior_prob(self.image, self.X, self.TY)
            P = np.sum((self.X[None, :, :] - self.TY[:, None, :]) ** 2, axis=2)
            c = (2 * np.pi * self.sigma2) ** (self.D / 2)
            c = c * self.w / (1 - self.w)
            c = c * self.M / self.N

            P = np.exp(-P / (2 * self.sigma2))
            den = np.sum(P, axis=0)
            den = np.tile(den, (self.M, 1))
            den[den == 0] = np.finfo(float).eps
            den += c
            self.P = np.divide(P, den)
        else:
            self.P = self.zncc

        # print(self.P[0])
        # print(np.sum(self.zncc))
        # print(np.max(self.P[0]))
        # sys.exit()

        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P1)

    def maximization(self):
        self.update_transform()
        self.transform_point_cloud()
        self.update_variance()


class AffineRegistration(EMRegistration):
    """
    Affine registration.

    Attributes
    ----------
    B: numpy array (semi-positive definite)
        DxD affine transformation matrix.

    t: numpy array
        1xD initial translation vector.

    YPY: float
        Denominator value used to update the scale factor.
        Defined in Fig. 2 and Eq. 8 of https://arxiv.org/pdf/0905.2635.pdf.

    X_hat: numpy array
        Centered target point cloud.
        Defined in Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf

    """

    def __init__(self, B=None, t=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if B is not None and (B.ndim is not 2 or B.shape[0] is not self.D or B.shape[1] is not self.D or not is_positive_semi_definite(B)):
            raise ValueError(
                'The rotation matrix can only be initialized to {}x{} positive semi definite matrices. Instead got: {}.'.format(self.D, self.D, B))

        if t is not None and (t.ndim is not 2 or t.shape[0] is not 1 or t.shape[1] is not self.D):
            raise ValueError(
                'The translation vector can only be initialized to 1x{} positive semi definite matrices. Instead got: {}.'.format(self.D, t))
        self.B = np.eye(self.D) if B is None else B
        self.t = np.atleast_2d(np.zeros((1, self.D))) if t is None else t

    def update_transform(self):
        """
        Calculate a new estimate of the rigid transformation.
        """

        # source and target point cloud means
        muX = np.divide(np.sum(np.dot(self.P, self.X), axis=0), self.Np)
        muY = np.divide(
            np.sum(np.dot(np.transpose(self.P), self.Y), axis=0), self.Np)

        self.X_hat = self.X - np.tile(muX, (self.N, 1))
        Y_hat = self.Y - np.tile(muY, (self.M, 1))

        self.A = np.dot(np.transpose(self.X_hat), np.transpose(self.P))
        self.A = np.dot(self.A, Y_hat)

        self.YPY = np.dot(np.transpose(Y_hat), np.diag(self.P1))
        self.YPY = np.dot(self.YPY, Y_hat)

        # Calculate the new estimate of affine parameters using update rules for (B, t)
        # as defined in Fig. 3 of https://arxiv.org/pdf/0905.2635.pdf.
        self.B = np.linalg.solve(np.transpose(self.YPY), np.transpose(self.A))
        self.t = np.transpose(muX) - np.dot(np.transpose(self.B), np.transpose(muY))

    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the affine transformation.
        """
        if Y is None:
            self.TY = np.dot(self.Y, self.B) + np.tile(self.t, (self.M, 1))
            self.TY_full = np.dot(self.Y_full, self.B) + np.tile(self.t, (self.Y_full.shape[0], 1))
            return
        else:
            return np.dot(Y, self.B) + np.tile(self.t, (Y.shape[0], 1))

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the affine transformation.
        See the update rule for sigma2 in Fig. 3 of of https://arxiv.org/pdf/0905.2635.pdf.
        """
        qprev = self.q

        trAB = np.trace(np.dot(self.A, self.B))
        xPx = np.dot(np.transpose(self.Pt1), np.sum(np.multiply(self.X_hat, self.X_hat), axis=1))
        trBYPYP = np.trace(np.dot(np.dot(self.B, self.YPY), self.B))
        self.q = (xPx - 2 * trAB + trBYPYP) / (2 * self.sigma2) + self.D * self.Np/2 * np.log(self.sigma2)
        self.diff = np.abs(self.q - qprev)

        dist = np.sum((self.X[None, :, :] - self.TY[:, None, :]) ** 2, axis=2)
        self.error = np.mean(dist[:, np.argmax(self.P, axis=1)])

        self.sigma2 = (xPx - trAB) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

    def get_registration_parameters(self):
        """
        Return the current estimate of the affine transformation parameters.
        """
        return self.B, self.t
