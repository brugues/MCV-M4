"""

MIT License (MIT)

Copyright (c) FALL 2016, Jahdiel Alvarez

Author: Jahdiel Alvarez

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Based on Scipy's cookbook:
http://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

"""

import numpy as np
import cv2
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

import utils as h
import reconstruction as rc

class PySBA:
    """Python class for Simple Bundle Adjustment"""

    def __init__(self, cameraArray, points3D, points2D, cameraIndices, point2DIndices):
        """Intializes all the class attributes and instance variables.
            Write the specifications for each variable:

            cameraArray with shape (n_cameras, 9) contains initial estimates of parameters for all cameras.
                    First 3 components in each row form a rotation vector,
                    next 3 components form a translation vector,
                    then a focal distance and two distortion parameters.

            points_3d with shape (n_points, 3)
                    contains initial estimates of point coordinates in the world frame.

            camera_ind with shape (n_observations,)
                    contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.

            point_ind with shape (n_observations,)
                    contains indices of points (from 0 to n_points - 1) involved in each observation.

            points_2d with shape (n_observations, 2)
                    contains measured 2-D coordinates of points projected on images in each observations.
        """
        self.cameraArray = cameraArray
        self.points3D = points3D
        self.points2D = points2D

        self.cameraIndices = cameraIndices
        self.point2DIndices = point2DIndices

    def rotate(self, points, rot_vecs):
        """Rotate points by given rotation vectors.

        Rodrigues' rotation formula is used.
        """
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


    def project(self, points, cameraArray):
        """Convert 3-D points to 2-D by projecting onto images."""
        points_proj = self.rotate(points, cameraArray[:, :3])
        points_proj += cameraArray[:, 3:6]
        points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        f = cameraArray[:, 6]
        k1 = cameraArray[:, 7]
        k2 = cameraArray[:, 8]
        n = np.sum(points_proj ** 2, axis=1)
        r = 1 + k1 * n + k2 * n ** 2
        points_proj *= (r * f)[:, np.newaxis]
        return points_proj


    def fun(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d):
        """Compute residuals.

        `params` contains camera parameters and 3-D coordinates.
        """
        camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
        points_3d = params[n_cameras * 9:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        return (points_proj - points_2d).ravel()

    def bundle_adjustment_sparsity(self, numCameras, numPoints, cameraIndices, pointIndices):
        m = cameraIndices.size * 2
        n = numCameras * 9 + numPoints * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(cameraIndices.size)
        for s in range(9):
            A[2 * i, cameraIndices * 9 + s] = 1
            A[2 * i + 1, cameraIndices * 9 + s] = 1

        for s in range(3):
            A[2 * i, numCameras * 9 + pointIndices * 3 + s] = 1
            A[2 * i + 1, numCameras * 9 + pointIndices * 3 + s] = 1

        return A


    def optimizedParams(self, params, n_cameras, n_points):
        """
        Retrieve camera parameters and 3-D coordinates.
        """
        camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
        points_3d = params[n_cameras * 9:].reshape((n_points, 3))

        return camera_params, points_3d


    def bundleAdjust(self):
        """ Returns the bundle adjusted parameters, in this case the optimized
         rotation and translation vectors. """
        numCameras = self.cameraArray.shape[0]
        numPoints = self.points3D.shape[0]

        x0 = np.hstack((self.cameraArray.ravel(), self.points3D.ravel()))
        f0 = self.fun(x0, numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D)

        A = self.bundle_adjustment_sparsity(numCameras, numPoints, self.cameraIndices, self.point2DIndices)

        res = least_squares(self.fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                            args=(numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D))

        #params = self.optimizedParams(res.x, numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D)
        params, points_3d = self.optimizedParams(res.x, numCameras, numPoints)

        return params, points_3d


def adapt_format_pysba(tracks, cams):
    tracks = np.array(tracks)
    cams = np.array(cams)

    # We know the shape from the init method of the PySBA class
    camera_params = np.zeros(shape=(cams.shape[0], 9))

    # Iterate over the number of cameras we have (First dimension of cams gives us the number of cameras)
    for i in range(cams.shape[0]):
        K, R, t = rc.K_R_t_from_camera_matrix(cams[i], method='qr')
        # According to the desciption of PySBA class, we need a rotation vector and not a rotation matrix, so we use
        # Rodrigues for the transormation
        r, _ = cv2.Rodrigues(R)  # Returns a tuple, we just need the first value. The second element is the jacobian

        # We now get the focal distance from K[0,0] and K[1,1]
        f = (K[0, 0] + K[1, 1]) / 2

        # And finally we set the 2 distortion parameters to 0, as we assume that there is no distortion
        k1 = k2 = 0

        # Add elements to the camera_params array
        camera_params[i, 0:3] = r.reshape(3)
        camera_params[i, 3:6] = t.reshape(3)
        camera_params[i, 6:] = np.array([f, k1, k2])

    # Iterate over the tracks objects. Each element is of the class Track, which has the following elements:
    # Self.views
    # Self.ref_views -> Refined self.views according to fundamental matrix F
    # Self.pt -> Vector of 4 elements --> Point coordinates
    # We use all this info to build the following 4 lists
    points_3d = np.zeros(shape=(len(tracks), 3))
    points_2d = []
    points_2d_indices = []
    camera_indices = []

    for track_idx, track in enumerate(tracks):
        points_3d[track_idx] = track.pt[:-1] / track.pt[-1]

        # We get the keys of the views dictionary, from which we can get the cameras of each view and its 2d points
        # Keys: 0 and 1 (cam index)
        # Values: a 2D point (tuple)
        for cam_index in track.views.keys():
            points_2d.append(track.views[cam_index])
            points_2d_indices.append(track_idx)
            camera_indices.append(cam_index)

    points_2d = np.array(points_2d)
    points_2d_indices = np.array(points_2d_indices)
    camera_indices = np.array(camera_indices)

    # For debug purposes
    if h.debug:
        print("camera_params", camera_params.shape)
        print("points_3d", points_3d.shape)
        print("points_2d", points_2d.shape)
        print("points_2d_indices", points_2d_indices.shape)
        print("camera_indices", camera_indices.shape)

    return camera_params, points_3d, points_2d, camera_indices, points_2d_indices
