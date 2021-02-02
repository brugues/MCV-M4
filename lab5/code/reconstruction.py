import cv2
import numpy as np
import scipy

import utils as h
import maths as mth

from scipy import optimize as opt
import random
from scipy.optimize import least_squares

def compute_proj_camera(F, i):
    # Result 9.15 of MVG (v = 0, lambda = 1). It assumes P1 = [I|0]
    # P' = [S @ F | e']
    # A good choice is: S = [e']x .
    # The epipole may be computed from e'.T @ F = 0

    # compute the epipole
    e = mth.nullspace(F.T, atol=1e-13, rtol=0)

    # obtain S
    S = mth.hat_operator(e)

    # compute P
    P = np.column_stack((S@F, e))
    
    return P

def estimate_3d_points_2(P1, P2, xr1, xr2):
    """
    Linear triangulation (Hartley ch 12.2 pg 312) to find the 3D point X
    where p1 = m1 * X and p2 = m2 * X. Solve AX = 0.
    :param p1, p2: 2D points in homo. or catesian coordinates. Shape (3 x n)
    :param m1, m2: Camera matrices associated with p1 and p2. Shape (3 x 4)
    :returns: 4 x n homogenous 3d triangulated points
    """
    num_points = xr1.shape[1]
    res = np.ones((4, num_points))

    for i in range(num_points):
        A = np.asarray([
            (xr1[0, i] * P1[2, :] - P1[0, :]),
            (xr1[1, i] * P1[2, :] - P1[1, :]),
            (xr2[0, i] * P2[2, :] - P2[0, :]),
            (xr2[1, i] * P2[2, :] - P2[1, :])
        ])

        _, _, V = np.linalg.svd(A)

        X = V[-1, :4]
        res[:, i] = X / X[3]

    return res

def compute_reproj_error(X, P1, P2, xr1, xr2):
    # project 3D points using P

    if xr1.shape[0] == 3:
        xr1 = euclid(xr1.T).T

    if xr2.shape[0] == 3:
        xr2 = euclid(xr2.T).T

    x1est = P1 @ X
    x2est = P2 @ X

    x1est = euclid(x1est.T).T
    x2est = euclid(x2est.T).T

    error = np.sum(np.sum(xr1 - x1est) ** 2 + np.sum(xr2 - x2est) ** 2)

    return error

def transform(hom, X, cams_matrix):
    # Algorithm 19.2 of MVG: step (i) (page 479)

    Xhom = np.linalg.inv(hom)@X
    Xhom = Xhom/Xhom[-1,:]

    cams_hom = []
    for cam_i in cams_matrix:
        cams_hom.append(cam_i@hom)

    return Xhom, cams_hom


# Function added by Team 7
def normalize_points(points, dim='2d'):
    if dim == '2d':
        points = points / points[2]

        # Centroid of given points
        points_centroid = np.empty(shape=(2,))
        points_centroid[0] = np.mean(points[0, :])
        points_centroid[1] = np.mean(points[1, :])

        # Mean distance to origin
        points_origin = np.empty(shape=(2, points.shape[1]))
        points_origin[0, :] = points[0, :] - points_centroid[0]
        points_origin[1, :] = points[1, :] - points_centroid[1]
        distances = np.sqrt(points_origin[0, :] ** 2 + points_origin[1, :] ** 2)
        mean_d = np.mean(distances)

        # Scaling factor
        s = np.sqrt(2) / mean_d

        # Translation factor
        t = [-s * points_centroid[0], -s * points_centroid[1]]
        T = np.array([[s, 0, t[0]],
                      [0, s, t[1]],
                      [0, 0, 1]])

        points_norm = T @ points
    elif dim == '3d':
        points = points / points[3]

        # Centroid of given points
        points_centroid = np.empty(shape=(3,))
        points_centroid[0] = np.mean(points[0, :])
        points_centroid[1] = np.mean(points[1, :])
        points_centroid[2] = np.mean(points[2, :])

        # Mean distance to origin
        points_origin = np.empty(shape=(3, points.shape[1]))
        points_origin[0, :] = points[0, :] - points_centroid[0]
        points_origin[1, :] = points[1, :] - points_centroid[1]
        points_origin[2, :] = points[1, :] - points_centroid[2]
        distances = np.sqrt(points_origin[0, :] ** 2 + points_origin[1, :] ** 2)
        mean_d = np.mean(distances)

        # Scaling factor
        s = np.sqrt(3) / mean_d

        # Translation factor
        t = [-s * points_centroid[0], -s * points_centroid[1], -s * points_centroid[2]]
        T = np.array([[s, 0, 0, t[0]],
                      [0, s, 0, t[1]],
                      [0, 0, s, t[2]],
                      [0, 0, 0, 1]])

        points_norm = T @ points
    else:
        raise NameError

    return points_norm, T

# Function added by Team 7
def geometric_error_terms(variables, data_points):
    points_2d, points_3d = data_points
    # camera projection matrix
    P = variables.reshape(3, 4)

    # project 3d points to the image space
    points_2d_h = P @ points_3d
    points_2d_h = points_2d_h[:2, :] / points_2d_h[2, :]

    # return the vector of residuals as the geometric error (without squaring the terms, as
    # the function 'least squares' constructs the cost function as a sum of squares of the residuals)
    return (points_2d - points_2d_h).flatten()


def resection(tracks, i):
    # extract 3D-2D correspondences from tracks
    points_2d = []
    points_3d = []

    for track in tracks:
        # Some points have the third coordinate as 0. Some points don't appear in the 3rd camera
        if i in track.views.keys() and track.pt[3] != 0:
            points_2d.append((track.views[i]))
            points_3d.append(track.pt.T)

    points_2d = homog(np.array(points_2d)).T
    points_3d = np.array(points_3d).T

    points_2d, T2d = normalize_points(points_2d, dim='2d')
    points_3d, T3d = normalize_points(points_3d, dim='3d')

    # Apply DLT algorithm to get an initial estimation of P (P0).
    # We need 6 pairs of points, so that we have 12 equations
    indices = random.sample(range(1, points_2d.shape[1]), 6)
    A = np.zeros(shape=(len(indices)*2, 12))
    for k, idx in enumerate(indices):
        X = points_3d[:, idx]
        x = points_2d[:, idx]

        A[2*k, :] = np.concatenate((np.zeros(shape=4), -x[2] * X, x[1] * X))
        A[2*k+1, :] = np.concatenate((x[2] * X, np.zeros(shape=4), -x[0] * X))

    U, D, VT = np.linalg.svd(A)
    P0 = VT[-1, :].reshape(3, 4)

    # use least squares to find P that minimizes the geometric error
    result = least_squares(geometric_error_terms, P0.flatten(), method='lm', args=([[points_2d[:2,:], points_3d]]))
    P_min = result.x[:].reshape(3,4)

    P = T2d.T @ P_min @ T3d  # Denormalization
    P = P / P[2, 3]

    print(P)

    return P

def homog(x):
    return np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)

def euclid(x):
    return x[:, :-1] / x[:, [-1]]

def compute_eucl_cam(F, x1, x2):

    K = np.array([[2362.12, 0, 1520.69], [0, 2366.12, 1006.81], [0, 0, 1]])
    E = K.T @ F @ K

    # camera projection matrix for the first camera
    P1 = K @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    # make sure E is rank 2
    U, S, V = np.linalg.svd(E)
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V
    E = np.dot(U, np.dot(np.diag([1, 1, 0]), V))

    # create matrices (Hartley p 258)
    Z = mth.skew([0, 0, -1])
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # return all four solutions
    P2 = [np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]

    ind = 0
    maxres = 0

    for i in range(4):
        # triangulate inliers and compute depth for each camera
        homog_3D = cv2.triangulatePoints(P1, P2[i], x1[:2], x2[:2])
        # the sign of the depth is the 3rd value of the image point after projecting back to the image
        d1 = np.dot(P1, homog_3D)[2]
        d2 = np.dot(P2[i], homog_3D)[2]

        if sum(d1 > 0) + sum(d2 < 0) > maxres:
            maxres = sum(d1 > 0) + sum(d2 < 0)
            ind = i
            infront = (d1 > 0) & (d2 < 0)

    list_cams = []
    list_cams.append(P1)
    list_cams.append(P2[ind])

    return list_cams


# Function added by Team 7
def K_R_t_from_camera_matrix(P, method='qr'):
    """
        Returns the parameters of a Camera Matrix

    :param matrix: 3x4 Camera matrix
    :param method: qr or skew
    :return: K, R and t camera parameters
    """

    # QR Decomposition
    if method == 'qr':
        # QR factorization
        R, K = np.linalg.qr(P[:, :3])

        # Check that diagonal elements of K are positive
        T = np.diag(np.sign(np.diag(K)))
        K = K @ T
        R = T @ R

        t = np.linalg.inv(K) @ P[:, 3]

        # Make sure that determinant of R is 1
        if np.linalg.det(R) < 0:
            R = -R
            t = -t

        # normalize K. From lecture 3 PDF, last element of K is 1.
        K = K / K[2, 2]

        return K, R, t

    # Assume cameras with 0 skew
    elif method == 'skew':
        Paux = P[:, :3]
        A = Paux @ Paux.T
        A = A / A[2, 2]

        K = np.array([[np.sqrt((A[0, 0] - A[0, 2]) ** 2), 0, 0],
                      [0, np.sqrt((A[1, 1] - A[1, 2]) ** 2), 0],
                      [A[0, 2], A[1, 2], 1]])

        Rt = np.linalg.inv(K) @ P

        R = Rt[:, :3]
        t = Rt[:, 3]

        return K, R, t
    else:
        raise (NameError)
