import numpy as np
from numpy import linalg as LA
import cv2
import math
import sys
import random
import matplotlib
from matplotlib import pyplot as plt
from operator import itemgetter
import plotly.graph_objects as go


def normalize_points(points):

    points = points/points[2,:]

    # Centroid of given points
    points_centroid = np.empty(shape=(2,))
    points_centroid[0] = np.mean(points[0,:])
    points_centroid[1] = np.mean(points[1,:])

    # Mean distance to origin
    points_origin = np.empty(shape=(2,points.shape[1]))
    points_origin[0,:] = points[0,:] - points_centroid[0]
    points_origin[1,:] = points[1,:] - points_centroid[1]
    distances = np.sqrt(points_origin[0,:]**2 + points_origin[1,:]**2)
    mean_d = np.mean(distances)

    # Scaling factor
    s = np.sqrt(2)/mean_d
    
    # Translation factor
    t = [-s*points_centroid[0], -s*points_centroid[1]]
    T = np.array([[s, 0, t[0]], 
                  [0, s, t[1]],
                  [0, 0, 1]])
    
    points_norm = T @ points

    return points_norm, T

def fundamental_matrix(points1, points2, check_norm=False):
    '''  
    Steps:
    0. Normalize points
    1. Create matrix W from pi and p’i correspondences
    2. Compute the SVD of matrix W=UDVT
    3. Create vector f from last column of V
    4. Compose fundamental matrix F_rank3
    5. Compute the SVD of fundamental matrix F_rank3 = UDVT
    6. Remove last singular value of D to create Ď
    7. Re-compute matrix F = U Ď VT (rank 2)
    8. Denormalize
    '''

    # Step 0. Normalize points (translation and scaling)
    points1, T1 = normalize_points(points1)
    points2, T2 = normalize_points(points2)

    # To check if the points are correctly normalized
    if check_norm:
        check_normalization(points1, '1')
        check_normalization(points2, '2')

    # Step 1. Create matrix W from pi and p’i correspondences. 
    # W has shape 8x9 if we use 8 correspondences
    points1 = points1.T
    points2 = points2.T
    W = np.ones(shape=(points1.shape[0], 9))

    W[:,0] = points1[:,0] * points2[:,0]
    W[:,1] = points1[:,1] * points2[:,0]
    W[:,2] = points2[:,0]
    W[:,3] = points1[:,0] * points2[:,1]
    W[:,4] = points1[:,1] * points2[:,1]
    W[:,5] = points2[:,1]
    W[:,6] = points1[:,0]
    W[:,7] = points1[:,1]

    # Step 2. Compute the SVD of matrix W=UDVT
    _, _, VT = LA.svd(W)

    # Step 3. Create vector f from the last column of V (last row of VT)
    # Step 4. Compose fundamental matrix F_rank3
    Fr3 = VT[-1,:].reshape(3,3)

    # Step 5. Compute the SVD of fundamental matrix F_rank3 = UDVT
    U, D, VT = LA.svd(Fr3)

    # Step 6. Remove last singular value of D
    D[-1] = 0
    D = np.diag(D)

    # Step 7. Recompute F (rank 2)
    Fr2 = U @ D @ VT

    # Step 8. Denormalize
    Fr2 = T2.T @ Fr2 @ T1

    return Fr2


def Normalise_last_coord(x):    
    return x / x[2,:]

# Function from last week tweaked a little bit
def Inliers(F, points1, points2, th):

    inliers = []

    points1 = Normalise_last_coord(points1)
    points2 = Normalise_last_coord(points2)

    points1 = np.transpose(points1)
    points2 = np.transpose(points2)
    
    for pidx,p1 in enumerate(points1):
        p2 = points2[pidx]

        # We check if a point is an inlier using the Sampson distance
        Fp1 = F @ p1
        Ftp2 = F.T @ p2

        Fp1 = Fp1/Fp1[2]
        Ftp2 = Ftp2/Ftp2[2]

        num = (p2.T @ F @ p1)**2    
        denom = Fp1[0]**2 \
               + Fp1[1]**2 \
               + Ftp2[0]**2 \
               + Ftp2[1]**2      

        d = num/denom

        if (d < th**2):
            inliers.append(pidx)
    
    return np.array(inliers)


def Ransac_fundamental_matrix(points1, points2, th, max_it_0):
    best_inliers = np.empty(1)
    it = 0

    while it < max_it_0:
        indices = random.sample(range(1, points1.shape[1]), 8)

        F = fundamental_matrix(points1[:,indices], points2[:,indices])
        inliers = Inliers(F, points1, points2, th)
        
        if inliers.shape[0] > best_inliers.shape[0]:
            best_inliers = inliers

        fracinliers = inliers.shape[0]/points1.shape[1]
        pNoOutliers = 1 - fracinliers**8
        eps = sys.float_info.epsilon
        pNoOutliers = max(eps, pNoOutliers)   # avoid division by -Inf
        pNoOutliers = min(1-eps, pNoOutliers) # avoid division by 0
        p = 0.99
        max_it = math.log(1-p)/math.log(pNoOutliers)
        it += 1
        
        # To avoid an infinite loop
        max_it_0 = min(max_it,max_it_0)

        #print(f'Total # iterations: {it}')
        #print(f'Total # inliers: {best_inliers.shape[0]}')
        F = fundamental_matrix(points1[:, best_inliers], points2[:, best_inliers])
        inliers = best_inliers

    return F, inliers


def plot_points(points, texture,fig): 
    
    fig.add_trace(go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2],mode='markers',name='test',marker=dict(color=texture,size=2)))

    return

def plot_camera_points(P, w, h, fig, legend):
    
    o = optical_center(P)
    scale = 200
    p1 = o + view_direction(P, [0, 0]) * scale
    p2 = o + view_direction(P, [w, 0]) * scale
    p3 = o + view_direction(P, [w, h]) * scale
    p4 = o + view_direction(P, [0, h]) * scale
    
    x = np.array([p1[0], p2[0], o[0], p3[0], p2[0], p3[0], p4[0], p1[0], o[0], p4[0], o[0], (p1[0]+p2[0])/2])
    y = np.array([p1[1], p2[1], o[1], p3[1], p2[1], p3[1], p4[1], p1[1], o[1], p4[1], o[1], (p1[1]+p2[1])/2])
    z = np.array([p1[2], p2[2], o[2], p3[2], p2[2], p3[2], p4[2], p1[2], o[2], p4[2], o[2], (p1[2]+p2[2])/2])
    
    fig.add_trace(go.Scatter3d(x=x, y=z, z=-y, mode='lines',name=legend))
    
    return