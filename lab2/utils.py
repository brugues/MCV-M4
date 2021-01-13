import numpy as np
import math
import sys
from math import ceil
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import random
import plotly.graph_objects as go

def plot_img(img, do_not_use=[0]):
    plt.figure(do_not_use[0])
    do_not_use[0] += 1
    plt.imshow(img)


def get_transformed_pixels_coords(I, H, shift=None):
    ys, xs = np.indices(I.shape[:2]).astype("float64")
    if shift is not None:
        ys += shift[1]
        xs += shift[0]
    ones = np.ones(I.shape[:2])
    coords = np.stack((xs, ys, ones), axis=2)
    coords_H = (H @ coords.reshape(-1, 3).T).T
    coords_H /= coords_H[:, 2, np.newaxis]
    cart_H = coords_H[:, :2]
    
    return cart_H.reshape((*I.shape[:2], 2))

def apply_H_fixed_image_size(I, H, corners):
    h, w = I.shape[:2] # when we convert to np.array it swaps
    
    # corners
    c1 = np.array([1, 1, 1])
    c2 = np.array([w, 1, 1])
    c3 = np.array([1, h, 1])
    c4 = np.array([w, h, 1])
    
    # transformed corners
    Hc1 = H @ c1
    Hc2 = H @ c2
    Hc3 = H @ c3
    Hc4 = H @ c4
    Hc1 = Hc1 / Hc1[2]
    Hc2 = Hc2 / Hc2[2]
    Hc3 = Hc3 / Hc3[2]
    Hc4 = Hc4 / Hc4[2]
    
    xmin = corners[0]
    xmax = corners[1]
    ymin = corners[2]
    ymax = corners[3]

    size_x = ceil(xmax - xmin + 1)
    size_y = ceil(ymax - ymin + 1)
    
    # transform image
    H_inv = np.linalg.inv(H)
    
    out = np.zeros((size_y, size_x, 3))
    shift = (xmin, ymin)
    interpolation_coords = get_transformed_pixels_coords(out, H_inv, shift=shift)
    interpolation_coords[:, :, [0, 1]] = interpolation_coords[:, :, [1, 0]]
    interpolation_coords = np.swapaxes(np.swapaxes(interpolation_coords, 0, 2), 1, 2)
    
    out[:, :, 0] = map_coordinates(I[:, :, 0], interpolation_coords)
    out[:, :, 1] = map_coordinates(I[:, :, 1], interpolation_coords)
    out[:, :, 2] = map_coordinates(I[:, :, 2], interpolation_coords)
    
    return out.astype("uint8")

def Normalise_last_coord(x):
    xn = x  / x[2,:]
    
    return xn

def normalize_points(points):
    
    # Get centroid of given points
    points_centroid = np.empty(shape=(2,))
    points_centroid[0] = np.mean(points[0,:])
    points_centroid[1] = np.mean(points[1,:])

    # get distance to origin
    points_origin = np.empty(shape=(2,points.shape[1]))
    points_origin[0,:] = points[0,:] - points_centroid[0]
    points_origin[1,:] = points[1,:] - points_centroid[1]

    distances = np.sqrt(points_origin[0,:]**2 + points_origin[1,:]**2)
    mean_d = np.mean(distances)

    s = np.sqrt(2)/mean_d
        
    t = [-s*points_centroid[0], -s*points_centroid[1]]
    T = np.array([[s, 0, t[0]], 
                  [0, s, t[1]],
                  [0, 0, 1]])
    
    points_norm = T @ points

    # points_centroid_norm = np.empty(shape=(2,))
    # points_centroid_norm[0] = np.mean(points_norm[0,:])
    # points_centroid_norm[1] = np.mean(points_norm[1,:])
    # print(f'centroid: {points_centroid_norm}')
    # distances_norm = np.sqrt(points_norm[0,:]**2 + points_norm[1,:]**2)
    # mean_d_norm = np.mean(distances_norm)
    # print(f'ideal value: {np.sqrt(2)}, dist: {mean_d_norm}')

    return points_norm, T

def compute_A(points1, points2):
    A = np.empty(shape=(2*points1.shape[1], 9))

    points1 = Normalise_last_coord(points1)
    points2 = Normalise_last_coord(points2)
    
    a1 = -points2[2,:] * points1
    a2 = points2[1,:] * points1
    a3 = -a1
    a4 = -points2[0,:] * points1

    for pidx in range(points1.shape[1]):
        A[2*pidx,:] = [0,0,0,*a1[:,pidx],*a2[:,pidx]]
        A[2*pidx+1,:] = [*a3[:,pidx],0,0,0,*a4[:,pidx]]
    return A

def DLT_homography(points1, points2):
    #print("125: I'm in DLT_H")
    '''
    1. normalization of x
    2. normalization of x'
    3. Appply DLT: 
        3.1 For each xi<->xi' compute Ai
        3.2 Assemble the n matrices Ai to form the 2n x 9 matrix A
        3.3 Compute SVD of A (h might be last column of V)
        3.4 H = [[h1, h2, h3]...]

    '''
    
    # Normalize (translation and scaling)
    points1_norm, T1 = normalize_points(points1)
    points2_norm, T2 = normalize_points(points2)

    A = compute_A(points1_norm, points2_norm)

    U,D,V = np.linalg.svd(A)
    h = np.transpose(V)[:,-1]
    H = h.reshape([3,3])
    H_norm = np.linalg.inv(T2) @ H @ T1

    return H_norm

def Inliers(H, points1, points2, th):
    # Check that H is invertible
    if abs(math.log(np.linalg.cond(H))) > 15:
        idx = np.empty(1)
        return idx
        
    inliers = []

    points1 = np.transpose(points1)
    points2 = np.transpose(points2)

    # for pidx,p1 in enumerate(points1):
    #     p2 = points2[pidx]
    #     p2h = H @ p1

    #     d = np.linalg.norm(np.cross(p2,p2h))**2
    #     #print("169: distance_calculated")
    #     if (d < th**2):
    #         inliers.append(pidx)

    for pidx,p1 in enumerate(points1):
        p2 = points2[pidx]
        p2h = H @ p1
        p1h = np.linalg.inv(H) @ p2

        p1 = [p1[0]/p1[2], p1[1]/p1[2]]
        p1h = [p1h[0]/p1h[2], p1h[1]/p1h[2]]
        p2 = [p2[0]/p2[2], p2[1]/p2[2]]
        p2h = [p2h[0]/p2h[2], p2h[1]/p2h[2]]

        d = (p1[0]-p1h[0])**2+(p1[1]-p1h[1])**2 + (p2[0]-p2h[0])**2+(p2[1]-p2h[1])**2

        if (d < th**2):
            inliers.append(pidx)
    
    return np.array(inliers)

def Ransac_DLT_homography(points1, points2, th, max_it):
    #print("176: I'm in Ransac_DLT_H")
    Ncoords, Npts = points1.shape

    it = 0
    best_inliers = np.empty(1)
    
    while it < max_it:
        indices = random.sample(range(1, Npts), 4)
        H = DLT_homography(points1[:,indices], points2[:,indices])
        inliers = Inliers(H, points1, points2, th)
        # test if it is the best model so far
        if inliers.shape[0] > best_inliers.shape[0]:
            best_inliers = inliers
        
        # update estimate of iterations (the number of trials) to ensure we pick, with probability p,
        # an initial data set with no outliers
        fracinliers = inliers.shape[0]/Npts
        pNoOutliers = 1 -  fracinliers**4
        eps = sys.float_info.epsilon
        pNoOutliers = max(eps, pNoOutliers)   # avoid division by -Inf
        pNoOutliers = min(1-eps, pNoOutliers) # avoid division by 0
        p = 0.99
        # print(f'inliers_shape: {inliers.shape[0]}')
        # print(f'fracinliers: {fracinliers}')
        # print(f'outliers: {1-fracinliers**4}')
        #print("n_outliers: {}, p: {}".format(pNoOutliers, p))
        #print("numerator: {}, denominator: {}".format(math.log(1-p), math.log(pNoOutliers)))
        max_it = math.log(1-p)/math.log(pNoOutliers)
        #print("201: maxit {}".format(max_it))
        it += 1
        
    
    # compute H from all the inliers
    H = DLT_homography(points1[:,best_inliers], points2[:,best_inliers])
    inliers = best_inliers
    
    return H, inliers

def optical_center(P):
    U, d, Vt = np.linalg.svd(P)
    o = Vt[-1, :3] / Vt[-1, -1]
    return o

def view_direction(P, x):
    # Vector pointing to the viewing direction of a pixel
    # We solve x = P v with v(3) = 0
    v = np.linalg.inv(P[:,:3]) @ np.array([x[0], x[1], 1])
    return v

def plot_camera(P, w, h, fig, legend):
    
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

def plot_image_origin(w, h, fig, legend):
    p1 = np.array([0, 0, 0])
    p2 = np.array([w, 0, 0])
    p3 = np.array([w, h, 0])
    p4 = np.array([0, h, 0])
    
    x = np.array([p1[0], p2[0], p3[0], p4[0], p1[0]])
    y = np.array([p1[1], p2[1], p3[1], p4[1], p1[1]])
    z = np.array([p1[2], p2[2], p3[2], p4[2], p1[2]])
    
    fig.add_trace(go.Scatter3d(x=x, y=z, z=-y, mode='lines',name=legend))
    
    return
