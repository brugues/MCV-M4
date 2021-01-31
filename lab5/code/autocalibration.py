import numpy as np
from numpy import linalg as LA
import scipy
import reconstruction as rc
import maths as mth
import fundamental as fd


def estimate_aff_hom(cams, vps):
    """
    MVG: 10.4 Stratified reconstruction, Parallel lines (pages 269-270)
    MVG: 10.4.1 The step to affine reconstruction (page 268)
    """
    # compute 3D Vanishing points
    vp3d = rc.estimate_3d_points_2(cams[0], cams[1], vps[0].T, vps[1].T)

    # compute plane at infinity
    pinf = mth.nullspace(vp3d.T)
    pinf = pinf/pinf[-1,:]

    # create affine homography using the plane at infinity
    aff_hom = np.row_stack((np.eye(3,4), pinf.T))

    return aff_hom

def row(vp1, vp2):
    r = np.array([vp1[0]*vp2[0], 
                  vp1[0]*vp2[1]+vp1[1]*vp2[0], 
                  vp1[0]*vp2[2]+vp1[2]*vp2[0], 
                  vp1[1]*vp2[1], 
                  vp1[1]*vp2[2]+vp1[2]*vp2[1], 
                  vp1[2]*vp2[2]])
    return r

# TODO Perform Metric rectification. First compute the transforming
# homography from vanishing points and the camera constrains skew = 0,
# squared pixels. Then perform the transformation to Euclidean space
# (reuse your code)

def estimate_euc_hom(cams, vps):
    # make points homogeneous
    hpts = fd.make_homogeneous(vps)
    
    #Compute A from the linear system
    vp1, vp2, vp3 = hpts
    
    row1 = row(vp1, vp2)
    row2 = row(vp1, vp3)
    row3 = row(vp2, vp3)
    
    A_sys = np.array([[row1[0], row1[1], row1[2], row1[3], row1[4], row1[5]],
                      [row2[0], row2[1], row2[2], row2[3], row2[4], row2[5]],
                      [row3[0], row3[1], row3[2], row3[3], row3[4], row3[5]],
                      [0, 1, 0, 0, 0, 0],
                      [1, 0, 0, -1, 0, 0]])
    
    #Compute w
    w_v = mth.nullspace(A_sys)
    
    w_v = np.squeeze(w_v)
    
    w = np.array([[w_v[0], w_v[1], w_v[2]],
                  [w_v[1], w_v[3], w_v[4]],
                  [w_v[2], w_v[4], w_v[5]]])
    
    #print(w)
    
    #Computing A for the Euclidean Homography
    
    M = cams[:, :3]
    
    A = scipy.linalg.cholesky(LA.inv(M.T@w@M), lower=False)
    
    invA = LA.inv(A)
    
    #print(invA)
    
    euc_hom = np.array([[invA[0][0], invA[0][1], invA[0][2], 0],
                       [invA[1][0], invA[1][1], invA[1][2], 0],
                       [invA[2][0], invA[2][1], invA[2][2], 0],
                       [0, 0, 0, 1]])
    
    #print(euc_hom)
    
    return euc_hom
    