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

def estimate_euc_hom(cams, vps):

    def _eqn(vp1, vp2):
        r = np.array([vp1[0] * vp2[0],
                      vp1[0] * vp2[1] + vp1[1] * vp2[0],
                      vp1[0] * vp2[2] + vp1[2] * vp2[0],
                      vp1[1] * vp2[1],
                      vp1[1] * vp2[2] + vp1[2] * vp2[1],
                      vp1[2] * vp2[2]])
        return r

    # make points homogeneous
    vps_h = fd.make_homogeneous(vps)

    # combine constraints to create a system of equations A_sys
    vp1, vp2, vp3 = vps_h

    eqn1 = _eqn(vp1, vp2)
    eqn2 = _eqn(vp1, vp3)
    eqn3 = _eqn(vp2, vp3)

    A_sys = np.array([[*eqn1],
                      [*eqn2],
                      [*eqn3],
                      [0, 1, 0, 0, 0, 0],
                      [1, 0, 0, -1, 0, 0]])

    # compute w_v as the null vector of A_sys
    w_v = mth.nullspace(A_sys)

    w_v = np.squeeze(w_v)

    # create matrix w from vector w_v
    w = np.array([[w_v[0], w_v[1], w_v[2]],
                  [w_v[1], w_v[3], w_v[4]],
                  [w_v[2], w_v[4], w_v[5]]])

    # compute A
    M = cams[:, :3]
    A = scipy.linalg.cholesky(LA.inv(M.T @ w @ M), lower=False)

    # create the corresponding homography using inv(A)
    invA = LA.inv(A)
    euc_hom = np.array([[invA[0][0], invA[0][1], invA[0][2], 0],
                        [invA[1][0], invA[1][1], invA[1][2], 0],
                        [invA[2][0], invA[2][1], invA[2][2], 0],
                        [0, 0, 0, 1]])

    return euc_hom
