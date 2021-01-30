import numpy as np
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

    # compute affine homography
    aff_hom = np.row_stack((np.eye(3,4), pinf.T))

    return aff_hom


def estimate_euc_hom(cams, vps):
    # make points homogeneous
    
    ...

    return euc_hom
