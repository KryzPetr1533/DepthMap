import numpy as np


def depth_to_pcl(depth: np.array, intrinsics: np.array):
    """
    :arg depth: matrix with shape H x W
        with depth per pixel, 0 for sky and transparent
    :arg intrinsics: intrinsic matrix 3 x 3
    :return: pointcloud in 3D camera frame with shape N x 3
        where N is number of depths not equal 0 pixels
    """
    i_dim, j_dim = depth.shape
    ij = np.mgrid[1: i_dim + 1: 1, 1: j_dim + 1: 1].reshape(2, -1).T.reshape(i_dim, j_dim, 2)
    z = np.ones((i_dim, j_dim, 1))
    ijz_matr = np.concatenate((ij, z), axis=-1)
    ijz_matr[:,:,1] *= -1
    points = (ijz_matr * np.expand_dims(depth, -1)).reshape(-1, 3)
    points_in_cam_frame = np.transpose((np.linalg.inv(intrinsics) @ points.T), (1, 0))
    pointcloud = points_in_cam_frame[
        np.where(np.logical_not(np.all(points_in_cam_frame == 0.0, axis=1)))
    ]
    return pointcloud
