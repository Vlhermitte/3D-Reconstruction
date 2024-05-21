import cv2
import numpy as np
import scipy
import scipy.linalg

from epipolar_geometry import E2RsCs, FKs2E, triangulate_points
from ransac import ransac_f


def compute_epipolar_geometry(pts0, pts1, K1, K2):
    """
    Compute the epipolar geometry from the points in the images and the cameras calibration matrices
    :param pts0: points in image 1
    :param pts1: points in image 2
    :param K1: camera 1 calibration matrix
    :param K2: camera 2 calibration matrix
    :return: Fundamental matrix, Essential matrix, Rotation matrices, Camera positions
    """
    u1 = np.vstack([pts0.T, np.ones(pts0.shape[0])])
    u2 = np.vstack([pts1.T, np.ones(pts1.shape[0])])

    # Compute the best Fundamental matrix using ransac
    F, selected_points = ransac_f(pts_matches=np.array([u1, u2]), th=10.0, conf=0.99)

    # Compute the Essential matrix
    E = FKs2E(F, K1=K1, K2=K2)

    # Compute the Rotation and Translation
    R1, R2, C1, C2 = E2RsCs(E, u1, u2)

    return F, E, R1, R2, C1, C2


def reconstruct_scene(pts0, pts1, K1, K2):
    """
    Reconstruct the scene from the Fundamental matrix and the cameras calibration matrices
    :param pts0: points in image 1
    :param pts1: points in image 2
    :param K1: camera 1 calibration matrix
    :param K2: camera 2 calibration matrix
    :return: 3D points
    """

    # Compute the epipolar geometry
    F, E, R1, R2, C1, C2 = compute_epipolar_geometry(pts0, pts1, K1, K2)

    # Compute the projection matrices
    P1 = K1 @ np.hstack([R1, C1.reshape(-1, 1)])
    P2 = K2 @ np.hstack([R2, C2.reshape(-1, 1)])  # Select the first rotation and translation

    # Triangulate the points
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)
    pts0 = np.vstack([pts0.T, np.ones(pts0.shape[0])])
    pts1 = np.vstack([pts1.T, np.ones(pts1.shape[0])])
    pts0 = K1_inv @ pts0
    pts1 = K2_inv @ pts1
    pts_3d = triangulate_points(P1, P2, pts0, pts1)

    return pts_3d
