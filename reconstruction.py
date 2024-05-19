import cv2
import numpy as np
import scipy
import scipy.linalg

from epipolar_geometry import E2RsCs, FKs2E
from ransac import ransac_f


def compute_epipolar_geometry(pts0, pts1, K1, K2):
    u1 = np.vstack([pts0.T, np.ones(pts0.shape[0])])
    u2 = np.vstack([pts1.T, np.ones(pts1.shape[0])])

    # Compute the best Fundamental matrix using ransac
    F, selected_points = ransac_f(pts_matches=np.array([u1, u2]), conf=0.99, LO_RANSAC=True)

    # Compute the Essential matrix
    E = FKs2E(F, K1=K1, K2=K2)

    # Compute the Rotation and Translation
    R1, R2s, C1, C2s = E2RsCs(E)

    return F, E, R1, R2s, C1, C2s


def triangulate_points(pts0, pts1, P1, P2):
    """
    Triangulate the points in 3D space using the projection matrices
    :param pts0: Points in the first image (Nx2 array)
    :param pts1: Points in the second image (Nx2 array)
    :param P1: Projection matrix of the first camera (3x4 matrix)
    :param P2: Projection matrix of the second camera (3x4 matrix)
    :return: 3D points (Nx3 array)
    """
    pts_3d = []

    for x1, x2 in zip(pts0, pts1):
        x1_h = np.array([x1[0], x1[1], 1.0])
        x2_h = np.array([x2[0], x2[1], 1.0])

        # Formulate the system of equations A @ X = 0
        A = np.zeros((4, 4))
        A[0] = x1_h[0] * P1[2] - P1[0]
        A[1] = x1_h[1] * P1[2] - P1[1]
        A[2] = x2_h[0] * P2[2] - P2[0]
        A[3] = x2_h[1] * P2[2] - P2[1]

        # Solve the system using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]

        # Normalize the homogeneous coordinates
        X = X / X[3]
        pts_3d.append(X[:3])

    return np.array(pts_3d)


def reconstruct_scene(pts0, pts1, K1, K2, image1, image2):
    """
    Reconstruct the scene from the Fundamental matrix and the cameras calibration matrices
    :param pts0: points in image 1
    :param pts1: points in image 2
    :param K1: camera 1 calibration matrix
    :param K2: camera 2 calibration matrix
    :param image1: image 1
    :param image2: image 2
    :return: 3D points
    """

    # Compute the epipolar geometry
    F, E, R1, R2s, C1, C2s = compute_epipolar_geometry(pts0, pts1, K1, K2)

    # Compute the projection matrices
    P1 = K1 @ np.hstack([R1, C1.reshape(-1, 1)])
    P2 = K2 @ np.hstack([R2s[0], C2s[0].reshape(-1, 1)])  # Select the first rotation and translation

    # Triangulate the points
    pts_3d = triangulate_points(pts0, pts1, P1, P2)

    return pts_3d
