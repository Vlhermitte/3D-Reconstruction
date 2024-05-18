import cv2
import numpy as np

from epipolar_geometry import E2RsCs, FKs2E
from visualization import plot_epipolar_lines
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
    :param pts0:
    :param pts1:
    :param P1:
    :param P2:
    :return:
    """
    # TODO : later implement own triangulation method
    pts_3d = cv2.triangulatePoints(P1, P2, pts0[:2], pts1[:2])
    pts_3d = pts_3d / pts_3d[3]
    return pts_3d

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
    P2 = K2 @ np.hstack([R2s[0], C2s[0].reshape(-1, 1)]) # Select the first rotation and translation

    # Triangulate the points
    pts_3d = triangulate_points(pts0, pts1, P1, P2)

    return pts_3d