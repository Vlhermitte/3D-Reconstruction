import cv2
import numpy as np

from visualization import plot_epipolar_lines


def compute_epipolar_geometry(pts0, pts1, K1, K2):

    # Compute the best Fundamental matrix using ransac
    F, mask = cv2.findFundamentalMat(pts0, pts1, cv2.RANSAC)
    pts0 = pts0[mask.ravel() == 1]
    pts1 = pts1[mask.ravel() == 1]

    # Compute the Essential matrix
    E, _ = cv2.findEssentialMat(pts0, pts1, K1, method=cv2.RANSAC)

    # Compute the Rotation and Translation
    _, R, C, _ = cv2.recoverPose(E, pts0, pts1, K1)

    return F, E, R, C

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

    F, mask = cv2.findFundamentalMat(pts0, pts1, cv2.RANSAC)
    # pts0 = pts0[mask.ravel() == 1]
    # pts1 = pts1[mask.ravel() == 1]

    E, _ = cv2.findEssentialMat(pts0, pts1, K1, method=cv2.RANSAC)

    # Plot epipolar lines
    u1 = np.vstack([pts0.T, np.ones(pts0.shape[0])])
    u2 = np.vstack([pts1.T, np.ones(pts1.shape[0])])

    # Compute the projection matrices
    _, R, C, _ = cv2.recoverPose(E, pts0, pts1, K1)
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, C])

    # Triangulate the points
    points = cv2.triangulatePoints(P1, P2, pts0.T, pts1.T)
    points /= points[3]
    points = points[:3].T

    return points
