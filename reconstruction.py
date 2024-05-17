import cv2
import numpy as np

from epipolar_geometry import get_best_E, EKs2F, E2RsCs
from visualization import plot_epipolar_line

def reconstruct_scene(pts0, pts1, K1, K2, image1, image2):
    u1 = np.vstack([pts0.T, np.ones(pts0.shape[0])])
    u2 = np.vstack([pts1.T, np.ones(pts1.shape[0])])

    idx = np.arange(u1.shape[1])    # TODO : Too many indices, select fewer indices
    E, Fe, best_idx = get_best_E(idx, u1, u2, K1, K2)

    F = EKs2F(E, K1, K2)

    # Plot epipolar lines
    plot_epipolar_line(image1, image2, u1, u2, idx, F, title='Epipolar lines (Custom)')

    # Compute the projection matrices
    R1, R2s, C1, C2s = E2RsCs(E)

    # Triangulate points
    points_3d = []
    for i in range(len(R2s)):
        R1 = np.eye(3)
        C1 = np.zeros(3)
        P1 = K1 @ np.hstack([R1, C1.reshape(-1, 1)])
        P2 = K2 @ np.hstack([R2s[i], C2s[i].reshape(-1, 1)])
        points_4d = cv2.triangulatePoints(P1, P2, u1[:2, best_idx], u2[:2, best_idx])
        points_4d /= points_4d[3, :]
        points_3d.append(points_4d[:3, :])

    # Keep the points with positive depth
    points_3d = np.array(points_3d[0]) # TODO : Fix this
    return points_3d