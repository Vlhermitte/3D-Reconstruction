import cv2
import numpy as np

from epipolar_geometry import get_best_E, EKs2F, E2RsCs
from visualization import plot_epipolar_line

def reconstruct_scene(pts0, pts1, K1, K2, image1, image2):
    u1 = np.vstack([pts0.T, np.ones(pts0.shape[0])])
    u2 = np.vstack([pts1.T, np.ones(pts1.shape[0])])

    idx = np.arange(u1.shape[1])    # TODO : Too many indices, select fewer indices (Using RANSAC for example)
    E, Fe, best_idx = get_best_E(idx, u1, u2, K1, K2)

    F = EKs2F(E, K1, K2)

    # Plot epipolar lines
    plot_epipolar_line(image1, image2, u1, u2, best_idx, F)

    # Compute the projection matrices
    R1, R2s, C1, C2s = E2RsCs(E)

    # Compute the 3D points
    points, _ = cv2.reconstructPoints(R1, R2s, C1, C2s, pts0, pts1)

    return points