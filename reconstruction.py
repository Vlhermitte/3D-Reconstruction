import cv2
import numpy as np

from epipolar_geometry import E2RsCs, FKs2E
from visualization import plot_epipolar_line
from ransac import ransac_f

def reconstruct_scene(pts0, pts1, K1, K2, image1, image2):
    u1 = np.vstack([pts0.T, np.ones(pts0.shape[0])])
    u2 = np.vstack([pts1.T, np.ones(pts1.shape[0])])

    # Compute the best Fundamental matrix using ransac
    F, selected_points = ransac_f(pts_matches=np.array([u1, u2]), LO_RANSAC=True)
    print(f"Fundamental matrix:\n{F}")
    print(f"Selected points: {selected_points}")

    # Compute the best Essential matrix
    E = FKs2E(F, K1=K1, K2=K2)
    print(f"Essential matrix:\n{E}")

    # Compute the best Rotation and Translation
    R1, R2s, C1, C2s = E2RsCs(E)
    print(f"R1:\n{R1}")
    print(f"R2:\n{R2s}")
    print(f"C1:\n{C1}")
    print(f"C2:\n{C2s}")

    # Plot the epipolar lines
    plot_epipolar_line(image1, image2, u1, u2, np.arange(u1.shape[1]), F, title='Epipolar lines (Custom)')

    # Compute the projection matrices
    P1 = K1 @ np.hstack([R1, C1.reshape(-1, 1)])
    P2 = K2 @ np.hstack([R2s[0], C2s[0].reshape(-1, 1)]) # Select the first rotation and translation

    # Triangulate the points
    points = cv2.triangulatePoints(P1, P2, pts0.T, pts1.T)
    points /= points[3]
    points = points[:3].T

    return points
