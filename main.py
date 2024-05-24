# Epipolar geometry, camera calibration, and 3D reconstruction

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import scipy.io as sio

import reconstruction_opencv as rcv
import reconstruction as rc
from features_detection import match_images
from visualization import plot_matches, plot_image, plot_epipolar_lines, plot_3D_points
from epipolar_geometry import compute_epipolar_errors



if __name__ == '__main__':
    # Load Cameras data
    df = pd.read_csv(os.path.join('data', 'cameras_parameters.csv'), sep=';')
    images_names = df['Screenshot'].values
    # Load images as RGB
    images = [cv2.cvtColor(cv2.imread(os.path.join('data', image_name)), cv2.COLOR_BGR2RGB) for image_name in images_names]
    image1 = images[0]
    image2 = images[1]

    # Compute the cameras calibration matrices (K) from the FOV
    FOV = df['FOV (deg)'].values
    Ks = []
    for i, fov in enumerate(FOV):
        width = images[0].shape[1]
        height = images[0].shape[0]
        K = np.zeros((3, 3))
        K[0, 0] = width / (2 * np.tan(np.deg2rad(fov / 2)))
        K[1, 1] = height / (2 * np.tan(np.deg2rad(fov / 2)))
        K[0, 2] = width / 2
        K[1, 2] = height / 2
        K[2, 2] = 1
        Ks.append(K)

    # Undistort the images
    for i in range(len(images)):
        k1, k2, k3, p1, p2 = df['k1'].values[i], df['k2'].values[i], df['k3'].values[i], df['p1'].values[i], df['p2'].values[i]
        images[i] = cv2.undistort(images[i], Ks[i], np.array([k1, k2, k3, p1, p2]))

    # plot_image(images[0], 'Image 0')
    # plot_image(images[1], 'Image 1')

    # Apply a feature detector and matcher
    feature2D = cv2.SIFT_create()
    pts0, pts1 = match_images(images, feature2D)
    print("Matches found: ", pts0.shape[0])

    # Plot the matches
    plot_matches(images[0], images[1], pts0, pts1)

    # Plot the epipolar lines
    u1 = np.vstack([pts0.T, np.ones(pts0.shape[0])])
    u2 = np.vstack([pts1.T, np.ones(pts1.shape[0])])

    # Compute the epipolar geometry (using OpenCV)
    F_, E_, R_, C_, selected_points_ = rcv.compute_epipolar_geometry(pts0, pts1, Ks[0], Ks[1])
    print("Fundamental matrix (OpenCV): ", F_)
    print(f"OpenCV Selected points: {selected_points_}")

    # Compute the epipolar geometry
    F, E, R1, R2, C1, C2, selected_points = rc.compute_epipolar_geometry(pts0, pts1, Ks[0], Ks[1])
    print("Fundamental matrix: ", F)
    print(f"Custom Selected points: {selected_points}")

    print(f"OpenCV Fundamental matrix Error: {sum(compute_epipolar_errors(F_, u1, u2)).mean()}")
    print(f"Custom Fundamental matrix Error: {sum(compute_epipolar_errors(F, u1, u2)).mean()}")

    plot_epipolar_lines(images[0], images[1], u1, u2, selected_points_, F_, title='Epipolar lines (OpenCV)')
    plot_epipolar_lines(images[0], images[1], u1, u2, selected_points, F, title='Epipolar lines (Custom)')

    # Triangulate the points
    pts3d_opencv = rcv.reconstruct_scene(pts0, pts1, Ks[0], Ks[1])
    pts3d = rc.reconstruct_scene(pts0, pts1, Ks[0], Ks[1])

    # Plot the 3D points
    plot_3D_points(pts3d_opencv, title='3D points (OpenCV)')
    plot_3D_points(pts3d, title='3D points (Custom)')

