# Epipolar geometry, camera calibration, and 3D reconstruction

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

import reconstruction_opencv
import reconstruction
from features_detection import match_images
from visualization import plot_matches, plot_image, plot_vertices



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

    plot_image(images[0], 'Image 0')
    plot_image(images[1], 'Image 1')

    # Apply SIFT to find keypoints and descriptors
    sift = cv2.SIFT_create()
    pts0, pts1 = match_images(images, sift)
    print("Matches found: ", pts0.shape[0])

    # Plot the matches
    #plot_matches(images[0], images[1], pts0, pts1)

    # 1. Reconstruction using OpenCV functions
    points_3d_cv = reconstruction_opencv.reconstruct_scene(pts0, pts1, Ks[0], Ks[1], images[0], images[1])

    # 2. Reconstruction using custom functions
    points_3d = reconstruction.reconstruct_scene(pts0, pts1, Ks[0], Ks[1], images[0], images[1])

    # 3. Plot the 3D points TODO: Implement this part
    # 3.1 Plot the 3D points using OpenCV functions
    #plot_vertices(points_3d_cv, title='3D points (OpenCV reconstruction)')

    # 3.2 Plot the 3D points using custom functions
    #plot_vertices(points_3d, title='3D points (Custom reconstruction)')