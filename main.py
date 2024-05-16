# Epipolar geometry, camera calibration, and 3D reconstruction

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

import reconstruction_opencv
import reconstruction
from object_loading import load_obj
from visualization import plot_image, plot_matches
from features_detection import match_images
from epipolar_geometry import get_best_E, EKs2F, E2RsCs




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

    #plot_image(images[0], 'Image 0')
    #plot_image(images[1], 'Image 1')

    # Apply SIFT to find keypoints and descriptors
    sift = cv2.SIFT_create()
    pts0, pts1 = match_images(images, sift)

    # 1. Reconstruction using OpenCV functions
    points_3d_cv = reconstruction_opencv.reconstruct_scene(pts0, pts1, Ks[0], Ks[1], images[0], images[1])

    # 2. Reconstruction using custom functions
    points_3d = reconstruction.reconstruct_scene(pts0, pts1, Ks[0], Ks[1], images[0], images[1])

    # 3. Plot the 3D points
    # 3.1 Plot the 3D points using OpenCV functions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d_cv[:, 0], points_3d_cv[:, 1], points_3d_cv[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D points (OpenCV)')
    plt.show()
    plt.close()

    # 3.2 Plot the 3D points using custom functions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D points (Custom)')
    plt.show()
    plt.close()