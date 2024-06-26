import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_vertices(vertices, title=None) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)
    plt.show()

def plot_faces(vertices, faces) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for face in faces:
        ax.plot(vertices[face, 0], vertices[face, 1], vertices[face, 2], c='b')
    plt.show()
    plt.close()

def plot_image(image, title=None) -> None:
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.show()
    plt.close()

def plot_matches(image0, image1, pts0, pts1) -> None:
    fig, ax = plt.subplots()
    ax.imshow(np.hstack((image0, image1)))
    ax.plot(pts0[:, 0], pts0[:, 1], 'o')
    ax.plot(pts1[:, 0] + image0.shape[1], pts1[:, 1], 'o')
    for i in range(len(pts0)):
        ax.plot([pts0[i, 0], pts1[i, 0] + image0.shape[1]], [pts0[i, 1], pts1[i, 1]], '-')
    # Add title
    ax.set_title('Matches')
    plt.show()
    plt.close()

def plot_epipolar_lines(image1, image2, u1, u2, ix, F, title=None) -> None:
    """
    Plot the epipolar lines and points

    :param image1: image 1
    :param image2: image 2
    :param u1: homogeneous coordinates of image 1
    :param u2: homogeneous coordinates of image 2
    :param ix: indices of the points
    :param F: Fundamental matrix
    """
    assert u1.shape[0] == 3, f"u1 shape is {u1.shape}, expected (3, N)"
    assert u2.shape[0] == 3, f"u2 shape is {u2.shape}, expected (3, N)"
    assert F.shape == (3, 3), f"F shape is {F.shape}, expected (3, 3)"

    # compute epipolar lines
    l2 = F @ u1[:, ix]
    l1 = F.T @ u2[:, ix]

    # Plot the epipolar lines and points
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image1)
    ax[1].imshow(image2)

    for i in range(len(ix)):
        # plot points
        ax[0].scatter(u1[0, ix[i]], u1[1, ix[i]])
        ax[1].scatter(u2[0, ix[i]], u2[1, ix[i]])

        # plot lines
        x = np.linspace(0, image1.shape[1], 100)
        y = (-l1[2, i] - l1[0, i] * x) / l1[1, i]
        ax[0].plot(x, y)

        x = np.linspace(0, image2.shape[1], 100)
        y = (-l2[2, i] - l2[0, i] * x) / l2[1, i]
        ax[1].plot(x, y)


    for ax_ in ax:
        ax_.set_title('Image 1') if ax_ == ax[0] else ax_.set_title('Image 2')
        ax_.set_xlim(0, image1.shape[1])
        ax_.set_ylim(image1.shape[0], 0)
        ax_.axis('off')

    plt.tight_layout()
    if title:
        fig.suptitle(title)
        # Save the figure
        plt.savefig('Results/' + title + '.png')
    plt.show()
    plt.close()

def plot_3D_points(X, edges=None, azimuth=-60, elevation=30, title=None) -> None:
    """
    Plot the 3D points
    param X: 3D points
    param edges: edges to plot
    param azimuth: azimuth angle
    param elevation: elevation angle
    param title: plot title
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect(aspect=(1.5, 1.2, 1))

    ax.scatter(X[0], X[1], X[2], c='r', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if edges is not None:
        for x, y, z in zip(X[0, edges].T, X[1, edges].T, X[2, edges].T):
            ax.plot(x, y, z, 'y-')

    ax.view_init(elev=elevation, azim=azimuth)

    if title is not None:
        plt.title(title)
        plt.savefig('Results/' + title + '.png')

    plt.show()
    plt.close()