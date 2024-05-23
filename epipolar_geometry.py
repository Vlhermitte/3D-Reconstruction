import numpy as np
import scipy
import scipy.linalg
import itertools

def u2F(u1, u2):
    """
    computes the fundamental matrix using the seven-point algorithm from 7 euclidean correspondences u1, u2
    :param u1: 3x7 array of homogeneous coordinates
    :param u2: 3x7 array of homogeneous coordinates
    :return:
    """

    assert u1.shape == (3, 7), f"u1 shape is {u1.shape}, expected (3, 7)"
    assert u2.shape == (3, 7), f"u2 shape is {u2.shape}, expected (3, 7)"

    A = np.zeros((7, 9))
    x1, y1 = u1[:2]
    x2, y2 = u2[:2]

    A = np.c_[x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones(7)]

    # If A is singular, return an empty list (we cannot solve a system of equations with a singular matrix)
    if np.linalg.matrix_rank(A) < 7:
        return []

    # Solve for the nullspace of A
    G1, G2 = scipy.linalg.null_space(A).T.reshape(2, 3, 3)  # (12.33)

    # Compute the polynomial coefficients
    a3, a2, a1, a0 = u2F_polynom(G1, G2)

    # Solve the polynomial
    roots = np.roots([a3, a2, a1, a0])  # Solving the polynomial (12.36)

    FF = [] # list of fundamental matrices
    for alpha in roots:
        if not np.iscomplex(alpha):
            alpha = np.real(alpha)
            G = G1 + alpha * G2

            # check if rank(G) = 2 and G is different from G2
            if np.linalg.matrix_rank(G) == 2 and not np.allclose(G, G2):
                FF.append(G)

    return FF

def FKs2E(F, K1, K2):
    """
    Compute the Essential matrix from the Fundamental matrix and the camera calibration matrix

    The Essential matrix can be computer as follows:
    .. math::
        E1 = K2^T @ F @ K1

        U, S, Vt = svd(E1)

        S = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]

        E = U @ S @ V^T

    :param F: Fundamental matrix
    :param K: Camera calibration matrix
    :return: Essential matrix
    """
    assert F.shape == (3, 3), f"F shape is {F.shape}, expected (3, 3)"
    assert K1.shape == (3, 3), f"K1 shape is {K1.shape}, expected (3, 3)"
    assert K2.shape == (3, 3), f"K2 shape is {K2.shape}, expected (3, 3)"

    E1 = K2.T @ F @ K1
    U, S, Vt = np.linalg.svd(E1)
    S = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    E = U @ S @ Vt
    return E

def EKs2F(E, K1, K2):
    """
    Compute the Fundamental matrix from the Essential matrix and the camera calibration matrix

    The Fundamental matrix can be computer as follows:
    .. math::
        F = K^-T @ E @ K^-1

    :param E: Essential matrix
    :param K: Camera calibration matrix
    :return: Fundamental matrix
    """
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F

def E2RsCs(E, u1, u2):
    """
    Compute the rotation matrices and camera positions from the Essential matrix
    :param E:
    :return R1, R2s, C1, C2s:
    """

    U, S, Vt = np.linalg.svd(E)

    # Camera 1 has the identity rotation and zero translation
    R1 = np.eye(3)
    C1 = np.zeros(3)

    # Camera 2 has two possible rotations and translations
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R2_1 = U @ W @ Vt
    R2_2 = U @ W.T @ Vt
    C2_1 = U[:, 2]
    C2_2 = -U[:, 2]

    in_front_both_best = -np.inf
    R2_best = None
    C2_best = None
    P1 = np.hstack([R1, -R1 @ C1.reshape(-1, 1)])
    for R2, C2 in itertools.product([R2_1, R2_2], [C2_1, C2_2]):
        P2 = np.hstack([R2, -R2 @ C2.reshape(-1, 1)])
        X = triangulate_points(P1, P2, u1, u2)[0:3, :]
        in_front_C1 = np.einsum('ij, ij -> j', u1, X) > 0
        in_front_C2 = np.einsum('ij, ij -> j', u2, X) > 0
        in_front_both = (in_front_C1 & in_front_C2).sum()

        if in_front_both > in_front_both_best:
            in_front_both_best = in_front_both
            R2_best = R2
            C2_best = C2

    return R1, R2_best, C1, C2_best

def compute_epipolar_errors(F, u1, u2):
    """
    computes the epipolar error for each pair of points
    :param F: 3x3 fundamental matrix
    :param u1: 3xN array of homogeneous coordinates
    :param u2: 3xN array of homogeneous coordinates
    :return: 1xN array of epipolar errors
    """
    assert F.shape == (3, 3), f"F shape is {F.shape}, expected (3, 3)"
    assert u1.shape[0] == 3, f"u1 shape is {u1.shape}, expected (3, N)"
    assert u2.shape[0] == 3, f"u2 shape is {u2.shape}, expected (3, N)"

    l2 = F @ u1
    l1 = F.T @ u2

    errors1 = np.abs(np.sum(u1 * l1, axis=0)) / np.linalg.norm(l1[:2], axis=0)
    errors2 = np.abs(np.sum(u2 * l2, axis=0)) / np.linalg.norm(l2[:2], axis=0)

    return errors1, errors2

def triangulate_points(P1, P2, pts0, pts1):
    """
    Triangulate the points in 3D space using the projection matrices
    :param pts0: Points in the first image (Nx2 array)
    :param pts1: Points in the second image (Nx2 array)
    :param P1: Projection matrix of the first camera (3x4 matrix)
    :param P2: Projection matrix of the second camera (3x4 matrix)
    :return: 3D points (Nx3 array)
    """
    N = pts0.shape[1]
    X = np.zeros((4, N))

    p11, p12, p13 = P1
    p21, p22, p23 = P2
    for i in range(N):
        A = np.vstack([
            pts0[0, i] * p13 - p11,
            pts0[1, i] * p13 - p12,
            pts1[0, i] * p23 - p21,
            pts1[1, i] * p23 - p22
        ])
        _, _, Vt = np.linalg.svd(A)
        X[:, i] = Vt[-1]

    # Normalize the 3D points
    X = X / X[-1]
    return X[:3]


def u2F_polynom(G1, G2):
    a3 = np.linalg.det(G2)

    a2 = (G2[1, 0] * G2[2, 1] * G1[0, 2]
          - G2[1, 0] * G2[0, 1] * G1[2, 2]
          + G2[0, 0] * G2[1, 1] * G1[2, 2]
          + G2[2, 0] * G1[0, 1] * G2[1, 2]
          + G2[2, 0] * G2[0, 1] * G1[1, 2]
          - G2[0, 0] * G1[2, 1] * G2[1, 2]
          - G2[2, 0] * G1[1, 1] * G2[0, 2]
          - G2[2, 0] * G2[1, 1] * G1[0, 2]
          - G2[0, 0] * G2[2, 1] * G1[1, 2]
          + G1[1, 0] * G2[2, 1] * G2[0, 2]
          + G2[1, 0] * G1[2, 1] * G2[0, 2]
          + G1[2, 0] * G2[0, 1] * G2[1, 2]
          - G1[1, 0] * G2[0, 1] * G2[2, 2]
          - G1[0, 0] * G2[2, 1] * G2[1, 2]
          - G2[1, 0] * G1[0, 1] * G2[2, 2]
          + G2[0, 0] * G1[1, 1] * G2[2, 2]
          + G1[0, 0] * G2[1, 1] * G2[2, 2]
          - G1[2, 0] * G2[1, 1] * G2[0, 2])

    a1 = (G1[0, 0] * G1[1, 1] * G2[2, 2]
          + G1[0, 0] * G2[1, 1] * G1[2, 2]
          + G2[2, 0] * G1[0, 1] * G1[1, 2]
          - G1[1, 0] * G1[0, 1] * G2[2, 2]
          - G2[0, 0] * G1[2, 1] * G1[1, 2]
          - G2[1, 0] * G1[0, 1] * G1[2, 2]
          - G2[2, 0] * G1[1, 1] * G1[0, 2]
          + G2[0, 0] * G1[1, 1] * G1[2, 2]
          + G1[1, 0] * G1[2, 1] * G2[0, 2]
          + G1[1, 0] * G2[2, 1] * G1[0, 2]
          + G1[2, 0] * G2[0, 1] * G1[1, 2]
          - G1[1, 0] * G2[0, 1] * G1[2, 2]
          - G1[2, 0] * G2[1, 1] * G1[0, 2]
          + G2[1, 0] * G1[2, 1] * G1[0, 2]
          - G1[0, 0] * G2[2, 1] * G1[1, 2]
          - G1[2, 0] * G1[1, 1] * G2[0, 2]
          + G1[2, 0] * G1[0, 1] * G2[1, 2]
          - G1[0, 0] * G1[2, 1] * G2[1, 2])

    a0 = np.linalg.det(G1)

    return a3, a2, a1, a0