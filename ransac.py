import numpy as np
from scipy.optimize import least_squares
from epipolar_geometry import u2F, compute_epipolar_errors


def sample(pts1, pts2, n):
    """
    Randomly sample n points from pts1 and pts2
    :param pts1: 3xN array of points
    :param pts2: 3xN array of points
    :param n: number of points to sample
    :return: pts1_sample, pts2_sample
    """
    assert pts1.shape[0] == 3, f"Expected pts1 to have shape (3, N), got {pts1.shape}"
    assert pts2.shape[0] == 3, f"Expected pts2 to have shape (3, N), got {pts2.shape}"
    assert n <= pts1.shape[1], f"Expected n <= pts1.shape[1], got {n} and {pts1.shape[1]}"
    assert n <= pts2.shape[1], f"Expected n <= pts2.shape[1], got {n} and {pts2.shape[1]}"
    idx = np.random.choice(pts1.shape[1], n, replace=False)
    pts1_sample = pts1[:, idx]
    pts2_sample = pts2[:, idx]

    return pts1_sample, pts2_sample


def nsamples(n_inliers: int, num_tc: int, sample_size: int, conf: float):
    """
    Function, which calculates number of samples needed to achieve desired confidence
    :param n_inliers: number of inliers
    :param num_tc: total number of correspondences
    :param sample_size: size of the sample
    :param conf: confidence
    :return: number of samples needed
    """
    assert 0 <= n_inliers <= num_tc, f"Expected 0 <= n_inliers <= num_tc, got {n_inliers} n_inliers and {num_tc} num_tc."
    assert 0 <= conf <= 1, f"Expected 0 <= conf <= 1, got {conf}"
    assert sample_size <= num_tc, f"Expected sample_size <= num_tc, got {sample_size} and {num_tc}"

    if n_inliers == num_tc:
        return 1.0
    inl_ratio = n_inliers / num_tc
    return np.ceil(np.log(1 - conf) / np.log(1 - np.power(inl_ratio, sample_size)))


def slope(p1, p2):
    return (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[0] != p1[0] else None


def fdist(F, pts1, pts2):
    """
    Function, which calculates the Sampson distance
    :param F: 3x3 fundamental matrix
    :param pts1: 3xN array of points
    :param pts2: 3xN array of points
    :return: distance
    """
    assert F.shape == (3, 3), f"Expected homography to have shape (3, 3), got {F.shape}"
    assert pts1.shape[0] == 3, f"Expected pts1 to have shape (3, N), got {pts1.shape}"
    assert pts2.shape[0] == 3, f"Expected pts2 to have shape (3, N), got {pts2.shape}"

    # Compute the epipolar lines
    l1 = F @ pts1
    l2 = F.T @ pts2

    # Compute the distance
    pts2_F_pts1 = np.einsum('ij,ji->i', pts2.T, F @ pts1)
    d1 = pts2_F_pts1 ** 2 / (l1[0] ** 2 + l1[1] ** 2)
    d2 = pts2_F_pts1 ** 2 / (l2[0] ** 2 + l2[1] ** 2)

    return d1 + d2


def optimize_fundamental_matrix(F, pts1, pts2):
    """
    Function, which optimizes the fundamental matrix using Levenberg-Marquardt optimization
    :param F: 3x3 fundamental matrix
    :param pts1: 3xN array of points
    :param pts2: 3xN array of points
    :return: optimized fundamental matrix
    """
    assert F.shape == (3, 3), f"Expected F to have shape (3, 3), got {F.shape}"
    assert pts1.shape[0] == 3, f"Expected pts1 to have shape (3, N), got {pts1.shape}"
    assert pts2.shape[0] == 3, f"Expected pts2 to have shape (3, N), got {pts2.shape}"

    def cost(F):
        return fdist(F.reshape(3, 3), pts1, pts2).sum()

    def jacobian(F):
        F = F.reshape(3, 3)
        J = np.zeros((1, 9))
        J[0, :] = np.array([
            pts2[0] * F[2, 1] - F[2, 2], pts2[1] * F[2, 1], pts2[2] * F[2, 1] - F[2, 0],
            pts2[0] * F[1, 2], pts2[1] * F[1, 2], pts2[2] * F[1, 2] - F[1, 0],
            pts2[0], pts2[1], pts2[2]
        ]).sum(axis=1)
        return J

    res = least_squares(cost, F.flatten(), jac=jacobian)
    return res.x.reshape(3, 3)


def ransac_f(pts_matches: np.array, th: float = 20.0, conf: float = 0.95, max_iter: int = 1000, LO_RANSAC=False):
    """
    RANSAC algorithm to find the best model with smart early stopping computation
    :param pts_matches: 2x3xN array of points
    :param th: threshold
    :param conf: confidence
    :param max_iter: maximum number of iterations
    :return: best_model, best_inliers
    """
    assert pts_matches.shape[0] == 2, f"Expected pts_matches to have shape (2, N), got {pts_matches.shape}"
    assert 0 <= conf <= 1, f"Expected 0 <= conf <= 1, got {conf}"
    assert max_iter > 0, f"Expected max_iter > 0, got {max_iter}"

    best_F = np.eye(3)
    best_inliers = np.zeros(pts_matches.shape[2], dtype=bool)

    i = 0
    while i < max_iter:
        # Sample 7 correspondences
        sample_pts = sample(pts_matches[0], pts_matches[1], 7)

        # Estimate Fundamental matrix from the sampled points
        FF = u2F(sample_pts[0], sample_pts[1])

        if len(FF) == 0:
            continue

        # Calculate the distance between the predicted points and the actual points
        for F in FF:
            # Calculate the distance between the predicted points and the actual points
            dist = fdist(F, pts_matches[0], pts_matches[1])
            inliers = dist < th

            if inliers.sum() > best_inliers.sum():
                best_F = F
                best_inliers = inliers

            if LO_RANSAC and inliers.sum() > 7:
                inlier_pts1 = pts_matches[0][:, best_inliers]
                inlier_pts2 = pts_matches[1][:, best_inliers]
                best_F = optimize_fundamental_matrix(best_F, inlier_pts1, inlier_pts2)

        max_iter = nsamples(best_inliers.sum(), pts_matches.shape[2], 7, conf)
        max_iter = min(max_iter, 100000)
        i += 1

    assert best_inliers.size == pts_matches.shape[2], f"Expected {pts_matches.shape[2]} inliers, got {best_inliers.size} inliers."
    assert best_F.shape == (3, 3), f"Expected (3, 3) fundamental matrix, got {best_F.shape} fundamental matrix."
    selected_points = np.where(best_inliers)[0]
    return best_F, selected_points
