import cv2
import numpy as np

def match_images(images, sift):
    """
    Match two images using SIFT
    :param images:
    :param sift:
    :return:
    """
    assert len(images) == 2, f"images length is {len(images)}, expected 2"
    assert isinstance(sift, cv2.SIFT), f"sift type is {type(sift)}, expected cv2.SIFT"
    keypoints = []
    descriptors = []
    for image in images:
        keypoints_i, descriptors_i = sift.detectAndCompute(image, None)
        keypoints.append(keypoints_i)
        descriptors.append(descriptors_i)
    # Match keypoints
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors[0], descriptors[1], k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    # sort them by distance
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    # Keep the best 100 matches
    good_matches = good_matches[:100]

    # Get keypoints coordinates
    pts0 = np.array([keypoints[0][m.queryIdx].pt for m in good_matches])
    pts1 = np.array([keypoints[1][m.trainIdx].pt for m in good_matches])
    return pts0, pts1

