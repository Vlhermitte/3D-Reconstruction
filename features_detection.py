import cv2
import numpy as np

def match_images(images, feature2d):
    """
    Match two images using a feature detector and matcher
    :param images:
    :param feature2d:
    :return: pts0, pts1
    """
    assert len(images) == 2, f"images length is {len(images)}, expected 2"
    assert isinstance(feature2d, cv2.Feature2D), f"sift type is {type(feature2d)}, expected a cv2.Feature2D object"
    keypoints = []
    descriptors = []
    for image in images:
        keypoints_i, descriptors_i = feature2d.detectAndCompute(image, None)
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

