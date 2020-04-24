import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_matches_from_sift_descriptors(img1, img2, flann_index_kdtree=0):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    saved_matches = []
    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            saved_matches.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    return saved_matches, pts1, pts2


def draw_lines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 8)
        img1 = cv2.circle(img1, tuple(pt1), 12, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 12, color, -1)
    return img1, img2


def find_and_draw_epipolar_lines(img1, img2, pts1, pts2, fundamental_matrix):
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, fundamental_matrix)
    lines1 = lines1.reshape(-1, 3)
    return draw_lines(img1, img2, lines1, pts1[0:5], pts2[0:5])
