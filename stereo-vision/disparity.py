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


input_image_left = cv2.imread('image_pairs/2_a.jpg', 0)
input_image_right = cv2.imread('image_pairs/2_b.jpg', 0)

good, points_left, points_right = get_matches_from_sift_descriptors(input_image_left, input_image_right)

points_left = np.int32(points_left)
points_right = np.int32(points_right)

F, mask = cv2.findFundamentalMat(points_left, points_right, cv2.FM_LMEDS)

# keep inliner points
points_left = points_left[mask.ravel() == 1]
points_right = points_right[mask.ravel() == 1]

img5, img6 = find_and_draw_epipolar_lines(input_image_left, input_image_right, points_left, points_right)
img3, img4 = find_and_draw_epipolar_lines(input_image_right, input_image_left, points_right, points_left)

# draw original images with their epipolar lines
# plt.subplot(121), plt.imshow(img5)
# plt.subplot(122), plt.imshow(img3)
# plt.show()

retBool, rect_matrix_left, rect_matrix_right = cv2.stereoRectifyUncalibrated(points_left, points_right, F, (1440, 1080))

rectified_image_left = cv2.warpPerspective(input_image_left, rect_matrix_left, (1440, 1080))
rectified_image_right = cv2.warpPerspective(input_image_right, rect_matrix_right, (1440, 1080))

# show rectify calibrated function's results
# plt.subplot(121), plt.imshow(dst11, 'gray')
# plt.subplot(122), plt.imshow(dst22, 'gray')
# plt.show()

window_size = 2
min_disparity = 10
max_disparity = 74
num_disparity = max_disparity - min_disparity  # Needs to be divisible by 16

stereo = cv2.StereoSGBM_create(minDisparity=min_disparity, numDisparities=num_disparity, blockSize=5, uniquenessRatio=5,
                               speckleWindowSize=5, speckleRange=5, disp12MaxDiff=1,
                               P1=8*3*window_size**2, P2=32*3*window_size**2)

disparity = stereo.compute(rectified_image_left, rectified_image_right)
plt.imshow(disparity, 'gray')
plt.show()
