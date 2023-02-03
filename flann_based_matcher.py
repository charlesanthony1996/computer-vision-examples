import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# query image
img1 = cv.imread("/users/charles/desktop/images/doc_hudson.jpg", cv.IMREAD_GRAYSCALE)

# train image
img2 = cv.imread("/users/charles/desktop/images/doc_hudson_with_mcqueen.jpg", cv.IMREAD_GRAYSCALE)

# initiate sift detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)


# flann parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 1)

# or pass empty dictionary
search_params = dict(checks = 10)

flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k = 2)

# need to draw only matches , so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]

# ratio test as per lowes paper
for i , (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]


draw_params = dict(matchColor = (0, 255, 0),
singlePointColor = (0, 255, 0),
matchesMask = matchesMask, 
flags = cv.DrawMatchesFlags_DEFAULT)

img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

plt.imshow(img3)

plt.show()

