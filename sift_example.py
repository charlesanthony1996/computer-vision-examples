#program doesnt work
#patent issues in sift
#was removed from open cv

import cv2 as cv
import numpy as np

# load the input image
img = cv.imread("/users/charles/desktop/images/doc_hudson.jpg")

# convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# create a sift object
sift = cv.xfeatures2d_SIFT()

# detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

# draw keypoints on the input image
img_keypoints = cv.drawKeypoints(img, keypoints, None)

# display the resulting image
cv.imshow("SIFT keypoints", img_keypoints)
cv.waitKey(0)
cv.destroyAllWindows()