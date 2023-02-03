import cv2 as cv
import numpy as np

# define the source and destination points
src_pts = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]]).reshape(-1, 1, 2)
dst_pts = np.float32([[0.5,0], [1.5, 0.5], [0, 1.5], [1, 2]]).reshape(-1, 1, 2)

# calculate the homography matrix using the ransac algorithm
homography_matrix, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

print(homography_matrix)