import time
import cv2 as cv
import numpy as np
import torch


# This is an example and you would use the actual data from your calibration process.
# Camera matrix (3x3)
mtx = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])

# Distortion coefficients (k1, k2, p1, p2, k3)
dist = np.array([0.1, 0.05, 0.001, 0.001, 0.01])

# Rotation vectors (example)
rvecs = [np.array([[0.01],
                   [0.02],
                   [0.03]])]

# Translation vectors (example)
tvecs = [np.array([[0.1],
                   [0.2],
                   [0.3]])]

np.savez('B.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# Load previously saved calibration data
with np.load('B.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

# Object points in the real world space
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Load the image
img = cv.imread("/users/charles/desktop/images/chessboard.jpg")
print(img.shape)
# Continue with your pose estimation code...
# pytesting here?
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# print(gray)


# find the chess board corners
ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

if ret == True:
    # find the rotation and translation vectors
    ret, rvecs, tvecs = cv.solvePnP(objp, corners, mtx, dist)

    # convert rotation vector to rotation matrix
    rmat, _ = cv.Rodrigues(rvecs)

    print(rmat)
    print("test")
    # print(img.shape)

    print("rotation vector: ", rvecs)
    print("translation matrix: ", tvecs)
    print ("rotation matrix: ", rmat)










