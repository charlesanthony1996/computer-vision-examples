import cv2 as cv
import numpy as np
import sys
import sympy


A = cv.imread("/users/charles/desktop/apple.jpg")
B = cv.imread("/users/charles/desktop/orange.jpg")


# generate gaussian pyramid of A
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv.pyrDown(G)
    gpA.append(G)


# generate gaussian pyramid of B
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv.pyrDown(G)
    gpB.append(G)

#generate laplacian pyramid for A
lpA = [gpA[5]]
for i in range(5, 0, -1):
    size = (gpA[i-1].shape[1], gpA[i-1].shape[0])
    # print(size)
    GE = cv.pyrUp(gpA[i], dstsize=size)
    # print(GE)
    L = cv.subtract(gpA[i-1], GE)
    # print(L)
    lpA.append(L)
    # print(lpA)
    # print(lpA)


# generate laplacian pyramid of B
lpB = [gpB[5]]
for i in range(5, 0, -1):
    size = (gpB[i - 1].shape[1], gpB[i - 1].shape[0])
    # print(size)
    GE = cv.pyrUp(gpB[i], dstsize=size)
    # print(GE)
    L = cv.subtract(gpB[i - 1], GE)
    lpB.append(L)
    # print(lpB)


# now add left and right halfes of images in each level
LS = []
for la, lb in zip(lpA, lpB):
    rows, cols, dst = la.shape
    ls = np.hstack((la[:, 0:cols// 2 ], lb[:, 0:cols//2]))
    LS.append(ls)
    print(LS)

# now reconstruct


ls_ = LS[0]
for i in range(1, 6):
    ls_ = cv.pyrUp(ls_)
    ls_= cv.resize(ls_, LS[i].shape[-2::-1])
    ls_ = cv.add(ls_, LS[i])


# image wth direct connecting each half
B = cv.resize(B, A.shape[-2::-1])
real = np.hstack((A[:, :cols//2], B[:, :cols//2:]))
# print(real)
cv.imwrite("/users/charles/desktop/pyramid_blending.jpg", ls_)
cv.imwrite("/users/charles/desktop/direct_blending.jpg", real)
