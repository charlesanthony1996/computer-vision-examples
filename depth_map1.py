import numpy as np
import time
import cv2 as cv
from matplotlib import pyplot as plt

imgL = cv.imread("/users/charles/desktop/images/doc_hudson_with_mcqueen.jpg", cv.IMREAD_GRAYSCALE)
imgR = cv.imread("/users/charles/desktop/images/doc_hudson_with_mcqueen.jpg", cv.IMREAD_GRAYSCALE)

stereo = cv.StereoBM_create(numDisparities = 16, blockSize = 15)
disparity = stereo.compute(imgL, imgR)

plt.imshow(disparity, "gray")
plt.show()