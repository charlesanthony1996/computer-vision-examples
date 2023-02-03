import cv2 as cv

import numpy as np

# load an image in grayscale
image = cv.imread("/users/charles/desktop/orange.jpg", cv.IMREAD_GRAYSCALE)

bins = 256

# compute the histogram
hist = cv.calcHist([image], [0], None, [bins], [0, 256])

# plot the histogram
import matplotlib.pyplot as plt
plt.hist(image.ravel(), bins, [0, 256])
plt.show()
