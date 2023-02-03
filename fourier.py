import cv2 as cv
import numpy as np

img = cv.imread("/users/charles/desktop/chessboard.jpg", cv.IMREAD_GRAYSCALE)

# get the image dimensions
rows, cols = img.shape

# get the optimal size for the DFT (closer to a power of 2)
nrows = cv.getOptimalDFTSize(rows)
ncols = cv.getOptimalDFTSize(cols)

# pad the image with zeros to get the optimal size
img_padded = np.zeros((nrows, ncols) , dtype=np.complex64)
img_padded[:rows, :cols] = img

# create the complex image for the DFT
complex_img = np.zeros((nrows, ncols), dtype=np.complex64)
complex_img[:, :] = img_padded

# create a single channel image (float32)
img_padded = np.float32(img_padded)

# perform the dft
dft = cv.dft(img_padded, flags=cv.DFT_COMPLEX_OUTPUT)

# shift the DFT to the center
dft_shift = np.fft.fftshift(dft)

#  get the magnitude of the dft
magnitude = cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

# normalize the magnitude
magnitude = cv.normalize(magnitude, None, 0, 1, cv.NORM_MINMAX)

# plot the magnitude
import matplotlib.pyplot as plt
plt.imshow(magnitude, cmap="gray")
plt.show()