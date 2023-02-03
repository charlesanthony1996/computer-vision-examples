import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# from sympy import symbols, diff
# x = symbols("x")
# y = x ** 2
# dy_dx = diff(y, x)
# print(dy_dx)

# import numpy as np
# def y(x):
#     return x * 2

# dx = 0.0001
# x = 3
# dy_dx = (y(x+dx) - y(x))/dx
# print(dy_dx)

# simple averaging filter without scaling parameter
mean_filter = np.ones((3, 3))

# create a gaussian filter
x = cv.getGaussianKernel(5, 10)
gaussian = x * x.T

# print(gaussian)

# different edge detection filters
scharr = np.array([[-3, 0, 0], [-10, 0, 10], [-3, 0, 3]])

# sobel in x direction
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

# print(sobel_x)

# sobel in y direction
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# print(sobel_y)

# laplacian 
laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
# print(laplacian)

filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr]
filter_name = ["mean_filter", "gaussian", "laplacian","sobel_x", "sobel_y", "scharr"]

fft_filters = [np.fft.fft2(x) for x  in filters]
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
mag_spectrum = [np.log(np.abs(z) + 1) for z in fft_shift]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(mag_spectrum[i], cmap="gray")
    plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])

plt.show()