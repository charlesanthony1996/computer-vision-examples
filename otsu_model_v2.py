import cv2 as cv
import numpy as np

# load the image
img = cv.imread("/users/charles/desktop/images/coins.jpg", 0)

# print(img)

# bilateral filtering for noise reduction while preserving edges
filtered = cv.bilateralFilter(img, d= 20, sigmaColor=75, sigmaSpace=75)

# apply clahe (contrast limited adaptive histogram equalization)
clahe = cv.createCLAHE(clipLimit = 2.0, tileGridSize=(8, 8))
cli = clahe.apply(filtered)

# print(cli)

# adaptive thresholding
adaptive_thresh = cv.adaptiveThreshold(cli, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2)

# morphological operations to remove small noise and seperate connected objects
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(adaptive_thresh, cv.MORPH_OPEN, kernel, iterations = 2)
sure_bg = cv.dilate(opening, kernel, iterations = 3)
# print(kernel)

# finding sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# identifying unknown region
unknown = cv.subtract(sure_bg, sure_fg)

# marker labelling
ret, markers = cv.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# applying watershed to segment objects
img_bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
cv.watershed(img_bgr, markers)
img_bgr[markers == -1] = [0, 0, 255]

# finding contours
contours, _ = cv.findContours(sure_fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# drawing ellipse for each contour
for contour in contours:
    if len(contour) >= 5:
        ellipse = cv.fitEllipse(contour)
        # print(ellipse)
        cv.ellipse(img, ellipse, (0, 255, 0), 2)

# show the image with contours and ellipses
cv.imshow("Processed image", img)
cv.waitKey(0)
cv.destroyAllWindows()




