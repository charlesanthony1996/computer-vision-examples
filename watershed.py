import cv2 as cv
import numpy as np

# load the image
img = cv.imread("/users/charles/desktop/images/coins.jpg")

# convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# apply gaussian blur to reduce noise
blurred = cv.GaussianBlur(gray, (5, 5), 0)

# threshold the image to create a binary image
thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

# perform morphpological operations 
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = 2)

# find sure background area
sure_bg = cv.dilate(opening, kernel, iterations = 3)

# find sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# find unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# marker labelling
ret, markers = cv.connectedComponents(sure_fg)

# add one to all labels so that sure background is not 0 or 1
markers = markers + 1

# now mark the region of unknown ith zero
markers[unknown == 255] = 0

# apply the watershed algorithm
mask = np.zeros(gray.shape, np.uint8)

# color the segments
for i in range(1, np.max(markers) +  1):
    mask[markers == i] = np.random.randint(0, 256)

# Apply color map
color_mask = cv.applyColorMap(mask, cv.COLORMAP_JET)


# overlay the mask on the original image
output = cv.addWeighted(img, 0.7, color_mask, 0.3, 0)


# show the image
cv.imshow("Segmented image", output)
cv.waitKey(0)
cv.destroyAllWindows()


