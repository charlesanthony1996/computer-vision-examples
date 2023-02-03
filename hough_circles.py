import numpy as np
import cv2 as cv

# load the image
img = cv.imread("/users/charles/desktop/images/open_cv_logo.jpg")

# convert the img to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# apply the hough circles function
circles = cv.HoughCircles(gray , cv.HOUGH_GRADIENT, 1, 30, param1=50, param2= 40, minRadius=0, maxRadius=0)

# make sure circles were found
if circles is not None:
    # convert the x y coords and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # iterate over the circles and draw them on the image
    for (x, y, r) in circles:
        cv.circle(img, (x, y), r, (0, 255, 0), 3)
        cv.circle(img, (x, y), 2, (0, 0, 255), 3)


# show the image
cv.imshow("circle", img)
cv.waitKey(0)
cv.destroyAllWindows()