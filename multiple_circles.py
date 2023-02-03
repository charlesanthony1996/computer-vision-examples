import cv2 as cv
import numpy as np

image = cv.imread("/users/charles/desktop/balls.jpg", cv.IMREAD_GRAYSCALE)

# apply a gaussian filter
image = cv.GaussianBlur(image, (5, 5), 0)

# apply a hough circle transform
circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, 1, 5 , param1=80, param2 = 40, minRadius = 0, maxRadius=0)

# if circles are found
if circles is not None:
    # convert the x, y coordinates and radius of the circle to integers
    circles = np.round(circles[0, :]).astype("int")
    
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle on the image
        cv.circle(image, (x, y) , r, (0, 255, 0), 2)


# show the resulting image
cv.imshow("Hough Circles", image)
cv.waitKey(0)
cv.destroyAllWindows()
