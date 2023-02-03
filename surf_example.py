import cv2 as cv
import numpy as np

# load the input image
img = cv.imread("/users/charles/desktop/images/doc_hudson.jpg", 0)

# convert the image to grayscale
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Create a SURF object
surf = cv.xfeatures2d.SURF_create(400)

kp, des = surf.detectAndCompute(img,None)


# program doesnt work