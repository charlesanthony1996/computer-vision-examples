import cv2 as cv

# loading the image
image = cv.imread("/users/charles/desktop/balls.jpg")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# apply a thresholding to the image
ret, thresh = cv.threshold(gray, 127, 255, 0)

# find contours in the thresholding image
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# find the convex hull of the largest contour
cnt = max(contours, key=cv.contourArea)
hull = cv.convexHull(cnt)

#  find the minimum enclosing circle of the convex hull
(x, y) , radius = cv.minEnclosingCircle(hull)
center = (int(x), int(y))
radius = int(radius)

# draw the enclosing circle on the image
cv.circle(image, center, radius, (0, 255, 0), 2)

# show the resulting image
cv.imshow("min enclosing circle", image)
cv.waitKey(0)
cv.destroyAllWindows()



