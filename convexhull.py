import cv2 as cv

# load image
image = cv.imread("/users/charles/desktop/apple.jpg")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# print(gray)

ret, thresh = cv.threshold(gray, 127, 255, 0)

# find contours in the threshold image
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


# find the convex hull of the largest contour
cnt = max(contours, key=cv.contourArea)

hull = cv.convexHull(cnt)
# print(hull)

#  find the bounding rectangle of the convex hull
x, y, w, h = cv.boundingRect(hull)

# draw the bounding rectangle on the image
cv.rectangle(image, (x, y), (x* 2 + w //2, y*2 + h//2), (0, 255, 0), 4)

cv.imshow("Bounding rectangle", image)
cv.waitKey(0)
cv.destroyAllWindows()



