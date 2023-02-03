import cv2 as cv

# load image
image = cv.imread("/users/charles/desktop/apple.jpg")

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# apply thresholding
ret, thresh = cv.threshold(gray, 127, 255, 0)
# print(ret)
# print(thresh)

# find contours in the thresholded image
contours, hierachy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cnt = max(contours, key=cv.contourArea)
M = cv.moments(cnt)

# print(cnt)

# extract centroid (x, y) in coords of the contour
cx = int(M["m10"] / M["m00"])
cy = int(M["m01"] / M["m00"])

cv.circle(image, (cx, cy), 10, (0, 0, 255), -1)

cv.imshow("centroid", image)
cv.waitKey(0)
cv.destroyAllWindows()
