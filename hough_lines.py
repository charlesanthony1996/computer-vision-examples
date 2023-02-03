import cv2 as cv
import numpy as np

# load the image
img = cv.imread("/users/charles/desktop/images/chessboard.jpg")

print(img)

# convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# apply the houghlines function
lines = cv.HoughLines(gray, 1, np.pi/180, 200)

# iterate over the output lines and draw them on the image
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv.imwrite("/users/charles/desktop/images/houghlines_example_generated.jpg", img)

cv.imshow("lines", img)
cv.waitKey(0)
cv.destroyAllWindows()