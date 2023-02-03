import cv2 as cv
import numpy as np

# load the image
img = cv.imread("/users/charles/desktop/images/coins.jpg", 0)

# print(img)

# apply otsu's thresholding
ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# show the image
cv.imshow("Binary image", thresh)
cv.waitKey(0)
cv.destroyAllWindows()

