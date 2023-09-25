import torch
import cv2 as cv

def compute_centroid(contour):
    M = cv.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

    else:
        cx, cy = 0, 0
    return cx, cy



# load image
image = cv.imread("/users/charles/downloads/images/apple.jpg")
# print(image)

# check if the image is loaded properly
if image is None:
    print("Error loading image")
    exit()


# convert the time to gray scale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# apply gaussian blur to reduce noise and improve thresholding
blur = cv.GaussianBlur(gray, (5, 5), 0)

# apply adaptive thresholding
thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

print(thresh)

# find contours in the threshold image
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


# find the largest contour based on area
cnt = max(contours, key= cv.contourArea)

# compare the centroid of largest contour
cx, cy = compute_centroid(cnt)


# draw a circle at the centroid
cv.circle(image, (cx, cy) , 10, (0, 0, 255), -1)

# display the result
cv.imshow("centroid", image)
cv.waitKey(0)
cv.destroyAllWindows()


