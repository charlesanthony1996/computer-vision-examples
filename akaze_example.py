import cv2
import numpy as np

# Load the input image
img = cv2.imread('/users/charles/desktop/images/doc_hudson_with_mcqueen.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create an AKAZE object
akaze = cv2.AKAZE_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = akaze.detectAndCompute(gray, None)

# Draw keypoints on the input image
img_keypoints = cv2.drawKeypoints(img, keypoints, None)

# Display the resulting image
cv2.imshow("AKAZE Keypoints", img_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
