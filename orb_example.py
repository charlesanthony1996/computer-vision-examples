import cv2 as cv
import numpy as np

# load the input image
img = cv.imread("/users/charles/desktop/images/doc_hudson.jpg")

# convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# create an orb object and adjust its parameters
orb = cv.ORB_create()
# example of setting max features , adjust as needed
orb.setMaxFeatures(500)

# detect keypoints and descriptors and compute descriptors
keypoints, descriptors = orb.detectAndCompute(gray, None)

# display number of keypoints detected
print(f"Number of keypoints detected: {len(keypoints)}")

# draw keypoints on a seperate output image (with green colour for keypoints)
img_keypoints = img.copy()
cv.drawKeypoints(img, keypoints, img_keypoints, color=(0, 255, 0))

# save the output to a file
output_path = "/users/charles/desktop/images/doc_hudson_keypoints2.jpg"
cv.imwrite(output_path, img_keypoints)
print(f"keypoints image saved at {output_path}")


# display the resulting image
cv.imshow("orb keypoints", img_keypoints)
cv.waitKey(0)
cv.destroyAllWindows()



