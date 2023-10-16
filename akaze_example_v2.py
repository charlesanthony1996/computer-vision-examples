import cv2
import numpy as np

def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width / height)
    return cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    # this line doesnt work for some reason
    # return cv2.resize (image, dim, interpolation=cv2.INTER_AREA)


# load the input image
img = cv2.imread("/users/charles/desktop/images/doc_hudson_with_mcqueen.jpg")

# error check for image
if img is None:
    print("Error: image not loaded")
    exit()

# if the image is large, resize it
img = resize_image(img, 50)

# print(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# print(gray)

# enhance contrast using histogram equalization
gray = cv2.equalizeHist(gray)

akaze = cv2.AKAZE_create()

# print(akaze)

# detect keypoints using histogram equalization
keypoints, descriptors = akaze.detectAndCompute(gray, None)

# draw keypoints with rich features on the input image
img_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags = cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

# display the resulting image
cv2.imshow("Akaze keypoints", img_keypoints)

cv2.imwrite("/users/charles/desktop/images/output_akaze_keypoints.jpg", img_keypoints)

cv2.waitKey(0)
cv2.destroyAllWindows()