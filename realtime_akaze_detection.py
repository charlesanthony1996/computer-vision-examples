import cv2
import numpy as np

def resize_image(image, scale_percent):
    width = int(image.shape[1]* scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)


def feature_matching(img1, img2, kp1, desc1, method="bf"):
    
    kp2, desc2 = akaze.detectAndCompute(img2, None)


    # use BF matcher for feature matching
    if method == "bf":
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key = lambda x:x.distance)
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
    else:
        raise ValueError("Unknown method for feature matching")

    return img_matches

# load the input image
img = cv2.imread("/users/charles/desktop/images/doc_hudson_with_mcqueen.jpg")
if img is None:
    print("error: image not loaded")
    exit()

img = resize_image(img, 50)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

# print(gray)


akaze = cv2.AKAZE_create()
keypoints, descriptors = akaze.detectAndCompute(gray, None)

# print(descriptors)

# for real time keypoint detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.equalizeHist(gray_frame)
    kp_frame, desc_frame = akaze.detectAndCompute(gray_frame, None)


    frame_matches = feature_matching(gray, frame, keypoints, descriptors)

    cv2.imshow("Real time akaze descriptors ", frame_matches)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

