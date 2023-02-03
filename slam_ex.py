import cv2
import numpy as np
from collections import deque

# Create a feature detector object
feature_detector = cv2.ORB_create()

# Create a feature matcher object
feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Create a camera object
camera = cv2.VideoCapture(0)

# Create a buffer of previous frames
frame_buffer = deque(maxlen=10)

# Initialize the first frame
ret, frame = camera.read()
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_kp, prev_des = feature_detector.detectAndCompute(prev_gray, None)

while True:
    # Read a new frame
    ret, frame = camera.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect and compute features in the current frame
    kp, des = feature_detector.detectAndCompute(gray, None)

    # Match the features between the current and previous frame
    matches = feature_matcher.match(prev_des, des)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract the matched keypoints
    src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute the essential matrix
    E, mask = cv2.findEssentialMat(src_pts, dst_pts)

    # Extract the rotation and translation
    _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts)

    # Update the previous frame
    prev_gray = gray
    prev_kp = kp
    prev_des = des

    # Draw the matches on the current frame
    img_matches = cv2.drawMatches(prev_gray, prev_kp, gray, kp, matches[:20], None, flags=2)
    cv2.imshow("Matches", img_matches)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()
