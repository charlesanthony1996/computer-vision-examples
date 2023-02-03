import cv2 as cv
import numpy as np

# im = cv.imread("/users/charles/desktop/apple.jpg")
# imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(imgray, 127, 255, 0)
# contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# # print(contours)
# # print(hierarchy)

# cv.drawContours(im, contours[4], -1, (0, 255, 0), 3)
# cv.imshow("img", im)
# cv.waitKey(0)

# for a video file

cap = cv.VideoCapture("/users/charles/desktop/tigerwoods.mp4")

# print(cap)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # apply thresholding to the frame
    ret, thresh = cv.threshold(gray, 127, 255, 0)

    # find contours in the thresholded frame
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # draw all contours on the current frame
    cv.drawContours(frame, contours, -1, (0, 255, 0), 3)

    # show the current frame
    cv.imshow("Video", frame)
    
    # exit if the user presses 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


