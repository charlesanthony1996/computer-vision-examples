import cv2
import imutils

def is_hip(x, y, w, h, frame_shape):
    aspect_ratio = w / h
    hip_ar_threshold = 0.8
    hip_size_threshold = 0.15 * frame_shape[0] * frame_shape[1]
    hip_y_threshold = 0.8 * frame_shape[0]
    if aspect_ratio > hip_ar_threshold and w*h > hip_size_threshold and y < hip_y_threshold:
        return True
    else:
        return False

def is_foot(x, y, w, h, frame_shape):
    aspect_ratio = w / h
    hip_ar_threshold = 0.8
    hip_size_threshold = 0.15 * frame_shape[0] * frame_shape[1]
    hip_y_threshold = 0.8 * frame_shape[0]
    if aspect_ratio > hip_ar_threshold and w*h > hip_size_threshold and y < hip_y_threshold:
        return True
    else:
        return False

def calculate_distance(hip, foot):
    distance = ((hip[0] - foot[0])**2 + (hip[1] - foot[1])**2)**(1/2)
    return distance

# initialize the video stream
vs = cv2.VideoCapture("/users/charles/desktop/tigerwoods.mp4")

# initialize the HOG descriptor
hog = cv2.HOGDescriptor()

# set the SVM detector
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# loop over the frames
while True:
    # read the frame
    _, frame = vs.read()

    # check if the frame is None
    if frame is None:
        break

    # resize the frame
    frame = imutils.resize(frame, width=600)

    # detect the hip and foot in the frame
    hip = None
    foot = None
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
    for (x, y, w, h) in rects:
        if is_hip(x, y, w, h, frame.shape):
            hip = (x, y, w, h)
        elif is_foot(x, y, w, h, frame.shape):
            foot = (x, y, w, h)

    # check if the hip and foot were detected
    if hip is not None and foot is not None:
        # calculate the distance between the hip and foot
        distance = calculate_distance(hip, foot)

        # draw the distance on the frame
        cv2.putText(frame, f"Distance: {distance:.2f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# release the video stream
vs.release()

