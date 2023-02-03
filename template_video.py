import cv2 as cv

# load the video
cap = cv.VideoCapture("/users/charles/desktop/tiger_woods_moments_2.mp4")

# load the template image
template = cv.imread("/users/charles/desktop/tiger_woods_face.jpg", cv.IMREAD_GRAYSCALE)

print(template)

while True:
    # read the next frame from the video
    ret, frame = cap.read()

    # exit if the loop if the video has ended
    if not ret:
        break
    
    # convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # apply template matching
    res = cv.matchTemplate(gray, template, cv.TM_CCOEFF_NORMED)


    # find the location of the highest match
    min_val, max_val , min_loc, max_loc = cv.minMaxLoc(res)

    # draw a rectangle around the matched region
    top_left = max_loc
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
    cv.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)

    # display the frame
    cv.imshow("Video", frame)

    # exit the loop if the 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


# release the video capture object
cap.release()

# close all windows
cv.destroyAllWindows()


    

