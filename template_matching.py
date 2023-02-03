import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("/users/charles/desktop/barca_team_2.jpg", 0)
img2 = img.copy()
template = cv.imread("/users/charles/desktop/messi_face.jpg", 0)
w, h = template.shape[::-1]


# all the 6 methods for comparison in a list
methods = ["cv.TM_CCOEFF", "cv.TM_CCOEFF_NORMED", "cv.TM_CCORR", 'cv.TM_CCORR_NORMED', "cv.TM_SQDIFF","cv.TM_SQDIFF_NORMED"]


for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # apply template matching
    res = cv.matchTemplate(img , template, method)
    min_val, max_val , min_loc , max_loc = cv.minMaxLoc(res)

    # if the method is TM_SQDIFF_NORMED , take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    
    else:
        top_left = max_loc

    bottom_right = (top_left[0]+ w, top_left[1] + h)
    cv.rectangle(img, top_left, bottom_right, 255, 2)

    plt.subplot(121), plt.imshow(res, cmap="gray")
    plt.title("Matching result"), plt.xticks([]), plt.yticks([])
    plt.title("Detected point"), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()