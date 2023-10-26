import cv2 as cv
import numpy as np

def orb_feature_matching(img1 , img2):
    # convert both images to grayscale
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)


    # create orb detector and detect keypoints and descriptors in both images
    orb = cv.ORB_create()
    orb.setMaxFeatures(10)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # use bf matcher to find the best matches
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # sort the matches based on distance (lowest distance is better)
    matches = sorted(matches, key=lambda x:x.distance)


    # draw matches
    img_matches = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

    return img_matches

# load input matches
img1 = cv.imread("/users/charles/desktop/images/doc_hudson.jpg")
img2 = cv.imread("/users/charles/desktop/images/doc_hudson_with_mcqueen.jpg")

# call the feature matching function
result = orb_feature_matching(img1, img2)


# save the output to a file
output_path = "/users/charles/desktop/images/matching_result.jpg"
cv.imwrite(output_path, result)
print(f"Matching result image saved at {output_path}")


# Display the resulting image
cv.imshow("ORB Feature Matching", result)
cv.waitKey(0)
cv.destroyAllWindows()