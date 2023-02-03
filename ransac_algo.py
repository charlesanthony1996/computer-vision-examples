import numpy as np
import random

def fit_line(points):
    # fit a line to the given set of 2d points using least squares method
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return k, b


def distance_to_line(point, k, b):
    # compute the perpendicular distance from a point to a line
    x, y = point
    d = np.abs(k*x - y + b)/ np.sqrt(k**2+ 1)
    return d


def RANSAC(points, threshold, num_iterations):
    # fit a line to the given set of 2d points using the ransac algorithm
    best_line = None
    best_inliers = []
    num_points = points.shape[0]

    for i in range(num_iterations):
        # randommly select two points
        idx = random.sample(range(num_points), 2)
        sample = points[idx,:]

        # fit a line to the sample
        k, b = fit_line(sample)

        # compute the distances from all points to the line
        distances = [distance_to_line(p, k, b) for p in points]

        # determine the inliers
        inliers = np.where(np.array(distances) < threshold)[0]

        # update the best line if necessary
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_line = (k, b)


    # re fit the line to all the inliers
    final_line = fit_line(points[best_inliers,:])
    return final_line


num_points = 100
x = np.linspace(0, 10, num_points)
y = 2 * x + 1 + np.random.normal(0, 1 , num_points)
y[-20:] = y[-20:] + 10 * np.random.rand(20)
points = np.column_stack((x, y))

# fit a line to the points using ransac
threshold = 2
num_iterations = 1000
final_line = RANSAC(points, threshold,num_iterations)
k, b = final_line
print("Final line: y = {:.2f}x + {:.2f}".format(k, b))

