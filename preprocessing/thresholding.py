import cv2
from preprocessing import homography
import numpy as np

def preprocessingPhoto(img, points, gamma=None, constant=None, blockSize=None):
    if constant is None:
      constant = 10
    if blockSize is None:
      blockSize = 35

    max_color = homography.getMostFrequentColor(img)
    img = homography.removeScore(img, color=max_color)
    if gamma is None:
        img = homography.adjustImage(img)
    else:
        img = homography.adjustImage(img, gamma=gamma)

    img = homography.computeHomograpy(img, points)
    img = homography.unique_color(img)

    img = cv2.bilateralFilter(img, 5, sigmaColor=8, sigmaSpace=8)
    bw = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, constant)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # img = cv2.erode(bw, kernel)
    return bw

def preprocessingScans(img, threshold=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if threshold is None: 
        threshold = homography.getThreshold(gray)
    img = homography.sharpenDrawing(gray, threshold)
    return img, threshold

def getSplitsFromROI(roi): 
    stripes = 2
    rois = []
    width_stripe = (roi[1][0] - roi[0][0]) / stripes
    height_stripe = (roi[2][1] - roi[1][1]) / stripes
    for i in range(stripes):
        for j in range(stripes):
            p0 = (roi[0][0] + j * width_stripe, roi[0][1] + i * height_stripe )
            p1 = (roi[0][0] + (j+1) * width_stripe, roi[0][1] + i * height_stripe )
            p2 = (roi[0][0] + (j+1) * width_stripe, roi[0][1] + (i+1) * height_stripe )
            p3 = (roi[0][0] + j * width_stripe, roi[0][1] + (i+1) * height_stripe )
            new_roi = [p0, p1, p2, p3]
            rois.append(new_roi)

    return np.array(rois).astype(int)
