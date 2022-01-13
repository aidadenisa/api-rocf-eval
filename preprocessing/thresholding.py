import cv2
from preprocessing import homography

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
    if threshold is None: 
        threshold = homography.getThreshold(img)
        img = homography.sharpenDrawing(img)
    else:
        img = homography.sharpenDrawing(img, threshold)
    return img, threshold
