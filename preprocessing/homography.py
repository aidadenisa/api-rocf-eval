import cv2
import numpy as np
import os
from PIL import Image
import base64
from io import StringIO
from io import BytesIO
from skimage.filters import threshold_local


p_dst = [(382, 219),(852, 219), (852, 537), (382, 537)]

def computeHomograpy(image, points):
    mask = np.ones(5, dtype=int)
    mask[2]=0
    img = cv2.imread(os.path.join(os.getcwd(), 'templates', 'original_rey[1360x768].png'))
    right_points = np.array(points)[np.ma.make_mask(mask)]
    hm, status = cv2.findHomography(np.array(right_points), np.array(p_dst))
    nH, nW, _ = img.shape
    im_dst =  cv2.warpPerspective(image, hm, (nW, nH))
    return im_dst

def convertImageB64ToMatrix(imageb64):
    image = base64.b64decode(imageb64)     
    npImage = np.frombuffer(image, dtype=np.uint8)
    matImage = cv2.imdecode(npImage, 3)
    return matImage

def convertImageFromMatrixToB64(imageMatrix): 
    retval, buffer = cv2.imencode('.jpg', imageMatrix)
    imgb64 = base64.b64encode(buffer)
    return imgb64

def saveImageFromB64(imageb64):
    imgdata = base64.b64decode(imageb64)
    filename = 'temp/original_image.jpg'
    with open(filename, 'wb') as f:
        f.write(imgdata)

def removeScore(image, color=(255, 255, 255)):
    #convert to HSV from RGB
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 30, 10])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Range for upper range
    lower_red = np.array([160, 30, 10])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Generating the final mask to detect red color
    BW = (mask1 + mask2) > 0
    result = image.copy()
    # result = cv2.bitwise_and(result, result, mask=BW)

    # set the pixels that are red in the iamge to white
    result[BW] = color

    return result
def emphasiseColor(image, contrast=0.2, brightness=(-20)):
    out = cv2.addWeighted( image, contrast, image, 0, brightness)
    return out

def adjustImage(image, increaseBrightness=False, alpha=1.4, beta=30):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray[gray == 0] = 255

    if increaseBrightness == True:
        # Linear transformation to make image lighter
        new_image = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    
    return new_image

def sharpenDrawing(image):
    image = cv2.bilateralFilter(image, 10, sigmaColor=15, sigmaSpace=15)
    # plt.imshow(dst, cmap='gray')
    # plt.show()

    
    #initialize the drawing array as fully white 
    threshed = np.ones(image.shape, np.uint8) * 255
    #if there is in the image at least a pixel that is not white (meaning we have drawing lines in the image)
    if np.any(image < 255):
        #create a histogram to see the distribution of pixels that are not white
        hist, _ = np.histogram(image[image < 255].flatten(), range(257))
        print(hist)
        # apply image thresholding using Unimodal Thresholding
        thresh_val = maxDeviationThresh(hist)
        print(thresh_val)
        # create a mask for when the image's pixels are under the threshold value, which will be true for most of the colored values that are belonging to the drawing (paper and white stuff will be bigger then threshold) => True for belonging to the drawing, 0 for not
        mask = image < thresh_val
        # print(mask)

        # on the white canvas created before, set the pixels that have the value under the threshold (so they belong to the drawing) to black (0) => you have extracted the drawing in a sharper version 
        threshed[mask] = 0
    # print(image)
    # print(threshed)
    return threshed


# method used to apply thresholding to the image (thesis section: 2.3.1 Unimodal Tresholding)
def maxDeviationThresh(hist):
    #get the color (value) that was used the most according to the histogram
    maximum = max(hist)
    #get the index of the color that was used the most
    index_max = list(hist).index(maximum)
    index_min = 0

    # max i where hi = 0
    for i in range(0, index_max):
        if not hist[i] and hist[i + 1]:
            index_min = i
            break
 
    distances = [] 
    x1 = index_min
    y1 = hist[index_min]
    
    x2 = index_max
    y2 = hist[index_max]

    #for all the colors in the range between the color used minimally and the color used the most
    for i in range(index_min + 1, index_max):
        x0 = i
        y0 = hist[i]
        # calculate the distance of the point to the line created by the points of the highest value and the lowest value in the histogram
        distance = np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt(
            (y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances.append(distance)
    #choose the index corresponging to the value that generates the maximum distance from the line created by the extremes
    if index_min < index_max - 1:
        T_index = distances.index(max(distances))
    else:
        T_index = -index_min
    return T_index + index_min
