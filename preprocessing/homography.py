import cv2
import numpy as np
import os
from PIL import Image
import base64
from io import StringIO
from io import BytesIO


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