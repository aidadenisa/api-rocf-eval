
import os
import base64
import cv2
from preprocessing import thresholding
import pandas as pd
import numpy as np
from datetime import datetime

def saveROCFImages(original, threshed, uploadPath, doctorID, patientCode, date):
    if date is None:
        date = datetime.today()
    date = date.strftime("%d:%m:%Y-%H:%M:%S")
    # filename = secure_filename(imageb64.filename)
    filename = patientCode + "_" + date
    doctorFolderPath = os.path.join(uploadPath, doctorID)
    if not os.path.isdir(doctorFolderPath):
        os.mkdir(os.path.join(doctorFolderPath))
    originalFolderPath = os.path.join(doctorFolderPath, 'original')
    if not os.path.isdir(originalFolderPath):
        os.mkdir(os.path.join(originalFolderPath))
    threshedFolderPath = os.path.join(doctorFolderPath, 'threshed')
    if not os.path.isdir(threshedFolderPath):
        os.mkdir(os.path.join(threshedFolderPath))

    if filename != '':
        cv2.imwrite(os.path.join(originalFolderPath, filename +'.png'), original)
        cv2.imwrite(os.path.join(threshedFolderPath, filename +'.png'), threshed)

    return filename

    # if filename != '':
    #     image = base64.b64decode(imageb64)
    #     # file_ext = os.path.splitext(filename)[1]
    #     # if file_ext not in app.config['UPLOAD_EXTENSIONS']:
    #     #     abort(400)
        
    #     if not os.path.isdir(doctorFolderPath):
    #         os.mkdir(os.path.join(doctorFolderPath))

    #     location = os.path.join(doctorFolderPath, filename + ".png")
        
    #     with open(location, "wb") as fh:
    #         fh.write(image)

def generateThresholdedHomographies(sourceFolderURL, destinationFolderURL, type="png"): 
    if not os.path.isdir(sourceFolderURL):
        print("error: source url is not a folder")
        return

    # get points
    file_homog = pd.read_json(os.path.join(sourceFolderURL, 'points.txt'), lines=True).set_index('name')

    for filename in os.listdir(sourceFolderURL):
        if filename.endswith(tuple([".png", ".jpg", ".PNG", ".JPG"])):
            patientCode = filename[:-4]

            # identify the 5 points of interest the homogram 
            if patientCode in file_homog.index :
                points = np.array(file_homog.loc[patientCode].to_numpy()[0])

                if points.shape == (1,):
                    points = np.array(points[0])
                #transform points from a matrix into an array of tuples
                points = [tuple(x) for x in points] 

                img = cv2.imread(os.path.join(sourceFolderURL,filename))
                # See if we need to do the scanned version better, or this one works also
                img = thresholding.preprocessingPhoto(img, points)

                if not os.path.isdir(destinationFolderURL):
                    os.mkdir(os.path.join(destinationFolderURL))

                location = os.path.join(destinationFolderURL, patientCode + ".png")
                
                cv2.imwrite(location, img)

    




