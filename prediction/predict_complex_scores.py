import os
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import csv
import copy

from preprocessing import homography
from prediction import Visualization

def initializeTemplates(template_folder, templates):
    for img in os.listdir(template_folder):
        if img != 'template.png':
            template = homography.background_thumbnail(cv2.imread(os.path.join(template_folder, img),
                                                            cv2.IMREAD_GRAYSCALE), 'L', (input_shape[0], input_shape[1]))        
            template = template.astype('float32')
            template /= 255
            templates[label_dict[img]] = template

def initializeResultsCSV(root): 
    if not os.path.isdir(os.path.join(root, 'results')):
        os.makedirs(os.path.join(root, 'results'))
    fieldnames = ['names', 'scores', 'distances', 'rect']
    if not os.path.isfile(os.path.join(root, 'results', 'scores.csv')):
        with open(os.path.join(root, 'results', 'scores.csv'), "w") as f:
            f.write(','.join(fieldnames)+'\n')
 
    folders = pd.read_csv(os.path.join(root, 'results', 'scores.csv'), header=0, usecols=['names']).values.squeeze()
    return fieldnames, folders

def predictComplexScores(image, points):
    or_points2 = copy.deepcopy(or_points)  
    points = np.array(points)
    if points.shape ==(1,):
      points = np.array(points[0])
    print(points.shape)
    points = [tuple(x) for x in points]
    print(points)

    r_points = homography.computeHomographyRhomb(points)
    or_points2['rhomb'].insert(2, r_points[0])

    img = image
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    for x,y in or_points2.items():
        or_points2[x] = np.array([int(p*(scale_percent/100)) for p in y])
        or_points2[x][0:2]-=pad
        or_points2[x][2:]+=pad
        or_points2[x] = or_points2[x].tolist()
    
    # Display the ROIs on the image
    # iiiiimage = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    # iiiiimage = cv2.rectangle(iiiiimage, (or_points2['cross'][0], or_points2['cross'][1]), (or_points2['cross'][2], or_points2['cross'][3]), (0,255, 0), 2)
    # iiiiimage = cv2.rectangle(iiiiimage, (or_points2['face'][0], or_points2['face'][1]), (or_points2['face'][2], or_points2['face'][3]), (0,255, 0), 2)
    # iiiiimage = cv2.rectangle(iiiiimage, (or_points2['rail'][0], or_points2['rail'][1]), (or_points2['rail'][2], or_points2['rail'][3]), (0,255, 0), 2)
    # iiiiimage = cv2.rectangle(iiiiimage, (or_points2['rhomb'][0], or_points2['rhomb'][1]), (or_points2['rhomb'][2], or_points2['rhomb'][3]), (0,255, 0), 2)
    # iiiiimage = cv2.rectangle(iiiiimage, (or_points2['rett_diag'][0], or_points2['rett_diag'][1]), (or_points2['rett_diag'][2], or_points2['rett_diag'][3]), (0,255, 0), 2)
    # iiiiimage = cv2.rectangle(iiiiimage, (or_points2['rect'][0], or_points2['rect'][1]), (or_points2['rect'][2], or_points2['rect'][3]), (0,255, 0), 2)
    # iiiiimage = cv2.rectangle(iiiiimage, (or_points2['cross_vert'][0], or_points2['cross_vert'][1]), (or_points2['cross_vert'][2], or_points2['cross_vert'][3]), (0,255, 0), 2)

    csv_file = open(os.path.join(root, 'results', 'scores.csv'), mode='a')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    app = Visualization.Visualization('newImage', img, or_points2, templates, input_shape, writer)

    result = app.run()      
    csv_file.close()
    del app
    
    return result  


root='./' #modificare se cartella rinominata
label_dict={
    'cross.png':0,
    'face.png':1,
    'rail.png':2,
    'rombo.png':3,
    'rett_diag.png':4,
    'rect.png':5,
    'cross_vert.png':6
}
or_points = {
    "cross":[324,109,378,383],
    "face":[742, 287, 829, 373],
    "rail":[617, 383, 847, 534],
    "rhomb":[852, 229, 531],
    "rett_diag":[359, 280, 522, 476],
    "rect":[360, 525, 510, 680],
    "cross_vert":[502, 540, 810, 661]
}
scale_percent = 100
pad = 0

input_shape = (100,100,1)
template_folder=os.path.join(root, 'templates', 'complex')
templates = np.zeros((7,input_shape[0], input_shape[1]))
# hom_folder = os.path.join(root, 'new_sample') # Because I am taking it from the call
# file_j = pd.read_json(os.path.join(hom_folder, 'points.txt'), lines=True).set_index('name')
# img_list = pd.read_json(os.path.join(hom_folder, 'points.txt'), lines=True).loc[:, 'name'].values

initializeTemplates(template_folder, templates)

fieldnames, folders = initializeResultsCSV(root)

