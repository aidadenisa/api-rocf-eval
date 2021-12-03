import numpy as np
import pandas as pd
import os
import joblib 
import ast

from shapely.geometry import Polygon, Point, LineString
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
import cv2

from preprocessing import homography

from prediction.patterns import pattern1
from prediction.patterns import pattern2
from prediction.patterns import pattern3
from prediction.patterns import pattern4
from prediction.patterns import pattern5
from prediction.patterns import pattern6
from prediction.patterns import pattern7
from prediction.patterns import pattern8
from prediction.patterns import pattern9
from prediction.patterns import pattern10
from prediction.patterns import pattern11
from prediction.patterns import pattern12
from prediction.patterns import pattern13
from prediction.patterns import pattern14
from prediction.patterns import pattern15
from prediction.patterns import pattern16
from prediction.patterns import pattern17
from prediction.patterns import pattern18

root = './' 
results_folder = root + 'results/'
models_folder = root + "models/"
results_DL_scores = results_folder + 'scores.csv'
# hom_folder = os.path.join(root, 'new_sample')

#read data about tests from database_patients_complete
labels = pd.read_csv(root + 'templates/database_patients_complete.csv', header=0, index_col=0, delimiter=';')

#save the pattern list 
pattern_list = labels.columns.values[3:-1]
print(pattern_list)


#guesses = []
sys_results = []
result_csv = {}
names = []
count = 0

# img_path = os.path.join('', 'NEW_ROCF_RC375.png')

def label_conv(s):
    if s == 'NORMALI':
        return 0
    if s == 'MCI':
        return 1
    if s == 'DEMENZA':
        return 2

def find_line(image, points, predictionComplexScores):

    # identify the 5 points of interest the homogram 
    # points = np.array(file_homog.loc[img_path[:-4]].to_numpy()[0])
    points = np.array(points)
    if points.shape == (1,):
        points = np.array(points[0])
    #transform points from a matrix into an array of tuples
    points = [tuple(x) for x in points]

    #compute the homography for the point corresponding to the rhomb
    r_points = homography.computeHomographyRhomb(points)

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # transform the black zones resulted from the homography transformation in the color of the paper
    img = homography.unique_color(grayscale)
    # img = homography.unique_color(cv2.imread(os.path.join(hom_folder, img_path), cv2.IMREAD_GRAYSCALE))
    #img = cv2.blur(img, (3, 3))

    #initialize the drawing as an array of 0s, in the shape of the image + 3 
    drawing = np.zeros((img.shape[0], img.shape[1], 3))
    #initialize the score result with 0s
    results = np.zeros(18)

    # [STUDIED] Generate score for pattern 2 (mostly computer vision, + SVM prediction)
    pat2 = pattern2.Pattern2(img, drawing, joblib.load(models_folder + 'rectangle_model.joblib'), joblib.load(models_folder + 'rectangle_scaler.joblib'))
    # get the drawing that contains the idenitfied lines and the polygon, with the circles drawn in its corners
    # get the resulted score
    # return the figure as a tuple: first is the polygon shape, and the second is the corners (vertices)
    drawing, results[1], ret_fig = pat2.get_score()
    # print(ret_fig)

   
    #Generate score for pattern 1 [ANALYTICAL] [COMPUTER VISION]
    # img - initial image; drawing - generated drawing after the initia image
    pat1 = pattern1.Pattern1(img, drawing)    
    # TODO: explain what is returned
    drawing, results[0], diag1, diag2 = pat1.get_score(ret_fig)

    pat6 = pattern6.Pattern6(img, drawing)
    drawing, results[5], oriz_coord = pat6.get_score(ret_fig, diag1, diag2)    

    pat3 = pattern3.Pattern3(img, drawing, joblib.load(models_folder + 'rett_diag_model.joblib'), joblib.load(models_folder + 'rett_diag_scaler.joblib'), joblib.load(models_folder + 'rett_diag_score_model.joblib'), joblib.load(models_folder + 'rett_diag_score_scaler.joblib'), predictionComplexScores)
    drawing, results[2] = pat3.get_score(ret_fig, diag1, diag2, oriz_coord)      
    
    pat5 = pattern5.Pattern5(img, drawing, joblib.load(models_folder + 'cross_model.joblib'), joblib.load(models_folder + 'cross_scaler.joblib'), joblib.load(models_folder + 'cross_score_model.joblib'), joblib.load(models_folder + 'cross_score_scaler.joblib'), predictionComplexScores)
    drawing, results[4] = pat5.get_score()   
    
    pat4 = pattern4.Pattern4(img, drawing, diag1, oriz_coord)
    drawing, results[3] = pat4.get_score(diag1, ret_fig) 
    
    pat7 = pattern7.Pattern7(img, drawing)
    drawing, results[6], vert = pat7.get_score(ret_fig, diag1, diag2)
    
    pat8 = pattern8.Pattern8(img, drawing, diag1, diag2, vert, oriz_coord)
    drawing, results[7] = pat8.get_score() 
    
    pat9 = pattern9.Pattern9(img, drawing, vert)
    drawing, results[8] = pat9.get_score(ret_fig, vert)

    
    pat10 = pattern10.Pattern10(img, drawing, joblib.load(models_folder + 'face_model.joblib'), joblib.load(models_folder + 'face_scaler.joblib'), joblib.load(models_folder + 'face_score_model.joblib'), joblib.load(models_folder + 'face_score_scaler.joblib'), predictionComplexScores)
    drawing, results[9] = pat10.get_score(diag2, oriz_coord)
    
    pat11 = pattern11.Pattern11(img, drawing, vert, diag2)
    drawing, results[10] = pat11.get_score(ret_fig, diag1, diag2)  
     
    pat12 = pattern12.Pattern12(img, drawing, joblib.load(models_folder + 'rail_model.joblib'), joblib.load(models_folder + 'rail_scaler.joblib'), joblib.load(models_folder + 'rail_score_model.joblib'), joblib.load(models_folder + 'rail_score_scaler.joblib'), predictionComplexScores)
    drawing, results[11] = pat12.get_score()
    
    pat13 = pattern13.Pattern13(img, drawing, r_points, ret_fig)
    drawing, results[12] = pat13.get_score() 
       
    pat14 = pattern14.Pattern14(img, drawing, r_points)
    drawing, results[13], rhomb_fig = pat14.get_score(r_points, diag1, diag2)
       
    pat15 = pattern15.Pattern15(img, drawing, joblib.load(models_folder + 'rect_model.joblib'), joblib.load(models_folder + 'rect_scaler.joblib'), joblib.load(models_folder + 'rect_score_model.joblib'), joblib.load(models_folder + 'rect_score_scaler.joblib'), predictionComplexScores)
    drawing, results[14] = pat15.get_score(ret_fig)
    
    pat16 = pattern16.Pattern16(img, drawing, r_points)
    drawing, results[15] = pat16.get_score(ret_fig, oriz_coord, r_points)

    pat17 = pattern17.Pattern17(img, drawing, joblib.load(models_folder + 'cross_vert_model.joblib'), joblib.load(models_folder + 'cross_vert_scaler.joblib'), joblib.load(models_folder + 'cross_vert_score_model.joblib'), joblib.load(models_folder + 'cross_vert_score_scaler.joblib'), predictionComplexScores)
    drawing, results[16] = pat17.get_score()

    pat18 = pattern18.Pattern18(img, drawing, joblib.load(models_folder + 'triang_model.joblib'), joblib.load(models_folder + 'triang_scaler.joblib'), joblib.load(models_folder + 'triang_score_model.joblib'), joblib.load(models_folder + 'triang_score_scaler.joblib'), predictionComplexScores)
    drawing, results[17] = pat18.get_score()
    # plt.imshow(drawing)
    # plt.show()    
    

    # convert image from grayscale to RGB
    overlap = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # extract from a generated image that contains true (1, white) for all the pixels in the drawing that is different than 0 (black), the pixels that follow the mask that is true for any pixel from the drawing that is different than green (which is usually the contour of the drawing)  
    drawing_mask = np.bitwise_and(np.any(drawing != [0, 0, 0], axis=-1), np.any(drawing != [0, 255, 0], axis=-1))
    # on the colored version of the image, put on the pixels saved on the drawing mask the pixels taken from the generated drawing that contains the analysis and patterns detected
    overlap[drawing_mask] = drawing[drawing_mask]

    # plt.imshow(overlap)
    # plt.show()    

    #return the results list from the analysis of all the patterns
    return results.astype(np.uint8)


def predictScores(image, points, predictionComplexScores):

    name = "newImage"

    # identify the patterns in the image
    results = find_line(image, points, predictionComplexScores)

    df = pd.DataFrame([results], columns=pattern_list)

    df["name"] = name

    is_header = True

    if is_header: 
        df.to_csv(results_folder + 'total_scores.csv', header = True, index=False)
        is_header = False
    else:
        df.to_csv(results_folder + 'total_scores.csv', mode = 'a', header = False, index=False)

    return results

 
