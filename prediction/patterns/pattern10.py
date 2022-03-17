import pandas as pd
import ast
import numpy as np
import cv2
from skimage.morphology import skeletonize
from shapely.geometry import LineString, Polygon

from preprocessing.homography import maxDeviationThresh, sharpenDrawing
from preprocessing.thresholding import getSplitsFromROI
# from prediction.image_processing import getBackground

# TODO: A DIFFERENCE ON SKELETONIZE method='lee', try to extract it later
def getBackground(external, img, morph=True, ret_hier=False, internal=None):
    background = np.zeros_like(img)
    points = np.array([external]).reshape((4, 1, 2))
    background = cv2.fillConvexPoly(background, points, (255, 255, 255))
    not_background = cv2.bitwise_not(background)
    background = cv2.bitwise_and(img, background)
    if internal is not None:
      int_points = np.array([internal]).reshape((4, 1, 2))
      background = cv2.fillConvexPoly(background, int_points, (255, 255, 255))
    '''overlap = cv2.polylines(cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB), [points], True, (255, 0, 0), 1)
    if internal is not None:
      overlap = cv2.polylines(overlap, [int_points], True, (255, 0, 0), 1)
      plt.imshow(overlap)
      plt.show()'''
    # background[background == 0] = 255
    # background = sharpenDrawing(background)
    background = cv2.bitwise_or(not_background,background)
    if morph:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        # background = cv2.bitwise_not(background)
        background = cv2.bitwise_not(cv2.erode(background, kernel))
        background = skeletonize(background / 255, method='lee').astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        background = cv2.dilate(background, kernel)
    else:
        background = cv2.bitwise_not(background)
        background = skeletonize(background / 255, method='lee').astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        background = cv2.dilate(background, kernel)
    # if internal is not None:    
    #   plt.imshow(background, cmap='gray')
    #   plt.show()
    cnts, hier = cv2.findContours(background, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if ret_hier:
        return background, cnts, hier
    else:
        return background, cnts

class Pattern10:
  def __init__(self, img, drawing, model_diag, scaler_diag, m, s, predictionComplexScores):
    self.img = img    
    self.drawing = drawing
    self.model_diag = model_diag
    self.scaler_diag = scaler_diag
    self.m = m
    self.s = s
    self.predictionComplexScores = predictionComplexScores
    self.roi = []
  
  def get_score(self, diag, oriz):
    coords = [742, 287, 829, 373]
    if self.predictionComplexScores:
      rail_bbox = self.predictionComplexScores['rect'][1]
      external = [(rail_bbox[0], rail_bbox[1]), (rail_bbox[0]+rail_bbox[2], rail_bbox[1]), (rail_bbox[0]+rail_bbox[2], rail_bbox[1]+rail_bbox[3]), (rail_bbox[0], rail_bbox[1]+rail_bbox[3])]
      
      embeddingsWithoutAnchor = self.predictionComplexScores['embeddings'][1][:1024]
      embeddings_scaled = self.s.transform(np.array([embeddingsWithoutAnchor])) 
      embeddings_prediction = self.m.predict(embeddings_scaled)
      
      if embeddings_prediction == 1:
        self.drawing = cv2.rectangle(self.drawing, (rail_bbox[0], rail_bbox[1]),
                                        (rail_bbox[0] + rail_bbox[2], rail_bbox[1] + rail_bbox[3]), (255, 0, 0), 2)
        bbox_fig = Polygon(external)
        if diag is not None:
          diag_fig = LineString(diag)
        else:
          diag_fig = LineString([[852, 219],[382, 537]])
        splitted = diag_fig.intersection(bbox_fig)
        if not splitted.is_empty:
          split_coord = splitted.xy  
          int_points = np.array(list(zip(split_coord[0], split_coord[1]))).astype(int)
        p1 = splitted.is_empty or max(int_points[:, 0]) < rail_bbox[0] + 2*rail_bbox[2]/3 and max(int_points[:,1]) < rail_bbox[1] + 2*rail_bbox[3]/3 
        if not p1:
          print('PATTERN10: diagonale mal posizionata')
        if oriz is not None:
          oriz_fig = LineString(oriz)
        else:
          dist = int((537 - 219) / 2)
          oriz_fig = LineString([[382, 537 - dist], [852, 537 - dist]])
        splitted = oriz_fig.intersection(bbox_fig)
        if not splitted.is_empty:
            split_coord = splitted.xy
            int_points = np.array(list(zip(split_coord[0], split_coord[1]))).astype(int)
        p2 = splitted.is_empty or min(int_points[:, 1]) > rail_bbox[1] + 2*rail_bbox[3]/3 
        if not p2:
            print('PATTERN10: orizzontale mal posizionata')
        if p1 and p2:
          label_rail = 3
        else:
            label_rail = 2
      
      else:
        roiSplit = getSplitsFromROI(external)
        roiPixels = []
        for roi in range(len(roiSplit)):
            background, _ = getBackground(roiSplit[roi], self.img)
            pixel_value= np.sum(np.divide(background, 255))
            roiPixels.append(pixel_value)

        pixel_prediction = self.model_diag.predict(self.scaler_diag.transform(np.array(roiPixels).reshape(1, -1)))

        if pixel_prediction == 1:
          self.drawing = cv2.rectangle(self.drawing, (rail_bbox[0], rail_bbox[1]), (rail_bbox[0]+rail_bbox[2], rail_bbox[1]+rail_bbox[3]), (0,0,255), 2)
          print('PATTERN10: disegno impreciso')
          label_rail = 1
        else:
          self.drawing = cv2.rectangle(self.drawing, (rail_bbox[0], rail_bbox[1]), (rail_bbox[0]+rail_bbox[2], rail_bbox[1]+rail_bbox[3]), (0,0,255), 2)
          print('PATTERN10: disegno assente')
          label_rail = 0
    else:
      x = coords[0] 
      y = coords[1] 
      w = np.abs(coords[0] - coords[2])
      h = np.abs(coords[1] - coords[3])
      external = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
      roiSplit = getSplitsFromROI(external)
      roiPixels = []
      for roi in range(len(roiSplit)):
          background, _ = getBackground(roiSplit[roi], self.img)
          pixel_value= np.sum(np.divide(background, 255))
          roiPixels.append(pixel_value)

      pixel_prediction = self.model_diag.predict(self.scaler_diag.transform(np.array(roiPixels).reshape(1, -1)))

      if pixel_prediction == 1:
        label_rail = 1
      else:
        label_rail = 0
    self.roi = [[[p[0], p[1]] for p in external]]
    return self.drawing, label_rail
  
  def get_ROI(self):
    return self.roi