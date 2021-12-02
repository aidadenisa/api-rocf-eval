import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from shapely.geometry import Polygon, Point, LineString
from matplotlib import pyplot as plt

from prediction.image_processing import getBackground, draw_contours
# from preprocessing.homography import maxDeviationThresh, sharpenDrawing

def best_line(backgrounds, idx, only_length, external, h=True, draw=False, drawing=None):
    background = backgrounds[idx]    
    lines_filtered = cv2.HoughLinesP(background, 1, np.pi / 180, 50, None, 20, 10)
    idx_ok = []
    if lines_filtered is not None:
        max_left = np.inf
        max_right = -np.inf
        points = []
        for i in range(0, len(lines_filtered)):
            l = lines_filtered[i][0]
            inclination = np.abs(np.rad2deg(np.arctan2(l[3] - l[1], l[2] - l[0])))
            if inclination > 60:
              points.append((l[0], l[1]))
              if l[1] < max_left:
                  max_left = l[1]
              if l[1] > max_right:
                  max_right = l[1]
              #if draw:
              #    drawing = cv2.circle(drawing, (l[0], l[1]), 5, (255, 0, 0), -1)
              points.append((l[2], l[3]))
              if l[3] < max_left:
                  max_left = l[3]
              if l[3] > max_right:
                  max_right = l[3]
              idx_ok.append(i)
              #if draw:
              #    drawing = cv2.circle(drawing, (l[2], l[3]), 5, (255, 0, 0), -1)
        
        if len(points) > 0:
          coverage = int_coverage(lines_filtered[idx_ok], external)
          if coverage > 60:
            [vx, vy, x, y] = cv2.fitLine(np.array(points), cv2.DIST_L12, 0, 0.01, 0.01)
            t0 = (max_left-y)/vy
            t1 = (max_right-y)/vy
            lefty = int(x + t0*vx)
            righty = int(x + t1*vx)    
            
                
            #print('line length = {}'.format(np.linalg.norm(np.array([max_left, righty]) - np.array([max_right, lefty]))))
            #print('inclination = {}'.format(np.rad2deg(np.arctan2(max_right - max_left, righty - lefty))))
            if only_length:
                return np.linalg.norm(np.array([lefty, max_left]) - np.array([righty, max_right]))
            else:
                return (lefty, max_left), (righty, max_right), drawing
        return None

def int_coverage(lines_filtered, external, drawing=False):
    matrix = np.array(external)    
    base_interval = set(range(min(matrix[:,1]), max(matrix[:,1])))
    point_int = []       
    for i in range(0, len(lines_filtered)):
        l = lines_filtered[i][0]
        if l[3] > l[1]:
          point_int.append(range(l[1], l[3]))
        else:
          point_int.append(range(l[3], l[1]))
    union_set = set().union(*point_int)
    inter = base_interval.intersection(union_set)
    coverage = (len(inter) / len(base_interval)) * 100
    #if drawing:
    #  print('coverage pattern 3= {}%'.format(coverage))
    return coverage


pad_v = 15
pad_h = 15
pad_move = 30
pad_move_d = 10

  
def diag_eq(diag, x):
    return int(diag[0][1] + ((diag[1][1]-diag[0][1])/(diag[1][0]-diag[0][0]))*(x-diag[0][0]))
  
class Pattern11:
  def __init__(self, img, drawing, vert, diag):
    self.img = img    
    self.drawing = drawing
    if vert is not None:
      vert_limit = max(vert[:, 0]) + 25
    else:
      dist = int((852 - 382) / 2)
      vert_limit = 382 + dist + 25
    if diag is not None:
      diagonale = diag
    else:
      diagonale = [(852, 219), (382, 537)]
    line = [(vert_limit, 219), (vert_limit, diag_eq(diagonale, vert_limit))]
    external1 = [(line[0][0] - pad_h, line[0][1] - pad_v), (line[0][0] + pad_h, line[0][1] - pad_v),
             (line[1][0] + pad_h, diag_eq(diagonale, line[1][0] + pad_h) + pad_v), (line[1][0] - pad_h, diag_eq(diagonale, line[1][0] - pad_h) + pad_v)]
    self.externals = []
    i = 0
    while external1[1][0] + i * pad_move < 852 - 10:      
      self.externals.append([(external1[0][0] + i * pad_move, external1[0][1]), (external1[1][0] + i * pad_move, external1[1][1]),
            (external1[2][0] + i * pad_move, diag_eq(diagonale, external1[2][0] + i * pad_move) + pad_v), 
            (external1[3][0] + i * pad_move, diag_eq(diagonale, external1[3][0] + i * pad_move) + pad_v)])
      self.externals.append([(external1[0][0] + i * pad_move - pad_move_d, external1[0][1]), (external1[1][0] + i * pad_move - pad_move_d, external1[1][1]),
            (external1[2][0]+ i * pad_move + pad_move_d, diag_eq(diagonale, external1[2][0]+ i * pad_move + pad_move_d) + pad_v), 
            (external1[3][0] + i * pad_move + pad_move_d, diag_eq(diagonale, external1[3][0] + i * pad_move + pad_move_d) + pad_v)])
      self.externals.append([(external1[0][0] + i * pad_move + pad_move_d, external1[0][1]), (external1[1][0] + i * pad_move + pad_move_d, external1[1][1]),
            (external1[2][0] + i * pad_move - pad_move_d, diag_eq(diagonale, external1[2][0] + i * pad_move - pad_move_d) + pad_v), 
            (external1[3][0] + i * pad_move - pad_move_d, diag_eq(diagonale, external1[3][0] + i * pad_move - pad_move_d) + pad_v)])
      i += 1



  def count_line(self, externals):    
    lines_found = []
    ex_idx = 0
    while ex_idx < len(externals):
      background, cnt = getBackground(externals[ex_idx], self.img, morph=False)
      result = best_line([background], 0, False, externals[ex_idx])
      if result is not None:
        (lefty, max_left), (righty, max_right), drawing = result
        if len(lines_found) > 0:
            line_before = lines_found[-1]
            if abs(lefty - line_before[0][0]) >= 10 and abs(righty - line_before[1][0]) >= 10:
                lines_found.append([(lefty, max_left), (righty, max_right)])
                self.drawing = cv2.line(self.drawing, (lefty, max_left), (righty, max_right), (0, 0, 255), 2, cv2.LINE_AA)
        else:
            lines_found.append([(lefty, max_left), (righty, max_right)])
            self.drawing = cv2.line(self.drawing, (lefty, max_left), (righty, max_right), (0, 0, 255), 2, cv2.LINE_AA)
        ex_idx += 1
        while ex_idx % 3 != 0:
            ex_idx +=1
      else:
        ex_idx += 1
    return np.array(lines_found)

  def get_score(self, rect, diag1, diag2):
    lines_found = self.count_line(self.externals)
    rect_or = None    
    if lines_found.shape[0] >= 1:    
      (lefty, max_left), (righty, max_right) = lines_found[0]             
      if np.abs(np.rad2deg(np.arctan2(max_right - max_left, righty - lefty))) > 60:
        #print('best inclination: {}'.format(np.abs(np.rad2deg(np.arctan2(righty - lefty, max_right - max_left)))))
        rect_or = np.array([[lefty, max_left], [righty, max_right]])  
    if rect_or is not None:
      self.drawing = cv2.circle(self.drawing, tuple(rect_or[0]), 15, (255, 0, 0), 2)
      self.drawing = cv2.circle(self.drawing, tuple(rect_or[1]), 15, (255, 0, 0), 2)
      if lines_found.shape[0] == 1:
        lines_point_or = [Point(tuple(rect_or[0])).buffer(15), Point(tuple(rect_or[1])).buffer(15)]
        line_or = LineString([lines_point_or[0].centroid, lines_point_or[1].centroid]).buffer(1.5)
        p1 = None
        p2 = None
        if rect is not None:
          p1 = lines_point_or[0].intersects(rect[0]) or line_or.intersects(rect[0])
          if not p1:
            print('PATTERN11: vertice in alto non tocca rettangolo')
        if diag2 is not None:
          diag_fig = LineString(diag2).buffer(1.5)
          p2 = lines_point_or[1].intersects(diag_fig) or line_or.intersects(diag_fig)
          if not p2:
            print('PATTERN11: vertice in basso non tocca diagonale')
        if not ((rect is None or p1) and (diag2 is None or p2)):
          label_or_line = 1
          return self.drawing, label_or_line
        if diag1 is not None and diag2 is not None:
          line1 = LineString(diag1)
          line2 = LineString(diag2)
        else:
          line1 = LineString([(382, 219), (852, 537)])
          line2 = LineString([(852, 219), (382, 537)])
        int_pt = line1.intersection(line2)
        point_of_intersection = Point((int_pt.x, int_pt.y)).buffer(30)     
        p3 = not point_of_intersection.intersects(line_or)
        if p3:
          label_or_line = 3
        else:
          print('PATTERN11: linea mal posizionata')
          label_or_line = 2
      else:
        print('PATTERN11: numero linee sbagliato')
        label_or_line = 1  
    else:
      print('PATTERN11: nessuna linea trovata')      
      label_or_line = 0
    return self.drawing, label_or_line  