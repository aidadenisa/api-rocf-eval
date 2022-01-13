import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from shapely.geometry import Polygon, Point, LineString
from shapely.geometry.base import CAP_STYLE


from preprocessing import homography
from prediction import image_processing as imgProcess

def int_coverage(lines_filtered, drawing=None):
    base_interval = set(range(382-20, 852+20))
    point_int = []    
    for i in range(0, len(lines_filtered)):
        l = lines_filtered[i][0]        
        if l[2] > l[0]:
          point_int.append(range(l[0], l[2]))
        else:
          point_int.append(range(l[2], l[0]))
    union_set = set().union(*point_int)
    inter = base_interval.intersection(union_set)
    coverage = (len(inter) / len(base_interval)) * 100
    #if drawing is not None:
    #  print('coverage = {}%'.format(coverage))
    return coverage

def best_line(backgrounds, idx, only_length, h=True, draw=False, drawing=None):
    background = backgrounds[idx]    
    lines_filtered = cv2.HoughLinesP(background, 1, np.pi / 180, 75, None, 40, 20)
    idx_ok = []
    if lines_filtered is not None:
        max_left = np.inf
        max_right = -np.inf
        points = []
        for i in range(0, len(lines_filtered)):
            l = lines_filtered[i][0]
            inclination = np.abs(np.rad2deg(np.arctan2(l[3] - l[1], l[2] - l[0])))
            if inclination < 10:
                points.append((l[0], l[1]))
                if l[0] < max_left:
                    max_left = l[0]
                if l[0] > max_right:
                    max_right = l[0]
                #if draw:
                #    drawing = cv2.circle(drawing, (l[0], l[1]), 5, (255, 0, 0), -1)
                points.append((l[2], l[3]))
                if l[2] < max_left:
                    max_left = l[2]
                if l[2] > max_right:
                    max_right = l[2]
                idx_ok.append(i)
                #if draw:
                #    drawing = cv2.circle(drawing, (l[2], l[3]), 5, (255, 0, 0), -1)
        #print(points)
        if len(points) > 0:          
          coverage =  int_coverage(lines_filtered[idx_ok])            
          if coverage > 80:
            [vx, vy, x, y] = cv2.fitLine(np.array(points), cv2.DIST_L12, 0, 0.01, 0.01)
            t0 = (max_left-x)/vx
            t1 = (max_right-x)/vx
            lefty = int(y + t0*vy)
            righty = int(y + t1*vy)
            #print((max_left, righty), (max_right, lefty))
            if draw and np.abs(np.rad2deg(np.arctan2(righty - lefty, max_right - max_left))) < 10:
                drawing = cv2.line(drawing, (max_left, lefty), (max_right, righty), (0, 0, 255), 2, cv2.LINE_AA)
            #print('line length = {}'.format(np.linalg.norm(np.array([max_left, righty]) - np.array([max_right, lefty]))))
            #print('inclination = {}'.format(np.rad2deg(np.arctan2(righty - lefty, max_right - max_left))))
            if only_length:
                return np.linalg.norm(np.array([max_left, lefty]) - np.array([max_right, righty]))
            else:
                return (max_left, lefty), (max_right, righty), drawing
        return None

def buildROIs(line): 
  rois = []
  lineObj = LineString(line)
  buffer = lineObj.buffer(30, cap_style=CAP_STYLE.square)
  # simplified = buffer.simplify(tolerance=0.95, preserve_topology=True)
  coords = np.array(list(buffer.exterior.coords)).astype(int)
  rois.append(coords.tolist())
  
  return rois

pad_v = 15
pad_h = 20
pad_move = 15
pad_move_d = 10
dist = int((537 - 219) / 2)
line = [(382, 537 - dist), (852, 537 - dist)]
external1 = [(line[0][0] - pad_h, line[0][1] - pad_v), (line[1][0] + pad_h, line[1][1] - pad_v),
                     (line[1][0] + pad_h, line[1][1] + pad_v), (line[0][0] - pad_h, line[0][1] + pad_v)]
externals = []
for i in range(4):
  externals.append(
        [(external1[0][0], external1[0][1] - i * pad_move), (external1[1][0], external1[1][1] - i * pad_move),
          (external1[2][0], external1[2][1] - i * pad_move), (external1[3][0], external1[3][1] - i * pad_move)])
  externals.append([(external1[0][0], external1[0][1] - i * pad_move - pad_move_d), (external1[1][0], external1[1][1] - i * pad_move + pad_move_d),
        (external1[2][0], external1[2][1] - i * pad_move + pad_move_d), (external1[3][0], external1[3][1] - i * pad_move - pad_move_d)])
  externals.append([(external1[0][0], external1[0][1] - i * pad_move + pad_move_d), (external1[1][0], external1[1][1] - i * pad_move - pad_move_d),
        (external1[2][0], external1[2][1] - i * pad_move - pad_move_d), (external1[3][0], external1[3][1] - i * pad_move + pad_move_d)])
  
  externals.append([(external1[0][0], external1[0][1] + i * pad_move), (external1[1][0], external1[1][1] + i * pad_move),
        (external1[2][0], external1[2][1] + i * pad_move), (external1[3][0], external1[3][1] + i * pad_move)])
  externals.append([(external1[0][0], external1[0][1] + i * pad_move - pad_move_d), (external1[1][0], external1[1][1] + i * pad_move + pad_move_d),
        (external1[2][0], external1[2][1] + i * pad_move + pad_move_d), (external1[3][0], external1[3][1] + i * pad_move - pad_move_d)])
  externals.append([(external1[0][0], external1[0][1] + i * pad_move + pad_move_d), (external1[1][0], external1[1][1] + i * pad_move - pad_move_d),
        (external1[2][0], external1[2][1] + i * pad_move - pad_move_d), (external1[3][0], external1[3][1] + i * pad_move + pad_move_d)])
  
class Pattern6:
  def __init__(self, img, drawing):
    self.img = img    
    self.drawing = drawing
    self.roi = []

  def get_score(self, rect, diag1, diag2):
    backgrounds = []
    cnts = []
    rect_or = None
    for external in externals:
      background, cnt = imgProcess.getBackground(external, self.img)
      backgrounds.append(background)
      cnts.append(cnt)
    best_diff = np.inf
    best_back = 0
    for background in range(len(backgrounds)):
      ideal_length = np.linalg.norm(np.array(externals[background][0]) - np.array(externals[background][1]))
      length = best_line(backgrounds, background, only_length=True, h=False)
      if length is not None and np.abs(length - ideal_length) < best_diff:
        best_diff = np.abs(length - ideal_length)
        best_back = background
    self.drawing = imgProcess.draw_contours(self.drawing, cnts[best_back])
    result = best_line(backgrounds, best_back, False, False, True, self.drawing)
    pixel_lines = np.sum(np.divide(backgrounds[best_back], 255))
    if result is not None:
      (max_left, lefty), (max_right, righty), drawing = result
      if np.abs(np.rad2deg(np.arctan2(righty - lefty, max_right - max_left))) < 10:
        #print('best inclination: {}'.format(np.abs(np.rad2deg(np.arctan2(righty - lefty, max_right - max_left)))))
        rect_or = np.array([[max_right, righty], [max_left, lefty]])
    if rect_or is not None:
      self.roi = buildROIs(rect_or)
      p1 = None
      p2 = None
      p3 = None
      p4 = None
      self.drawing = cv2.circle(self.drawing, tuple(rect_or[0]), 15, (255, 0, 0), 2)
      self.drawing = cv2.circle(self.drawing, tuple(rect_or[1]), 15, (255, 0, 0), 2)
      lines_points_or = [Point(tuple(rect_or[0])).buffer(15), Point(tuple(rect_or[1])).buffer(15)]
      line_or = LineString([lines_points_or[0].centroid, lines_points_or[1].centroid]).buffer(1.5)
      if rect is not None:
        p1 = lines_points_or[0].intersects(rect[0])
        if not p1:
          print('PATTERN6: vertice dx non tocca rettangolo')
        p2 = lines_points_or[1].intersects(rect[0])
        if not p2:
          print('PATTERN6: vertice sx non tocca rettangolo')
        p4 = line_or.overlaps(rect[0])
      if not (rect is None or (p1 and p2 or p4)):
        label_or_line = 1
        return self.drawing, label_or_line, rect_or
      if diag1 is not None and diag2 is not None:
        line1 = LineString(diag1)
        line2 = LineString(diag2)
      else:
        line1 = LineString([(382, 219), (852, 537)])
        line2 = LineString([(852, 219), (382, 537)])
      int_pt = line1.intersection(line2)
      point_of_intersection = Point((int_pt.x, int_pt.y)).buffer(35)
     
      p3 = point_of_intersection.intersects(line_or)
      if not p3:
        print('PATTERN6: non interseca diagonali')  
        self.drawing = cv2.circle(self.drawing, (int(int_pt.x), int(int_pt.y)), 35, (255,0 ,0), 2)    
        label_or_line = 2        
      else:
        label_or_line = 3
      return self.drawing, label_or_line, rect_or
    else:     
      label_or_line = 0
    return self.drawing, label_or_line, None
  
  def get_ROI(self):
    return self.roi