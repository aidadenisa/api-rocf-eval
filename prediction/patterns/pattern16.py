import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union

from preprocessing.homography import sharpenDrawing
from prediction.image_processing import draw_contours

#TODO: TRY TO EXTRACT, IT HAS A BIT DIFFERENT SKELETONIZE
def getBackground(external, img, morph=True, ret_hier=False):
    background = np.zeros_like(img)
    points = np.array([external]).reshape((4, 1, 2))
    background = cv2.fillConvexPoly(background, points, (255, 255, 255))
    background = cv2.bitwise_and(img, background)    
    '''overlap = cv2.polylines(cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB), [points], True, (255, 0, 0), 1)
    if internal is not None:
      overlap = cv2.polylines(overlap, [int_points], True, (255, 0, 0), 1)
      plt.imshow(overlap)
      plt.show()'''
    background[background == 0] = 255
    background = sharpenDrawing(background)
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
    cnts, hier = cv2.findContours(background, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if ret_hier:
        return background, cnts, hier
    else:
        return background, cnts

def best_line(backgrounds, idx, only_length, draw=False, drawing=None):
    background = backgrounds[idx]    
    lines_filtered = cv2.HoughLinesP(background, 1, np.pi / 180, 45, None, 60, 50)
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
                #if draw:
                #    drawing = cv2.circle(drawing, (l[2], l[3]), 5, (255, 0, 0), -1)
        #print(points)
        if len(points) > 0:         
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


pad_v = 15
pad_h = 20
pad_move = 30
pad_move_d = 10
dist = int((537 - 219) / 2)

#TODO: TRY to extract it
def diag_eq(diag, y):
  return int(diag[0][0] + ((diag[1][0]-diag[0][0])/(diag[1][1]-diag[0][1]))*(y-diag[0][1]))  

class Pattern16:
  def __init__(self, img, drawing, r_points):
    self.img = img    
    self.drawing = drawing    
    line = [(849, 537 - dist), (r_points[0], 537 - dist)]
    external1 = [(line[0][0] - pad_h, line[0][1] - pad_v), (line[1][0] + pad_h, line[1][1] - pad_v),
                     (line[1][0] + pad_h, line[1][1] + pad_v), (line[0][0] - pad_h, line[0][1] + pad_v)]
    self.externals = []
    for i in range(4):
      self.externals.append(
            [(external1[0][0], external1[0][1] - i * pad_move), (external1[1][0], external1[1][1] - i * pad_move),
              (external1[2][0], external1[2][1] - i * pad_move), (external1[3][0], external1[3][1] - i * pad_move)])
      self.externals.append([(external1[0][0], external1[0][1] - i * pad_move - pad_move_d), (external1[1][0], external1[1][1] - i * pad_move + pad_move_d),
            (external1[2][0], external1[2][1] - i * pad_move + pad_move_d), (external1[3][0], external1[3][1] - i * pad_move - pad_move_d)])
      self.externals.append([(external1[0][0], external1[0][1] - i * pad_move + pad_move_d), (external1[1][0], external1[1][1] - i * pad_move - pad_move_d),
            (external1[2][0], external1[2][1] - i * pad_move - pad_move_d), (external1[3][0], external1[3][1] - i * pad_move + pad_move_d)])
      
      self.externals.append([(external1[0][0], external1[0][1] + i * pad_move), (external1[1][0], external1[1][1] + i * pad_move),
            (external1[2][0], external1[2][1] + i * pad_move), (external1[3][0], external1[3][1] + i * pad_move)])
      self.externals.append([(external1[0][0], external1[0][1] + i * pad_move - pad_move_d), (external1[1][0], external1[1][1] + i * pad_move + pad_move_d),
            (external1[2][0], external1[2][1] + i * pad_move + pad_move_d), (external1[3][0], external1[3][1] + i * pad_move - pad_move_d)])
      self.externals.append([(external1[0][0], external1[0][1] + i * pad_move + pad_move_d), (external1[1][0], external1[1][1] + i * pad_move - pad_move_d),
            (external1[2][0], external1[2][1] + i * pad_move - pad_move_d), (external1[3][0], external1[3][1] + i * pad_move + pad_move_d)])

  def get_score(self, rect, oriz, r_points):
    backgrounds = []
    cnts = []
    rect_h = None
    for external in self.externals:
      background, cnt = getBackground(external, self.img)
      backgrounds.append(background)
      cnts.append(cnt)
    best_diff = np.inf
    best_back = 0
    for background in range(len(backgrounds)):
      ideal_length = np.linalg.norm(np.array(self.externals[background][0]) - np.array(self.externals[background][1]))
      length = best_line(backgrounds, background, only_length=True)
      if length is not None and np.abs(length - ideal_length) < best_diff:
        best_diff = np.abs(length - ideal_length)
        best_back = background
    self.drawing = draw_contours(self.drawing, cnts[best_back])
    result = best_line(backgrounds, best_back, False, True, self.drawing)
    pixel_lines = np.sum(np.divide(backgrounds[best_back], 255))
    if result is not None:
      (max_left, lefty), (max_right, righty), self.drawing = result
      if np.abs(np.rad2deg(np.arctan2(righty - lefty, max_right - max_left))) < 10:
        #print('best inclination: {}'.format(np.abs(np.rad2deg(np.arctan2(righty - lefty, max_right - max_left)))))
        rect_h = np.array([[max_right, righty], [max_left, lefty]])         
    if rect_h is not None:
      self.drawing = cv2.circle(self.drawing, tuple(rect_h[0]), 15, (255, 0, 0), 2)
      self.drawing = cv2.circle(self.drawing, tuple(rect_h[1]), 15, (255, 0, 0), 2)
      lines_points_h = [Point(tuple(rect_h[0])).buffer(15), Point(tuple(rect_h[1])).buffer(15)]
      line_h = LineString([lines_points_h[0].centroid, lines_points_h[1].centroid]).buffer(1.5)
      p1 = None
      p2 = None
      p3 = None
      if rect is not None:
        p1 = lines_points_h[1].intersects(rect[0]) or line_h.intersects(rect[0])
        if not p1:
          print('PATTERN16: vertice sx non tocca rettangolo')
      if oriz is not None:
        oriz_fig = unary_union([Point(oriz[0]).buffer(15), LineString(oriz).buffer(3)])
        p2 = lines_points_h[1].intersects(oriz_fig) or line_h.intersects(oriz_fig)
        if not p2:
          print('PATTERN16: vertice sx non allineato con orizzontale')
      if not ((rect is None or p1) and (oriz is None or p2)):
        label_h_line = 1
        return self.drawing, label_h_line  
      r_points_fig = Point(list(r_points)).buffer(15)
      p3 = line_h.intersects(r_points_fig) or lines_points_h[0].intersects(r_points_fig)
      if not p3:
        print('PATTERN16: vertice dx non tocca punta')
        label_h_line = 2
      else:
        label_h_line = 3      
    else:
      label_h_line = 0
    return self.drawing, label_h_line  