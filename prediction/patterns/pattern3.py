import pandas as pd
import ast
import numpy as np
import cv2
from skimage.morphology import skeletonize
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from prediction.image_processing import thick_rect
from prediction.utils import to_tuple, to_float
from preprocessing import homography

def get_external(line):
  thick = 20
  pad_move = 15
  external = thick_rect(line, thick)
  externals = [external]      
  for i in range(1, 4):
    externals.append(
      [(external[0][0] + i * pad_move, external[0][1]), (external[1][0] + i * pad_move, external[1][1]),
        (external[2][0] + i * pad_move, external[2][1]), (external[3][0] + i * pad_move, external[3][1])])  
    if i == 1:
      externals.append(
      [(external[0][0] - i * pad_move, external[0][1]), (external[1][0] - i * pad_move, external[1][1]),
        (external[2][0] - i * pad_move, external[2][1]), (external[3][0] - i * pad_move, external[3][1])])
  return externals

def int_coverage(lines_filtered, external, drawing=False):
    matrix = np.array(external)    
    base_interval = set(range(min(matrix[:,0]), max(matrix[:,0])))
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
    #if drawing:
    #  print('coverage pattern 3= {}%'.format(coverage))
    return coverage
      
def best_line(backgrounds, idx, only_length, external, draw=False):
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
            if inclination > 20:
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
          coverage = int_coverage(lines_filtered[idx_ok], external)
          if coverage > 50:         
            [vx, vy, x, y] = cv2.fitLine(np.array(points), cv2.DIST_L2, 0, 0.01, 0.01)
            t0 = (max_left-x)/vx
            t1 = (max_right-x)/vx
            lefty = int(y + t0*vy)
            righty = int(y + t1*vy)
            #print((max_left, righty), (max_right, lefty))            
            if only_length:
                return np.linalg.norm(np.array([max_left, lefty]) - np.array([max_right, righty]))
            else:
                return (max_left, lefty), (max_right, righty)
        return None

def get_score_externals(externals, img, threshold=None):
    backgrounds = []
    cnts = []
    rect_or = None
    for external in externals:
      background, cnt = getBackground(external, img, threshold=threshold)
      backgrounds.append(background)
      cnts.append(cnt)
    best_diff = np.inf
    best_back = 0
    for background in range(len(backgrounds)):
      ideal_length = np.linalg.norm(np.array(externals[background][0]) - np.array(externals[background][1]))
      length = best_line(backgrounds, background, True, externals[background])
      if length is not None and np.abs(length - ideal_length) < best_diff:
        best_diff = np.abs(length - ideal_length)
        best_back = background
    #print('best_back: {}'.format(best_back))
    result = best_line(backgrounds, best_back, False, externals[best_back], True)
    pixel_lines = np.sum(np.divide(backgrounds[best_back], 255))
    if result is not None:
      (max_left, lefty), (max_right, righty) = result
      if np.abs(np.rad2deg(np.arctan2(righty - lefty, max_right - max_left))) > 20:
        #print('best inclination: {}'.format(np.abs(np.rad2deg(np.arctan2(righty - lefty, max_right - max_left)))))
        rect_or = np.array([[max_right, righty], [max_left, lefty]])   
    if rect_or is not None:     
      label_diag_line = 3
      return label_diag_line, rect_or       
    else:      
      label_diag_line = 0      
    return label_diag_line, None

def get_diag(bbox, img):
  inclination = 20
  line1 = [bbox[0], bbox[2]]
  line1_sx = [(line1[0][0]-inclination,line1[0][1]), (line1[1][0]+inclination,line1[1][1])]
  line1_dx = [(line1[0][0]+inclination,line1[0][1]), (line1[1][0]-inclination,line1[1][1])]
  line2 = [bbox[1], bbox[3]]
  line2_sx = [(line2[0][0]-inclination,line2[0][1]), (line2[1][0]+inclination,line2[1][1])]
  line2_dx = [(line2[0][0]+inclination,line2[0][1]), (line2[1][0]-inclination,line2[1][1])]
  externals1 = get_external(line1)
  externals1.extend(get_external(line1_sx))
  externals1.extend(get_external(line1_dx))
  externals2 = get_external(line2)
  externals2.extend(get_external(line2_sx))
  externals2.extend(get_external(line2_dx))
  label1, diag1_coord = get_score_externals(externals1, img)
  label2, diag2_coord = get_score_externals(externals2, img)
  if label1==3 and label1==label2:
    return 1, (diag1_coord, diag2_coord)
  else:
    print('manca diag')
  return 0, None


# TODO: TRY TO MAKE THEM RESPECT INTEGRATED AS THE OTHERS IN IMAGE_PROCESSING
def getBackground(external, img, morph=True, ret_hier=False, threshold=None):
    # TODO: FIX WHITE BACKGROUND!
    background = np.zeros_like(img)
    points = np.array([external]).reshape((4, 1, 2))
    background = cv2.fillConvexPoly(background, points, (255, 255, 255))
    background = cv2.bitwise_and(img, background)    
    # background[background == 0] = 255
    # background, t_val = extract_drawing(background, threshold=threshold)
    if threshold > 245:
        background = np.ones_like(img) * 255   
    background = cv2.bitwise_not(background)
    background = skeletonize(background / 255, method='lee').astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    background = cv2.dilate(background, kernel)   
    cnts, hier = cv2.findContours(background, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if ret_hier:
        return background, cnts, hier
    else:
        return background, cnts

# def extract_drawing(image):
#     dst = cv2.bilateralFilter(image, 10, sigmaColor=15, sigmaSpace=15)
#     threshed = np.ones(dst.shape, np.uint8) * 255
#     thresh_val = 0
#     if np.any(dst < 255):
#         hist, _ = np.histogram(dst[dst < 255].flatten(), range(257))
#         thresh_val = homography.maxDeviationThresh(hist)
#         #print(thresh_val)
#         mask = dst < thresh_val
#         threshed[mask] = 0
#     return threshed, thresh_val

#######


class Pattern3:
  def __init__(self, img, drawing, model_diag, scaler_diag, m, s, predictionComplexScores):
    self.img = img    
    self.drawing = drawing
    self.model_diag = model_diag
    self.scaler_diag = scaler_diag
    self.m = m
    self.s = s
    self.predictionComplexScores = predictionComplexScores
  
  def get_score(self, rect, diag1, diag2, oriz, threshold):
    coords = [379, 300, 502, 456]
    # df_rail = pd.read_csv(DLScoresPath, header=0, usecols=['names', 'scores', 'rect'], index_col='names', converters={'scores': to_float, 'rect': to_tuple})

    # TODO: CHECK IF IT'S OKAY TO PUT IT HERE
    rail_bbox = self.predictionComplexScores['rect'][4]

    # TODO: VERIFY EXPLANATION AND REPLACEMENT
    # if self.img_path[:-4] in df_rail.index:
    if self.predictionComplexScores: 
      external = [(rail_bbox[0], rail_bbox[1]), (rail_bbox[0]+rail_bbox[2], rail_bbox[1]), (rail_bbox[0]+rail_bbox[2], rail_bbox[1]+rail_bbox[3]), (rail_bbox[0], rail_bbox[1]+rail_bbox[3])]
      background_rail, _ = getBackground(external, self.img, threshold=threshold)
      pixel_rail = np.sum(np.divide(background_rail, 255))
      #rail_prediction, diags = get_diag(external, self.img)
      rail_prediction = self.model_diag.predict(self.scaler_diag.transform(np.array([pixel_rail]).reshape(-1, 1)))

      # TODO: VERIFY EXPLANATION AND REPLACEMENT -
      # score_rail = self.s.transform(df_rail.loc[self.img_path[:-4], 'scores'][4].reshape(-1,1))
      score_rail = self.s.transform(np.array(self.predictionComplexScores['scores'][4]).reshape(-1,1))
      rail_score = self.m.predict(score_rail)
      p1 = None
      p2 = None         
      if rail_score == 1:
          self.drawing = cv2.rectangle(self.drawing, (rail_bbox[0], rail_bbox[1]), (rail_bbox[0]+rail_bbox[2], rail_bbox[1]+rail_bbox[3]), (0,0,255), 2)
          #d1_fig = unary_union([Point(tuple(diags[0][0])).buffer(20), LineString(diags[0]), Point(tuple(diags[0][1])).buffer(30)])
          #d2_fig = unary_union([Point(tuple(diags[1][0])).buffer(20), LineString(diags[1]), Point(tuple(diags[1][1])).buffer(30)])
          points = [Point(p).buffer(15) for p in external]
          points.extend([Polygon(external).buffer(1.5)])
          bbox_fig = unary_union(points)
          if diag1 is not None and diag2 is not None:
              diag1_fig = LineString(diag1).buffer(1.5)
              diag2_fig = LineString(diag2).buffer(1.5)
              p1 = diag1_fig.intersects(bbox_fig)
              p2 = diag2_fig.intersects(bbox_fig)               
              for v in external:
                  self.drawing = cv2.circle(self.drawing, tuple(v), 15, (255,0,0), 2)
              if not p1 or not p2:
                  print('PATTERN3: diagonali non toccano diagonali')
          if rect is not None:
              p3 = bbox_fig.intersects(rect[0])
              if not p3:
                  print('PATTERN3: diagonali non toccano rettangolo')
          if oriz is not None:
              dist = int((external[3][1] - external[0][1])/4)
              p4 = external[0][1] + dist <= oriz[1][1] <= external[3][1] - dist
              if not p4:
                  print('PATTERN3: mal posizionato orizzontale')
          if (rect is None or p3) and (diag1 is None and diag2 is None or p1 and p2) and (oriz is None or p4):
              label_rail = 3
          else:
              label_rail = 2
      else:    
          if rail_prediction == 1:
            #self.drawing = cv2.rectangle(self.drawing, (rail_bbox[0], rail_bbox[1]), (rail_bbox[0]+rail_bbox[2], rail_bbox[1]+rail_bbox[3]), (255,0,0), 2)
            print('PATTERN3: disegno impreciso')
            label_rail = 1
          else:
            #self.drawing = cv2.rectangle(self.drawing, (rail_bbox[0], rail_bbox[1]), (rail_bbox[0]+rail_bbox[2], rail_bbox[1]+rail_bbox[3]), (0,0,255), 2)
            print('PATTERN3: disegno impreciso e diagonali non trovate')
            label_rail = 0
    else:
      self.drawing = cv2.rectangle(self.drawing, (rail_bbox[0], rail_bbox[1]), (rail_bbox[0]+rail_bbox[2], rail_bbox[1]+rail_bbox[3]), (255,0,0), 2)
      x = coords[0]
      y = coords[1]
      w = np.abs(coords[0] - coords[2])
      h = np.abs(coords[1] - coords[3])
      external = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
      background_rail, _ = getBackground(external, self.img, threshold=threshold)
      pixel_rail = np.sum(np.divide(background_rail, 255))
      rail_prediction = self.model_diag.predict(self.scaler_diag.transform(np.array([pixel_rail]).reshape(-1,1)))
      if rail_prediction == 1:
        label_rail = 1
      else:
        label_rail = 0
    return self.drawing, label_rail


 
