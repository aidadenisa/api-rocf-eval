import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union


from preprocessing.homography import unique_color, maxDeviationThresh
from prediction.image_processing import draw_contours

# #TODO: TRY TO EXTRACT IT, IT HAS A THRESH_VALUE returned
# def extract_drawing(image):
#     dst = cv2.bilateralFilter(image, 10, sigmaColor=15, sigmaSpace=15)
#     # dst = img.copy()
#     # max_occ = np.bincount(dst[dst > 0]).argmax()
#     # dst[dst == 0] = max_occ
#     threshed = np.ones(dst.shape, np.uint8) * 255
#     thresh_val = 0
#     if np.any(dst < 255):
#         hist, _ = np.histogram(dst[dst < 255].flatten(), range(257))
#         thresh_val = maxDeviationThresh(hist)
#         #print(thresh_val)
#         mask = dst < thresh_val
#         threshed[mask] = 0
#     return threshed, thresh_val

#TODO: CANNOT BE EXTRACTED EASILY, IT HAS THE THRESH VALUE
def getBackground(external, img, morph=True, ret_hier=False, internal=None, threshold=None):
    points = np.array(external)
    interval = (max(points[:,1])-min(points[:,1]), max(points[:,0])-min(points[:,0]))
    points_scaled = points.copy()
    points_scaled[:, 0] -= min(points[:, 0])
    points_scaled[:, 1] -= min(points[:, 1])
    background_t = np.zeros(interval, dtype=np.uint8)
    background_t = cv2.fillConvexPoly(background_t, points_scaled.reshape((4, 1, 2)), 255)
    not_background_t = cv2.bitwise_not(background_t)
    image_interval = img[min(points[:,1]):max(points[:,1]), min(points[:,0]):max(points[:,0])]
    background_t = cv2.bitwise_and(image_interval, background_t)
    
    # background_t = unique_color(background_t)
    # background_t, t_val = extract_drawing(background_t)

    background_t = cv2.bitwise_or(not_background_t,background_t)
    if threshold > 246:
        background_t = np.ones(interval, dtype=np.uint8) * 255
    background = np.ones_like(img) * 255
    background[min(points[:,1]):max(points[:,1]), min(points[:,0]):max(points[:,0])] = background_t
    if morph:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        # background = cv2.bitwise_not(background)
        background = cv2.bitwise_not(cv2.erode(background, kernel))
        background = skeletonize(background / 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        background = cv2.dilate(background, kernel)
    else:
        background = cv2.bitwise_not(background)
        background = skeletonize(background / 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        background = cv2.dilate(background, kernel)
   
    cnts, hier = cv2.findContours(background, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if ret_hier:
        return background, cnts, hier
    else:
        return background, cnts

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
            if inclination > 80:
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
          if coverage > 50:            
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

#TODO: TRY TO EXTRACT
def diag_eq(diag, x):
    return int(diag[0][1] + ((diag[1][1]-diag[0][1])/(diag[1][0]-diag[0][0]))*(x-diag[0][0]))
 
class Pattern13:
  def __init__(self, img, drawing, r_points, rect):
    self.img = img    
    self.drawing = drawing
    if rect is None:
        line1 = [(852, 219), r_points]
        line2 = [(852, 537), r_points]
        line = [(852 + 30, diag_eq(line1, 852 + 30)), (852 + 30, diag_eq(line2, 852 + 30))]
    else:
        line1 = [(int(rect[1][1].centroid.x), int(rect[1][1].centroid.y)), r_points]
        line2 = [(int(rect[1][2].centroid.x), int(rect[1][2].centroid.y)), r_points]
        line = [(line1[0][0]+30, diag_eq(line1, line1[0][0]+30)), (line2[0][0]+30, diag_eq(line2, line2[0][0]+30))]
    external1 = [(line[0][0] - pad_h, diag_eq(line1, line[0][0] - pad_h) - pad_v), (line[0][0] + pad_h, diag_eq(line1, line[0][0] + pad_h) - pad_v),
             (line[1][0] + pad_h, diag_eq(line2, line[1][0] + pad_h) + pad_v), (line[1][0] - pad_h, diag_eq(line2, line[1][0] - pad_h) + pad_v)]
    self.externals = []
    i = 0
    while external1[2][0] + i * pad_move < r_points[0] - 30:      
      self.externals.append([(external1[0][0] + i * pad_move, diag_eq(line1, external1[0][0] + i * pad_move)-pad_v), (external1[1][0] + i * pad_move,  diag_eq(line1, external1[1][0] + i * pad_move)-pad_v),
            (external1[2][0] + i * pad_move, diag_eq(line2,external1[2][0] + i * pad_move)+ pad_v), (external1[3][0] + i * pad_move, diag_eq(line2,external1[3][0] + i * pad_move)+ pad_v)])
      self.externals.append([(external1[0][0] + i * pad_move - pad_move_d, diag_eq(line1, external1[0][0] + i * pad_move - pad_move_d)-pad_v), 
                             (external1[1][0] + i * pad_move - pad_move_d, diag_eq(line1, external1[1][0] + i * pad_move - pad_move_d)-pad_v),
            (external1[2][0]+ i * pad_move + pad_move_d, diag_eq(line2, external1[2][0]+ i * pad_move + pad_move_d)+ pad_v), 
            (external1[3][0] + i * pad_move + pad_move_d, diag_eq(line2, external1[3][0] + i * pad_move + pad_move_d)+ pad_v)])
      self.externals.append([(external1[0][0] + i * pad_move + pad_move_d, diag_eq(line1, external1[0][0] + i * pad_move + pad_move_d)-pad_v), 
                             (external1[1][0] + i * pad_move + pad_move_d, diag_eq(line1, external1[1][0] + i * pad_move + pad_move_d)-pad_v),
            (external1[2][0] + i * pad_move - pad_move_d, diag_eq(line2,external1[2][0] + i * pad_move - pad_move_d)+ pad_v), 
            (external1[3][0] + i * pad_move - pad_move_d,  diag_eq(line2,external1[3][0] + i * pad_move - pad_move_d)+ pad_v)])
      i += 1
    self.line1 = line1
    self.line2 = line2
    self.limit = r_points

  def count_line(self, externals, threshold=None):    
    lines_found = []
    ex_idx = 0
    while ex_idx < len(externals):
      background, cnt = getBackground(externals[ex_idx], self.img, threshold=threshold)
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

  def get_score(self, threshold):
    lines_found = self.count_line(self.externals, threshold=threshold)
    rect_or = None    
    if lines_found.shape[0] >= 1:    
      (lefty, max_left), (righty, max_right) = lines_found[0]             
      if np.abs(np.rad2deg(np.arctan2(max_right - max_left, righty - lefty))) > 80:
        #print('best inclination: {}'.format(np.abs(np.rad2deg(np.arctan2(righty - lefty, max_right - max_left)))))
        rect_or = np.array([[lefty, max_left], [righty, max_right]])  

    if rect_or is not None:
      self.drawing = cv2.circle(self.drawing, tuple(rect_or[0]), 15, (255, 0, 0), 2)
      self.drawing = cv2.circle(self.drawing, tuple(rect_or[1]), 15, (255, 0, 0), 2)
      if lines_found.shape[0] == 1:
        
        line_fig = unary_union([Point(tuple(rect_or[0])).buffer(15), LineString(rect_or).buffer(2), Point(tuple(rect_or[1])).buffer(15)])
        line1_fig = LineString(self.line1).buffer(2)
        line2_fig = LineString(self.line2).buffer(2)
        p1 = line_fig.intersects(line2_fig)
        if not p1:
          print('PATTERN13: vertice basso non tocca triangolo')
        p2 = line_fig.intersects(line1_fig)
        if not p2:
          print('PATTERN13: vertice alto non tocca triangolo')
        if not (p1 and p2):
          label_or_line = 1
          return self.drawing, label_or_line
        dist = int((self.limit[0]-852)/2)
        p3 = max(rect_or[:,0]) > 852 + dist
        if p3:
          print('PATTERN13: oltre meta malposizionato')
          label_or_line = 2
        else:
          label_or_line = 3
      else:
        print('PATTERN13: troppe linee')
        label_or_line = 1
    else:
      print('PATTERN13: nessuna linea trovata')     
      label_or_line = 0
    return self.drawing, label_or_line  