import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from shapely.geometry import Polygon, Point, LineString

from preprocessing.homography import maxDeviationThresh
from prediction.image_processing import draw_contours

# TODO: TRY TO EXTRACT THIS VERSION
def extract_drawing(image):
    dst = cv2.bilateralFilter(image, 10, sigmaColor=15, sigmaSpace=15)
    # dst = img.copy()
    # max_occ = np.bincount(dst[dst > 0]).argmax()
    # dst[dst == 0] = max_occ
    threshed = np.ones(dst.shape, np.uint8) * 255
    thresh_val = 0
    if np.any(dst < 255):
        hist, _ = np.histogram(dst[dst < 255].flatten(), range(257))
        thresh_val = maxDeviationThresh(hist)
        mask = dst < thresh_val
        threshed[mask] = 0
    return threshed, thresh_val

# a bit different than the most common version => cannot extract it
def getBackground(external, img, morph=False, ret_hier=False, internal=None):
    points = np.array(external)
    interval = (max(points[:,1])-min(points[:,1]), max(points[:,0])-min(points[:,0]))
    points_scaled = points.copy()
    points_scaled[:, 0] -= min(points[:, 0])
    points_scaled[:, 1] -= min(points[:, 1])
    background_t = np.zeros(interval, dtype=np.uint8)
    background_t = cv2.fillConvexPoly(background_t, points_scaled.reshape((4, 1, 2)), 255)
    image_interval = img[min(points[:,1]):max(points[:,1]), min(points[:,0]):max(points[:,0])]
    background_t = cv2.bitwise_and(image_interval, background_t)
    #overlap = cv2.polylines(cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB), [points.reshape(4,1,2)], True, (255, 0, 0), 1)
    #plt.imshow(overlap)
    #plt.show()
    background_t[background_t == 0] = 255
    background_t, t_val = extract_drawing(background_t)
    if t_val > 245:
        background_t = np.ones(interval, dtype=np.uint8) * 255
    background = np.ones_like(img) * 255
    background[min(points[:,1]):max(points[:,1]), min(points[:,0]):max(points[:,0])] = background_t
    #plt.imshow(background, cmap='gray')
    #plt.show()
    if morph:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
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

def best_line(backgrounds, idx, only_length, external, draw=False, drawing=None):
    background = backgrounds[idx]    
    lines_filtered = cv2.HoughLinesP(background, 1, np.pi / 180, 40, None, 20, 5)
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
          coverage = int_coverage(lines_filtered[idx_ok], external)
          if coverage > 60:
            [vx, vy, x, y] = cv2.fitLine(np.array(points), cv2.DIST_L12, 0, 0.01, 0.01)
            t0 = (max_left-x)/vx
            t1 = (max_right-x)/vx
            lefty = int(y + t0*vy)
            righty = int(y + t1*vy)
            #print((max_left, righty), (max_right, lefty))

            #print('line length = {}'.format(np.linalg.norm(np.array([max_left, righty]) - np.array([max_right, lefty]))))
            #print('inclination = {}'.format(np.rad2deg(np.arctan2(righty - lefty, max_right - max_left))))
            if only_length:
                return np.linalg.norm(np.array([max_left, lefty]) - np.array([max_right, righty]))
            else:
                return (max_left, lefty), (max_right, righty), drawing
        return None

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

pad_v = 5
pad_h = 10
pad_move = 10
pad_move_d = 5

# TODO: try to extract
def diag_eq(diag, y):
  return int(diag[0][0] + ((diag[1][0]-diag[0][0])/(diag[1][1]-diag[0][1]))*(y-diag[0][1]))

class Pattern4:
  def __init__(self, img, drawing, diag1, oriz):
    self.img = img    
    self.drawing = drawing
    if diag1 is not None:
      diagonale = diag1
    else:
      diagonale = [(382, 219), (852, 537)]
    if oriz is not None:
      or_limit = min(oriz[:,1]) - 20
    else:
      or_limit = 358

    line = [(382, diagonale[0][1]+20), (diag_eq(diagonale, diagonale[0][1]+20), diagonale[0][1]+20)]
    external1 = [(line[0][0] - pad_h, line[0][1] - pad_v), (diag_eq(diagonale,line[1][1] - pad_v) + pad_h, line[1][1] - pad_v),
                     (diag_eq(diagonale, line[1][1] + pad_v) + pad_h, line[1][1] + pad_v), (line[0][0] - pad_h, line[0][1] + pad_v)]
    self.externals = []
    i = 0
    while  external1[0][1] + i * pad_move < or_limit:
      '''self.externals.append(
            [(external1[0][0], external1[0][1] - i * pad_move), (diag_eq(diagonale, external1[1][1] - i * pad_move), external1[1][1] - i * pad_move),
              (diag_eq(diagonale, external1[2][1] - i * pad_move), external1[2][1] - i * pad_move), (external1[3][0], external1[3][1] - i * pad_move)])
      self.externals.append([(external1[0][0], external1[0][1] - i * pad_move - pad_move_d), (diag_eq(diagonale, external1[1][1] - i * pad_move + pad_move_d), external1[1][1] - i * pad_move + pad_move_d),
            (diag_eq(diagonale, external1[2][1] - i * pad_move + pad_move_d), external1[2][1] - i * pad_move + pad_move_d), (external1[3][0], external1[3][1] - i * pad_move - pad_move_d)])
      self.externals.append([(external1[0][0], external1[0][1] - i * pad_move + pad_move_d), (diag_eq(diagonale, external1[1][1] - i * pad_move - pad_move_d), external1[1][1] - i * pad_move - pad_move_d),
            (diag_eq(diagonale, external1[2][1] - i * pad_move - pad_move_d), external1[2][1] - i * pad_move - pad_move_d), (external1[3][0], external1[3][1] - i * pad_move + pad_move_d)])'''      
      self.externals.append([(external1[0][0], external1[0][1] + i * pad_move), (diag_eq(diagonale, external1[1][1] + i * pad_move), external1[1][1] + i * pad_move),
            (diag_eq(diagonale, external1[2][1] + i * pad_move), external1[2][1] + i * pad_move), (external1[3][0], external1[3][1] + i * pad_move)])
      self.externals.append([(external1[0][0], external1[0][1] + i * pad_move - pad_move_d), (diag_eq(diagonale, external1[1][1] + i * pad_move + pad_move_d), external1[1][1] + i * pad_move + pad_move_d),
            (diag_eq(diagonale, external1[2][1] + i * pad_move + pad_move_d), external1[2][1] + i * pad_move + pad_move_d), (external1[3][0], external1[3][1] + i * pad_move - pad_move_d)])
      self.externals.append([(external1[0][0], external1[0][1] + i * pad_move + pad_move_d), (diag_eq(diagonale, external1[1][1] + i * pad_move - pad_move_d), external1[1][1] + i * pad_move - pad_move_d),
            (diag_eq(diagonale, external1[2][1] + i * pad_move - pad_move_d), external1[2][1] + i * pad_move - pad_move_d), (external1[3][0], external1[3][1] + i * pad_move + pad_move_d)])
      i += 1

  def count_line(self, externals):    
    lines_found = []
    ex_idx = 0
    while ex_idx < len(externals):
      background, cnt = getBackground(externals[ex_idx], self.img)
      result = best_line([background], 0, False, externals[ex_idx])
      if result is not None:
        (max_left, lefty), (max_right, righty), drawing = result
        if len(lines_found) > 0:
            line_before = lines_found[-1]
            if abs(lefty - line_before[0][1]) >= 5 and abs(righty - line_before[1][1]) >= 5:
                lines_found.append([(max_left, lefty), (max_right, righty)])
                self.drawing = cv2.line(self.drawing, (max_left, lefty), (max_right, righty), (0, 0, 255), 2, cv2.LINE_AA)
        else:
            lines_found.append([(max_left, lefty), (max_right, righty)])
            self.drawing = cv2.line(self.drawing, (max_left, lefty), (max_right, righty), (0, 0, 255), 2, cv2.LINE_AA)
        ex_idx += 1
        while ex_idx % 3 != 0:
            ex_idx +=1
      else:
        ex_idx += 1
    return np.array(lines_found)

  def get_score(self, diag1, rect):    

    lines_found = self.count_line(self.externals)
    rect_or = None
    
    if lines_found.shape[0] >= 2:    
      (max_left, lefty), (max_right, righty) = lines_found[0]             
      if np.abs(np.rad2deg(np.arctan2(righty - lefty, max_right - max_left))) < 10:
        #print('best inclination: {}'.format(np.abs(np.rad2deg(np.arctan2(righty - lefty, max_right - max_left)))))
        rect_or = np.array([[max_left, lefty], [max_right, righty]])

    if rect_or is not None:
        p1 = None
        p2 = None
        self.drawing = cv2.circle(self.drawing, tuple(rect_or[0]), 15, (255, 0, 0), 2)
        self.drawing = cv2.circle(self.drawing, tuple(rect_or[1]), 20, (255, 0, 0), 2)
        lines_points_or = [Point(tuple(rect_or[0])).buffer(15), Point(tuple(rect_or[1])).buffer(20)]
        line_or = LineString(rect_or).buffer(1.5)
        if diag1 is not None:
          d1 = LineString(diag1).buffer(1.5)
          p1_1 = lines_points_or[1].intersects(d1)
          p1_2 = line_or.intersects(d1)
          if not (p1_1 or p1_2):
            print('PATTERN4: linea non tocca diagonale')
        if rect is not None:
          p2 = line_or.intersects(rect[0]) or lines_points_or[0].intersects(rect[0])
          if not p2:
            print('PATTERN4: linea non tocca quadrato')
        if lines_found.shape[0] > 2:
          print('PATTERN4: troppe linee')
        if (p1 is None or p1) and (p2 is None or p2) and lines_found.shape[0] == 2:      
          label_or_line = 3
        else:          
          label_or_line = 1      
    else:
      print('PATTERN4: nessuna linea trovata')      
      label_or_line = 0
    return self.drawing, label_or_line