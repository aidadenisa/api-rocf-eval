import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union

from preprocessing.homography import unique_color, maxDeviationThresh
from prediction.image_processing import draw_contours

# # TODO: SEE IF WE CAN EXTRACT THIS
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

#not the same, cannot be extracted
def getBackground(external, img, show=False, morph=False, ret_hier=False, internal=None, threshold=None):
    # TODO: FIX WHITE BACKGROUND!
    points = np.array(external)
    interval = (max(points[:,1])-min(points[:,1]), max(points[:,0])-min(points[:,0]))
    points_scaled = points.copy()
    points_scaled[:, 0] -= min(points[:, 0])
    points_scaled[:, 1] -= min(points[:, 1])
    background_t = np.zeros(interval, dtype=np.uint8)
    background_t = cv2.fillConvexPoly(background_t, points_scaled.reshape((4, 1, 2)), 255)
    image_interval = img[min(points[:,1]):max(points[:,1]), min(points[:,0]):max(points[:,0])]
    background_t = cv2.bitwise_and(image_interval, background_t)
    if show:
        overlap = cv2.polylines(cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB), [points.reshape(4,1,2)], True, (255, 0, 0), 1)
        # plt.imshow(overlap)
        # plt.show()
    # background_t = unique_color(background_t)
    # if show:
    #     plt.imshow(background_t, cmap='gray')
    #     plt.show()
    # background_t, t_val = extract_drawing(background_t)
    if threshold > 246:
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

def best_line(backgrounds, idx, only_length, external):
    background = backgrounds[idx]
    #plt.imshow(background, cmap='gray')
    #plt.show()
    lines_filtered = cv2.HoughLinesP(background, 1, np.pi / 180, 30, None, 10, 5)
    lines_d = cv2.cvtColor(background.copy(), cv2.COLOR_GRAY2RGB)
    idx_ok = []
    if lines_filtered is not None:
        max_left = np.inf
        max_right = -np.inf
        points = []
        for i in range(0, len(lines_filtered)):
            l = lines_filtered[i][0]
            lines_d = cv2.line(lines_d, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2)
            inclination = np.abs(np.rad2deg(np.arctan2(l[3] - l[1], l[2] - l[0])))
            if inclination < 15:
                points.append((l[0], l[1]))
                if l[0] < max_left:
                    max_left = l[0]
                if l[0] > max_right:
                    max_right = l[0]                
                points.append((l[2], l[3]))
                if l[2] < max_left:
                    max_left = l[2]
                if l[2] > max_right:
                    max_right = l[2]  
                idx_ok.append(i)
                lines_d = cv2.line(lines_d, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 2)
        # print(points)
        #plt.imshow(lines_d)
        #plt.show()
        if len(points) > 0:
            coverage = int_coverage(lines_filtered[idx_ok], external)
            if coverage > 30:
                [vx, vy, x, y] = cv2.fitLine(np.array(points), cv2.DIST_L12, 0, 0.01, 0.01)
                t0 = (max_left - x) / vx
                t1 = (max_right - x) / vx
                lefty = int(y + t0 * vy)
                righty = int(y + t1 * vy)
                # print((max_left, righty), (max_right, lefty))
                
                # print('line length = {}'.format(np.linalg.norm(np.array([max_left, righty]) - np.array([max_right, lefty]))))
                # print('inclination = {}'.format(np.rad2deg(np.arctan2(righty - lefty, max_right - max_left))))
                if only_length:
                    return np.linalg.norm(np.array([max_left, lefty]) - np.array([max_right, righty])), np.abs(
                        np.rad2deg(np.arctan2(righty - lefty, max_right - max_left)))
                else:
                    return (max_left, lefty), (max_right, righty), coverage
        return None

def int_coverage(lines_filtered, external, drawing=None):
    base_interval = set(range(external[3][0] - 20, external[2][0] + 20))
    point_int = []
    if drawing is not None:
        draw_lines = np.zeros_like(drawing)
    for i in range(0, len(lines_filtered)):
        l = lines_filtered[i][0]
        if drawing is not None:
            draw_lines = cv2.line(draw_lines, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 2, cv2.LINE_AA)
        if l[2] > l[0]:
            point_int.append(range(l[0], l[2]))
        else:
            point_int.append(range(l[2], l[0]))
    union_set = set().union(*point_int)
    inter = base_interval.intersection(union_set)
    coverage = (len(inter) / len(base_interval)) * 100
    # if drawing is not None:
    #  print('coverage = {}%'.format(coverage))
    return coverage

pad_v = 6
pad_h_dx = 10
pad_h_sx = 15
pad_move = 12
pad_move_d = 5
dist = int((852 - 382) / 2)

# TODO: SEE IF WE CAN EXTRACT IT
def diag_eq(diag, y):
    return int(diag[0][0] + ((diag[1][0]-diag[0][0])/(diag[1][1]-diag[0][1]))*(y-diag[0][1]))

class Pattern8:
  def __init__(self, img, drawing, diag1, diag2, vert, oriz):
    self.img = img    
    self.drawing = drawing
    if diag1 is None:
      diagonale = [(382, 219), (852, 537)]
    else:
      diagonale = diag1       
    if vert is None:
      dist = int((852 - 382) / 2)
      verticale = [(852 - dist, 219), (852 - dist, 537)]
    else:
      verticale = vert
    line = [(diag_eq(diagonale, diagonale[0][1]+20), diagonale[0][1]+20), (diag_eq(verticale, diagonale[0][1]+20), diagonale[0][1]+20)]
    
    if oriz is not None:
      or_limit = min(oriz[:,1]) - 5
    else:
      dist = int((537 - 219) / 2)
      or_limit = 219 + dist - 5     
    if diag1 is not None and diag2 is not None:
        line1 = LineString(diag1)
        line2 = LineString(diag2)
        int_pt = line1.intersection(line2)
        point_of_intersection = (int(int_pt.x), int(int_pt.y))
        if point_of_intersection[1] < or_limit:
          or_limit = point_of_intersection[1] - 5
      
    external1 = [(line[0][0] - pad_h_sx, line[0][1] - pad_v), (line[1][0] + pad_h_dx, line[1][1] - pad_v),
                  (line[1][0] + pad_h_dx, line[1][1] + pad_v), (line[0][0] - pad_h_sx, line[0][1] + pad_v)]
    self.externals = []
    i = 0
    while external1[3][1] + i * pad_move < or_limit:
        if diag_eq(verticale, external1[2][1] + i * pad_move) + pad_h_dx <= diag_eq(diagonale, external1[3][1] + i * pad_move) - pad_h_sx or diag_eq(verticale,
            external1[2][1] + i * pad_move + pad_move_d) + pad_h_dx <= diag_eq(diagonale, external1[3][1] + i * pad_move - pad_move_d) - pad_h_sx or diag_eq(verticale,
            external1[2][1] + i * pad_move - pad_move_d) + pad_h_dx <= diag_eq(diagonale, external1[3][1] + i * pad_move + pad_move_d) - pad_h_sx:
                break
        self.externals.append([(diag_eq(diagonale, external1[0][1] + i * pad_move) - pad_h_sx, external1[0][1] + i * pad_move),
                          (diag_eq(verticale, external1[1][1] + i * pad_move) + pad_h_dx, external1[1][1] + i * pad_move),
                          (diag_eq(verticale, external1[2][1] + i * pad_move) + pad_h_dx, external1[2][1] + i * pad_move),
                          (diag_eq(diagonale, external1[3][1] + i * pad_move) - pad_h_sx, external1[3][1] + i * pad_move)])
        self.externals.append([(diag_eq(diagonale, external1[0][1] + i * pad_move - pad_move_d) - pad_h_sx, external1[0][1] + i * pad_move - pad_move_d),
                          (diag_eq(verticale, external1[1][1] + i * pad_move + pad_move_d) + pad_h_dx, external1[1][1] + i * pad_move + pad_move_d),
                          (diag_eq(verticale, external1[2][1] + i * pad_move + pad_move_d) + pad_h_dx, external1[2][1] + i * pad_move + pad_move_d), 
                          (diag_eq(diagonale, external1[3][1] + i * pad_move - pad_move_d) - pad_h_sx, external1[3][1] + i * pad_move - pad_move_d)])
        self.externals.append([(diag_eq(diagonale, external1[0][1] + i * pad_move + pad_move_d) - pad_h_sx, external1[0][1] + i * pad_move + pad_move_d),
                          (diag_eq(verticale, external1[1][1] + i * pad_move - pad_move_d) + pad_h_dx, external1[1][1] + i * pad_move - pad_move_d),
                          (diag_eq(verticale, external1[2][1] + i * pad_move - pad_move_d) + pad_h_dx, external1[2][1] + i * pad_move - pad_move_d), 
                          (diag_eq(diagonale, external1[3][1] + i * pad_move + pad_move_d) - pad_h_sx, external1[3][1] + i * pad_move + pad_move_d)])
        i += 1
    self.vert = verticale
    self.diag = diagonale

  def count_line(self, externals, threshold=None):    
    lines_found = []
    last_best_cov = 0
    ex_idx = 0
    while ex_idx < len(externals):
      background1, cnt1 = getBackground(externals[ex_idx], self.img, threshold=threshold)
      background2, cnt2 = getBackground(externals[ex_idx + 1], self.img, threshold=threshold)
      background3, cnt3 = getBackground(externals[ex_idx + 2], self.img, threshold=threshold)
      result1 = best_line([background1], 0, False, externals[ex_idx])
      result2 = best_line([background2], 0, False, externals[ex_idx + 1])
      result3 = best_line([background3], 0, False, externals[ex_idx + 2])
      if result1 is not None:
          cov1 = result1[-1]
      else:
          cov1 = 0
      if result2 is not None:
          cov2 = result2[-1]
      else:
          cov2 = 0
      if result3 is not None:
          cov3 = result3[-1]
      else:
          cov3 = 0
      if cov1 or cov2 or cov3:
        best = np.argmax([cov1, cov2, cov3])
        if best == 0:
            (max_left, lefty), (max_right, righty) = result1[:-1]
        elif best == 1:
            (max_left, lefty), (max_right, righty) = result2[:-1]
        elif best == 2:
            (max_left, lefty), (max_right, righty) = result3[:-1]
        if len(lines_found) > 0:
            line_before = lines_found[-1]
            if abs(lefty - line_before[0][1]) >= 10 and abs(righty - line_before[1][1]) >= 10:
                lines_found.append([(max_left, lefty), (max_right, righty)])
                last_best_cov = [cov1, cov2, cov3][best]
            else:
                if [cov1, cov2, cov3][best] > last_best_cov:
                    lines_found[-1] = [(max_left, lefty), (max_right, righty)]
                    last_best_cov = [cov1, cov2, cov3][best]
        else:
            lines_found.append([(max_left, lefty), (max_right, righty)])
            last_best_cov = [cov1, cov2, cov3][best]

      ex_idx += 3
    return np.array(lines_found)

  def get_score(self, threshold):
    lines_found = self.count_line(self.externals, threshold=threshold)

    if lines_found.shape[0] > 0:    
      for l in lines_found:
        self.drawing = cv2.line(self.drawing, tuple(l[0]), tuple(l[1]), (0,0,255), 2)
        for p in l:
            self.drawing = cv2.circle(self.drawing, tuple(p), 15, (255, 0, 0), 2)
      if lines_found.shape[0] == 4:
          diag_fig = LineString(self.diag).buffer(5)
          vert_fig = LineString(self.vert).buffer(5)
          p1 = True
          for l in range(lines_found.shape[0]):
            line_fig = unary_union([Point(tuple(lines_found[l][0])).buffer(15), LineString(lines_found[l]).buffer(1.5), Point(tuple(lines_found[l][1])).buffer(15)])
            p1 = p1 and line_fig.intersects(diag_fig) and line_fig.intersects(vert_fig)
            if not (line_fig.intersects(diag_fig) and line_fig.intersects(vert_fig)):
              print('PATTERN8: diagonale {} distorta'.format(l))
          if p1:
            label_l = 3
          else:
            label_l = 1
      else:
        print('PATTERN8: numero linee sbagliato')
        label_l = 1
    else:
      print('PATTERN8: nessuna linea trovata')
      label_l = 0
    return self.drawing, label_l