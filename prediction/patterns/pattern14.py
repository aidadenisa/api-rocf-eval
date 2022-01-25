import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from shapely.geometry import Polygon
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from shapely.geometry.base import CAP_STYLE

from preprocessing.homography import sharpenDrawing
from prediction.image_processing import draw_contours

#TODO: TRY TO EXTRACT, IT HAS SKELETONIZE A BIT DIFFERENT
def getBackground(external, img, morph=True, ret_hier=False):
    background = np.zeros_like(img)
    points = np.array([external]).reshape((4, 1, 2))
    background = cv2.fillConvexPoly(background, points, (255, 255, 255))
    not_background = cv2.bitwise_not(background)
    background = cv2.bitwise_and(img, background)    
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
    cnts, hier = cv2.findContours(background, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if ret_hier:
        return background, cnts, hier
    else:
        return background, cnts

h_dist = 25  # distanza diviso 2
v_dist = 132

def buildROIs(triangle):
  rois = []

  if len(triangle) > 2:
    roi = Polygon([tuple(p) for p in triangle])
    buffer = roi.buffer(30, cap_style=CAP_STYLE.square)
    simplified = buffer.simplify(tolerance=0.95, preserve_topology=True)
    coords = np.array(list(simplified.exterior.coords)).astype(int)
    rois.append(coords.tolist())

  return rois

class Pattern14:
  def __init__(self, img, drawing, r_points):
    self.img = img    
    self.drawing = drawing
    self.roi = []
    p_dst_r = [(r_points[0] - h_dist, r_points[1]), (r_points[0] + h_dist, r_points[1]),
               (r_points[0] + h_dist, r_points[1] + v_dist), (r_points[0] - h_dist, r_points[1] + v_dist)]
    pad = 50
    self.external = [(p_dst_r[0][0] - pad, p_dst_r[0][1] - pad), (int(p_dst_r[1][0] + 1.5 * pad), p_dst_r[1][1] - pad),
                (int(p_dst_r[2][0] + 1.5 * pad), int(p_dst_r[2][1] + 1.5 * pad)), (p_dst_r[3][0] - pad, int(p_dst_r[3][1] + 1.5 * pad))]

  def tree(self, hier, selected):
    if len(selected) == 1:
        return selected[0]
    idx = selected[0]
    if hier[0][idx][2] != -1:
        return self.tree(hier, selected[1:])
    else:
        return selected[0]

  def get_score(self, r_points, diag1, diag2):
    rhomb = None
    background_r, cnts_r, hier_r = getBackground(self.external, self.img, False, True)
    pixel_rhomb = np.sum(np.divide(background_r, 255))
    ok_idx = []
    wrong_shapes = []
    for c in range(len(cnts_r)):
        self.drawing = draw_contours(self.drawing, [cnts_r[c]])
    for c in range(len(cnts_r)):
        peri = cv2.arcLength(cnts_r[c], True)
        approx = cv2.approxPolyDP(cnts_r[c], 0.05 * peri, True)
        hull = cv2.convexHull(approx)
        confront = [h in approx for h in hull]
        if len(approx) == 4 and peri > 100 and np.all(confront):
          ok_idx.append(c)
            # approx = enlargeRhomb(approx, 1)
            # approx = cv2.approxPolyDP(approx, 0.05 * peri, True)
        elif len(approx) != 4 and peri > 100 and np.all(confront):
          wrong_shapes.append(c)
    if len(ok_idx) > 0:
        selected_c_idx = selected_c_idx = self.tree(hier_r, ok_idx)
        peri = cv2.arcLength(cnts_r[selected_c_idx], True)
        approx = cv2.approxPolyDP(cnts_r[selected_c_idx], 0.05 * peri, True)
        rhomb = np.array([coord[0] for coord in approx])
        rect = cv2.minAreaRect(approx)
        (x, y), (width, height), angle = rect               
        # print('orientation = {}'.format(angle))
        self.drawing = cv2.polylines(self.drawing, [approx], True, (0, 191, 255), 2)
        for vert in approx:
            self.drawing = cv2.circle(self.drawing, tuple(vert[0]), 5, (0, 0, 255), -1)
    if rhomb is not None:
        self.roi = buildROIs(rhomb)
        rhomb_points = []
        sort = np.argsort(rhomb[:, 1])
        for p in sort:
            self.drawing = cv2.circle(self.drawing, tuple(rhomb[p]), 20, (255, 0, 0), 2)
            rhomb_points.append(Point(tuple(rhomb[p])).buffer(20))
        rhomb_points.append(Polygon(rhomb).buffer(1.5),)
        rhomb_fig = unary_union(rhomb_points)
        if diag1 is not None and diag2 is not None:
          diag1_fig = LineString([diag1[0], list(r_points)]).buffer(3)
          diag2_fig = LineString([diag2[0], list(r_points)]).buffer(3)
        else:
          diag1_fig = LineString([[852, 219], list(r_points)]).buffer(3)
          diag2_fig = LineString([[852, 537], list(r_points)]).buffer(3)
        p1 = diag1_fig.intersects(rhomb_fig) and diag2_fig.intersects(rhomb_fig)
        if not p1:
          print('PATTERN14: non tocca diagonali')
          label_rhomb = 2
        else:
          label_rhomb = 3        
        return self.drawing, label_rhomb, (rhomb_fig, rhomb_points)
    elif len(wrong_shapes) > 0:
          print('PATTERN14: forma sbagliata')
          for c in wrong_shapes:
            peri = cv2.arcLength(cnts_r[c], True)
            approx = cv2.approxPolyDP(cnts_r[c], 0.05 * peri, True)
            self.roi = buildROIs(np.array(approx).reshape(-1, 2))
            self.drawing = cv2.polylines(self.drawing, [approx], True, (0, 191, 255), 1)
          label_rhomb = 1
    else:
        print('PATTERN14: forma assente')
        label_rhomb = 0
    return self.drawing, label_rhomb, None

  def get_ROI(self):
    return self.roi