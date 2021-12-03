import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from shapely.geometry import Polygon
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

from prediction.image_processing import getBackground, draw_contours

h_dist = 25  # distanza diviso 2
v_dist = 150
dist = int((852 - 382) / 2)
line = [(852 - dist - 100, 219), (852, 219)]


class Pattern9:
  def __init__(self, img, drawing, vert):
    self.img = img    
    self.drawing = drawing    
    if vert is None:
        self.external = [(line[0][0] - h_dist, line[0][1]+10), (line[0][0] - h_dist, line[0][1]-v_dist),
               (line[1][0] + h_dist, line[1][1]-v_dist), (line[1][0] + h_dist, line[1][1]+10)]
    else:
        self.external = [(vert[1][0] - h_dist, vert[1][1] + 20), (vert[1][0] - h_dist, vert[1][1] - v_dist),
                         (line[1][0] + h_dist, line[1][1] - v_dist), (line[1][0] + h_dist, line[1][1] + 20)]

  def tree(self, hier, selected):
    if len(selected) == 1:
        return selected[0]
    idx = selected[0]
    if hier[0][idx][2] != -1:
        return self.tree(hier, selected[1:])
    else:
        return selected[0]

  def get_score(self, rett, vert):
    rhomb = None
    background_r, cnts_r, hier_r = getBackground(self.external, self.img, True, True)
    ok_idx = []
    wrong_shapes = []
    pixel_rhomb = np.sum(np.divide(background_r, 255))
    for c in range(len(cnts_r)):
        self.drawing = draw_contours(self.drawing, [cnts_r[c]])
    for c in range(len(cnts_r)):
        peri = cv2.arcLength(cnts_r[c], True)
        approx = cv2.approxPolyDP(cnts_r[c], 0.05 * peri, True)
        # print('peri: {}, vert: {}'.format(peri, len(approx)))
        if len(approx) == 3 and peri > 200:
            ok_idx.append(c)
            # approx = enlargeRhomb(approx, 1)
            # approx = cv2.approxPolyDP(approx, 0.05 * peri, True)
        elif len(approx) != 3 and peri > 200:
          wrong_shapes.append(c)
    if len(ok_idx) > 0:
        selected_c_idx = self.tree(hier_r, ok_idx)
        peri = cv2.arcLength(cnts_r[selected_c_idx], True)
        approx = cv2.approxPolyDP(cnts_r[selected_c_idx], 0.05 * peri, True)
        rhomb = np.array([coord[0] for coord in approx])
        rhomb = rhomb[np.argsort(rhomb[:,0]), :]
        rect = cv2.minAreaRect(approx)
        (x, y), (width, height), angle = rect               
        # print('orientation = {}'.format(angle))
        self.drawing = cv2.polylines(self.drawing, [approx], True, (0, 191, 255), 2)
           
    if rhomb is not None:
        tri_points = []
        incl1 = np.abs(np.rad2deg(np.arctan2(rhomb[1][1] - rhomb[0][1], rhomb[1][0] - rhomb[0][0])))
        for p in rhomb:
            self.drawing = cv2.circle(self.drawing, tuple(p), 25, (255, 0, 0), 2)
            tri_points.append(Point(tuple(p)).buffer(25))
        if incl1 > 80:
            tri_points.extend([Polygon(rhomb).buffer(3)])
            tri_fig = unary_union(tri_points)
            if rett is not None:
              p1 = tri_fig.intersects(rett[1][1])
              if not p1:
                print('PATTERN9: non tocca angolo dx')
            if vert is not None:
              v = Point(vert[1]).buffer(15)
              p2 = tri_fig.intersects(v)
              if not p2:
                print('PATTERN9: non tocca angolo verticale')
            if (rett is None or p1) and (vert is None or p2):
              label_rhomb = 3
            else:
              label_rhomb = 2
        else:
            print('PATTERN9: forma distorta')
            label_rhomb = 1 
    elif len(wrong_shapes) > 0:
          found = False
          for c in wrong_shapes:
            peri = cv2.arcLength(cnts_r[c], True)
            approx = cv2.approxPolyDP(cnts_r[c], 0.05 * peri, True)
            hull = cv2.convexHull(approx)
            uguali = np.all(hull in approx)
            if len(approx) > 2 and uguali:
                self.drawing = cv2.polylines(self.drawing, [approx], True, (0, 191, 255), 1)
                found = True
          if found:
            print('PATTERN9: forma sbagliata')
            label_rhomb = 1
          else:
              print('PATTERN9: forma assente')
              label_rhomb = 0
    else:
        print('PATTERN9: forma assente')
        label_rhomb = 0
    return self.drawing, label_rhomb
