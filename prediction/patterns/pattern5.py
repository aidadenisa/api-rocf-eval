import pandas as pd
import ast
import numpy as np
import cv2
from skimage.morphology import skeletonize


from preprocessing.homography import maxDeviationThresh

# TODO: SEE IF WE CAN EXTRACT THIS IN THE FUTURE
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
        #print(thresh_val)
        mask = dst < thresh_val
        threshed[mask] = 0
    return threshed, thresh_val


def best_line(backgrounds, idx, only_length, external, draw=False):
    background = backgrounds[idx]
    lines_filtered = cv2.HoughLinesP(background, 1, np.pi / 180, 20, None, 20, 5)
    idx_ok = []
    if lines_filtered is not None:
        max_left = np.inf
        max_right = -np.inf
        points = []
        for i in range(0, len(lines_filtered)):
            l = lines_filtered[i][0]
            inclination = np.abs(np.rad2deg(np.arctan2(l[3] - l[1], l[2] - l[0])))
            if inclination < 20:
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
          if coverage > 30:
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

# not the same as the others 
def getBackground(external, img, morph=False, ret_hier=False, internal=None):
    points = np.array(external)
    interval = (max(points[:, 1]) - min(points[:, 1]), max(points[:, 0]) - min(points[:, 0]))
    points_scaled = points.copy()
    points_scaled[:, 0] -= min(points[:, 0])
    points_scaled[:, 1] -= min(points[:, 1])
    background_t = np.zeros(interval, dtype=np.uint8)
    background_t = cv2.fillConvexPoly(background_t, points_scaled.reshape((4, 1, 2)), 255)
    image_interval = img[min(points[:, 1]):max(points[:, 1]), min(points[:, 0]):max(points[:, 0])]
    background_t = cv2.bitwise_and(image_interval, background_t)
    # overlap = cv2.polylines(cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB), [points.reshape(4,1,2)], True, (255, 0, 0), 1)
    # plt.imshow(overlap)
    # plt.show()
    background_t[background_t == 0] = 255
    background_t, t_val = extract_drawing(background_t)
    if t_val > 245:
        background_t = np.ones(interval, dtype=np.uint8) * 255
    background = np.ones_like(img) * 255
    background[min(points[:, 1]):max(points[:, 1]), min(points[:, 0]):max(points[:, 0])] = background_t
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
        # plt.imshow(background, cmap='gray')
        # plt.show()
    cnts, hier = cv2.findContours(background, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if ret_hier:
        return background, cnts, hier
    else:
        return background, cnts

def get_score_externals(externals, img):
    backgrounds = []
    cnts = []
    rect_or = None
    for external in externals:
      background, cnt = getBackground(external, img)
      backgrounds.append(background)
      cnts.append(cnt)
    best_diff = np.inf
    best_back = 0
    count_found = 0
    for background in range(len(backgrounds)):
      ideal_length = np.linalg.norm(np.array(externals[background][0]) - np.array(externals[background][1]))
      length = best_line(backgrounds, background, True, externals[background])
      if length is not None and np.abs(length - ideal_length) < best_diff:
        best_diff = np.abs(length - ideal_length)
        best_back = background
        count_found += 1
    #print('best_back: {}'.format(best_back))
    result = best_line(backgrounds, best_back, False, externals[best_back], True)
    pixel_lines = np.sum(np.divide(backgrounds[best_back], 255))
    if result is not None:
      (max_left, lefty), (max_right, righty) = result
      if np.abs(np.rad2deg(np.arctan2(righty - lefty, max_right - max_left))) < 20:
        #print('best inclination: {}'.format(np.abs(np.rad2deg(np.arctan2(righty - lefty, max_right - max_left)))))
        rect_or = np.array([[max_right, righty], [max_left, lefty]])
    if rect_or is not None and count_found == 1:
      label_diag_line = 3
      return label_diag_line, rect_or
    elif rect_or is not None and count_found > 1:
      label_diag_line = 1
      return label_diag_line, rect_or
    else:
        label_diag_line = 0
    return label_diag_line, None


def get_diag(bbox, img):
    pad_v = 10
    pad_move = 20
    if bbox[1][0] > 380:
        line = [bbox[3], (bbox[1][0], bbox[3][1])]
    else:
        line = [bbox[3], (380, bbox[3][1])]
    external = [(line[0][0], line[0][1] - pad_v), (line[1][0], line[1][1] - pad_v),(line[1][0], line[1][1] + pad_v), (line[0][0], line[0][1] + pad_v)]
    i = 0
    externals = []
    dist = int((bbox[3][1] - bbox[0][1])/4)
    while external[0][1] - i * pad_move > bbox[0][1] + dist:
        externals.append([(external[0][0], external[0][1] - i * pad_move), (external[1][0], external[1][1] - i * pad_move),
                          (external[2][0], external[2][1] - i * pad_move), (external[3][0], external[3][1] - i * pad_move)])
        i += 1
    label, rect_or = get_score_externals(externals, img)
    if label == 3:
        if 230 < rect_or[0][1] < 360 and rect_or[0][1] > bbox[0][1] + dist:
            return 3, rect_or
        else:
            return 2, rect_or
    elif label == 1:
        return 1, rect_or
    return 0, None


class Pattern5:
  def __init__(self, img, drawing, model_diag, scaler_diag, m, s, predictionComplexScores):
    self.img = img    
    self.drawing = drawing
    self.model_diag = model_diag
    self.scaler_diag = scaler_diag
    self.m = m
    self.s = s
    self.predictionComplexScores = predictionComplexScores
  
  def get_score(self):
    coords = [324,119,378,373]
    
    if self.predictionComplexScores:
        rail_bbox = self.predictionComplexScores['rect'][4]
        external = [(rail_bbox[0], rail_bbox[1]), (rail_bbox[0] + rail_bbox[2], rail_bbox[1]),(rail_bbox[0] + rail_bbox[2], rail_bbox[1] + rail_bbox[3]),
                    (rail_bbox[0], rail_bbox[1] + rail_bbox[3])]
        background_rail, _ = getBackground(external, self.img, False)
        pixel_rail = np.sum(np.divide(background_rail, 255))
        rail_prediction = self.model_diag.predict(self.scaler_diag.transform(np.array([pixel_rail]).reshape(-1, 1)))
        score_rail = self.s.transform(np.array(self.predictionComplexScores['scores'][0]).reshape(-1,1))
        rail_score = self.m.predict(score_rail)        
        if rail_score == 1:
            self.drawing = cv2.rectangle(self.drawing, (rail_bbox[0], rail_bbox[1]), (rail_bbox[0] + rail_bbox[2], rail_bbox[1] + rail_bbox[3]), (255, 0, 0), 2)
            result = get_diag(external, self.img)
            if result[0] != 0:
                label_rail, l = result
                self.drawing = cv2.line(self.drawing, tuple(l[0]), tuple(l[1]), (0, 0, 255), 2)
                if label_rail != 3:
                    print('PATTERN5: distorto')                
            else:
                label_rail = 1
                print('PATTERN5: linea attacco rettangolo mancante')
        else:
            if rail_prediction == 1:
                self.drawing = cv2.rectangle(self.drawing, (rail_bbox[0], rail_bbox[1]),(rail_bbox[0] + rail_bbox[2], rail_bbox[1] + rail_bbox[3]),
                                              (0, 0, 255), 2)
                print('PATTERN5: disegno impreciso')
                label_rail = 1
            else:
                self.drawing = cv2.rectangle(self.drawing, (rail_bbox[0], rail_bbox[1]),
                                              (rail_bbox[0] + rail_bbox[2], rail_bbox[1] + rail_bbox[3]), (0, 0, 255), 2)
                print('PATTERN5: disegno assente')
                label_rail = 0
    else:
        x = coords[0]
        y = coords[1]
        w = np.abs(coords[0] - coords[2])
        h = np.abs(coords[1] - coords[3])
        external = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        background_rail, _ = getBackground(external, self.img, False)
        pixel_rail = np.sum(np.divide(background_rail, 255))
        rail_prediction = self.model_diag.predict(self.scaler_diag.transform(np.array([pixel_rail]).reshape(-1, 1)))
        if rail_prediction == 1:
            label_rail = 1
        else:
            label_rail = 0
    return self.drawing, label_rail