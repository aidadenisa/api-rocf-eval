import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from prediction import image_processing as imgProcess
from prediction.image_processing import thick_rect

# Hough Line Transform - Probabilistically detect the presence of a line
def best_line(backgrounds, idx, external, draw=False, drawing=None):
    # get the background from the array of backgrounds
    background = backgrounds[idx]   
    # detects the presence of lines using probabilistic Hough Transform
    # returns a vector that will store the parameters (xstart,ystart,xend,yend) of the detected lines 
    lines_filtered = cv2.HoughLinesP(background, 1, np.pi / 180, 50, None, 40, 20)
    idx_ok = []
    # if there were some lines detected
    if lines_filtered is not None:
        #initialize interval sizes to - and + infinity; initialize points list
        max_left = np.inf
        max_right = -np.inf
        points = []
        # for every line detected
        for i in range(0, len(lines_filtered)):
            # take the line information and save it in l
            l = lines_filtered[i][0]
            # identify the inclination of the line detected 
            inclination = np.abs(np.rad2deg(np.arctan2(l[3] - l[1], l[2] - l[0])))
            if 20 < inclination < 60:
                # append in points a tuple formed by the coords of the start point of the lin
                points.append((l[0], l[1]))
                 # if the Xstart coord. of the point identified is smaller than the max left, then max_left becomes the new point
                if l[0] < max_left:
                    max_left = l[0]
                # if the X coord. of the point identified is bigger than the max right, then max_right becomes the new point
                if l[0] > max_right:
                    max_right = l[0]
                #if draw:
                #    drawing = cv2.circle(drawing, (l[0], l[1]), 5, (255, 0, 0), -1)
                # append in points a tuple formed by the coords of the end point of the line 
                points.append((l[2], l[3]))
                # if the Ystart is smaller than max_left, then max_left becomes the coord.
                if l[2] < max_left:
                    max_left = l[2]
                # if the Ystart is bigger than max_right, then max_right becomes the coord.
                if l[2] > max_right:
                    max_right = l[2]
                # save the index of a correctly detected inclined line
                idx_ok.append(i)
                #if draw:
                #    drawing = cv2.circle(drawing, (l[2], l[3]), 5, (255, 0, 0), -1)
        #if the length of points array is bigger than 0
        if len(points) > 0:          
          # print((max_left, righty), (max_right, lefty))
          #print(points)
          #print([(max_left, lefty), (max_right, righty)])
          if draw:
                # fit a line between all the points in the points array
                # line	=	cv.fitLine(	points, distType, param, reps, aeps[, line]	)
          # (vx, vy, x0, y0), where (vx, vy) is a normalized vector collinear to the line and (x0, y0) is a point on the line.
                [vx, vy, x, y] = cv2.fitLine(np.array(points), cv2.DIST_L2, 0, 0.01, 0.01)
                # TODO: ASK
                t0 = (max_left - x) / vx
                t1 = (max_right - x) / vx
                lefty = int(y + t0 * vy)
                righty = int(y + t1 * vy)
                #drawing = cv2.line(drawing, (max_left, lefty), (max_right, righty), (0, 0, 255), 2, cv2.LINE_AA)
            #print('line length = {}'.format(np.linalg.norm(np.array([max_left, righty]) - np.array([max_right, lefty]))))
            #print('inclination = {}'.format(np.rad2deg(np.arctan2(righty - lefty, max_right - max_left))))
                # return the points that define the line + the generated drawing.
                return (max_left, lefty), (max_right, righty), drawing
          else:
                # calculate the percentage of coverage that the lines detected have, with respect to the ROI (external)
                # lines_filtered[idx_ok] = the detected lines that have the correct inclination
                # external - ROI that is analysed 
                return int_coverage(lines_filtered[idx_ok], external, draw)
        return None

# calculate the coverage of the lines detected with respect to the ROI 
def int_coverage(lines_filtered, external, drawing=False):
    # external = current ROI that is analysed 
    # lines_filtered = lines detected in the ROI
    matrix = np.array(external)  
    # select the maximum and the minimum x and y from the lines detected and save the range 
    base_interval_x = set(range(min(matrix[:,0]), max(matrix[:,0])))
    base_interval_y = set(range(min(matrix[:,1]), max(matrix[:,1])))
    point_int = []    
    # for each line
    for i in range(0, len(lines_filtered)):
        l = lines_filtered[i][0]    
        # save the interval between the x coords of the points that form the line in the ascending order
        if l[2] > l[0]:
          point_int.append(range(l[0], l[2]))
        else:
          point_int.append(range(l[2], l[0]))
    # create a set ("mulțime") of all the x points that are present in the point_int which cover the detected lines
    union_set = set().union(*point_int)
    # create the intersection of the ROI interval of X coords with the covered X coords from the detected lines
    inter = base_interval_x.intersection(union_set)
    # calculate the report of the intersection over the ROI
    coverage_x = (len(inter) / len(base_interval_x)) * 100
    point_int = []
    # do the save for Y-coordinates: 
    for i in range(0, len(lines_filtered)):
        l = lines_filtered[i][0]
        # save the interval between the y coords of the points that form the line in the ascending order
        if l[3] > l[1]:
            point_int.append(range(l[1], l[3]))
        else:
            point_int.append(range(l[3], l[1]))
    union_set = set().union(*point_int)
    inter = base_interval_y.intersection(union_set)
    # determine the percentage of coverage of the Y coord 
    coverage_y = (len(inter) / len(base_interval_y)) * 100
    #if drawing is not None:
    #  print('coverage = {}%'.format(coverage))
    # return the 2 coverages
    # print("coverage", coverage_x,coverage_y)
    return coverage_x, coverage_y

#initialize the pad values
pad_h = 20
pad_v = 20
pad_rot = 30

# calculate the y coord of the point that has x as x-coord, and that is found on the diag
# this is used for finding the y to represent a point on the diagonal, usually with some buffer already added to x; we need this because we want to add some margin of error
def diag_eq(diag, x):
  # intercept + slope of the diagonal calculated from its points times
  # print("x-diag[0][0])", x-diag[0][0])
  #TODO: why we use (x-diag[0][0])
  return int(diag[0][1] + ((diag[1][1]-diag[0][1])/(diag[1][0]-diag[0][0]))*(x-diag[0][0]))

#generate a new line from the diagonal, which is skewed/rotated from the diagonal by the pad, towards the left (counterclockwise)
def diag_sx(diag):
  return [(diag[0][0]-pad_rot, diag[0][1]), (diag[1][0]+pad_rot, diag[1][1])]

#generate a new line from the diagonal, which is skewed/rotated from the diagonal by the pad, towards the right (counterclockwise)
def diag_dx(diag):
  return [(diag[0][0]+pad_rot, diag[0][1]), (diag[1][0]-pad_rot, diag[1][1])]

#initialize the ideal coordinates of the rectangle in which we have the diagonals (pattern 1)
p_dst = [(382, 219), (852, 219), (852, 537), (382, 537)]
first_diagonal = [p_dst[0], p_dst[2]]
# second_diagonal = 

#calculate half of the distance between the 2 diagonal extreme points
dist_y = int((537-219)/2)
dist_x = int((852-382)/2)

''' calculate the coordinatues of the first diagonal by taking into consideration the 2 halves, as well as a length padding (20) to allow for some margin of error '''

# far left point
# x1 coord of the first diagonal, calculated from the far left point of the first diagonal, from which we remove the margin of error
x11_buffered = p_dst[0][0] - pad_h
# y1 coord of the first diagonal, calculated as a point belonging to the same support vector defined by the first diagonal, which has as x-coord x11_buffered
y11_buffered = diag_eq(first_diagonal, x11_buffered)

# middle point
# x2 coord of the first diagonal, calculated from the far right point of the first diagonal, from which we remove half of the distance between diag points, and we add the margin of error
x12_buffered = p_dst[2][0] - dist_x + pad_h
# y2 coord of the first diagonal, calculated as a point belonging to the same support vector defined by the first diagonal, which has as x-coord x12_buffered
y12_buffered = diag_eq(first_diagonal, x12_buffered)

# far right point
# x3 coord of the first diagonal, calculated from the far right point of the first diagonal to which we add the margin of error
x13_buffered = p_dst[2][0] + pad_h
# y2 coord of the first diagonal, calculated as a point belonging to the same support vector defined by the first diagonal, which has as x-coord x12_buffered
y13_buffered = diag_eq(first_diagonal, x13_buffered)

# diag11 = [(p_dst[0][0] - pad_h, diag_eq([p_dst[0], p_dst[2]], p_dst[0][0] - pad_h)), (p_dst[2][0] - dist_x + pad_h, diag_eq([p_dst[0], p_dst[2]], p_dst[2][0] - dist_x + pad_h))]
# diag12 = [(p_dst[2][0] - dist_x - pad_h, diag_eq([p_dst[0], p_dst[2]], p_dst[2][0] - dist_x - pad_h)), (p_dst[2][0] + pad_h, diag_eq([p_dst[0], p_dst[2]], p_dst[2][0] + pad_h))]
diag11 = [(x11_buffered, y11_buffered), (x12_buffered, y12_buffered)]
diag12 = [(x12_buffered, y12_buffered), (x13_buffered, y13_buffered)]

#generate new line, rotated counterclockwise (to the left) from the original diagonal, with x coords with error buffer
diag11_rot_sx = diag_sx(diag11)
#generate new line, rotated clockwise (to the right) from the original diagonal, with x coords with error buffer
diag11_rot_dx = diag_dx(diag11)
# second half of the diagonal, copy rotated counterclockwise + buffer
diag12_rot_sx = diag_sx(diag12)
# second half of the diagonal, copy rotated clockwise + buffer
diag12_rot_dx = diag_dx(diag12)

''' calculate the coordinatues of the second diagonal by taking into consideration the 2 halves, as well as a length padding (20) to allow for some margin of error '''

diag21 = [(p_dst[1][0] + pad_h, diag_eq([p_dst[1], p_dst[3]], p_dst[1][0] + pad_h)), (p_dst[3][0] + dist_x - pad_h, diag_eq([p_dst[1], p_dst[3]], p_dst[3][0] + dist_x - pad_h))]
diag22 = [(p_dst[3][0] + dist_x + pad_h, diag_eq([p_dst[1], p_dst[3]], p_dst[3][0] + dist_x + pad_h)), (p_dst[3][0] - pad_h, diag_eq([p_dst[1], p_dst[3]], p_dst[3][0] - pad_h))]
diag21_rot_sx = diag_sx(diag21)
diag21_rot_dx = diag_dx(diag21)
diag22_rot_sx = diag_sx(diag22)
diag22_rot_dx = diag_dx(diag22)

thick = 50
pad_move = 20

# list of ROIs build around all diagonals and generated lines, 6 for each diagonal (3 for each half-diagonal)
ex_list = [[thick_rect(diag11, thick), thick_rect(diag11_rot_sx, thick), thick_rect(diag11_rot_dx, thick), thick_rect(diag21, thick), thick_rect(diag21_rot_sx, thick), thick_rect(diag21_rot_dx, thick)],
           [thick_rect(diag12, thick), thick_rect(diag12_rot_sx, thick), thick_rect(diag12_rot_dx, thick), thick_rect(diag22, thick), thick_rect(diag22_rot_sx, thick), thick_rect(diag22_rot_dx, thick)]]

# print(ex_list)

externals1 = [[],[]]
externals2 = [[],[]]
j = 0

# we construct a collection of ROIs by shifting ROI on the right (left for the second diagonal) by a certain margin m (=20) for n 0 shift times. The i-th ROI of the collection based on the starting ROI0 is given by the formula:

# ROIi = [(x1 +i·mshift,y1),(x2 +i·mshift,y2),(x3 +i·mshift,y3), (x4 + i · mshift, y4)] for i = 1,...,5

for external1, external1_rot_sx,  external1_rot_dx, external2, external2_rot_sx, external2_rot_dx in ex_list:
  for i in range(6):
      # create new ROIs to the right of the main diag, by moving them to the left by 20
      externals1[j].append(
          [(external1[0][0] + i * pad_move, external1[0][1]), (external1[1][0] + i * pad_move, external1[1][1]),
          (external1[2][0] + i * pad_move, external1[2][1]), (external1[3][0] + i * pad_move, external1[3][1])])
      externals1[j].append(
          [(external1_rot_dx[0][0] + i * pad_move, external1_rot_dx[0][1]),(external1_rot_dx[1][0] + i * pad_move, external1_rot_dx[1][1]),
          (external1_rot_dx[2][0] + i * pad_move, external1_rot_dx[2][1]),(external1_rot_dx[3][0] + i * pad_move, external1_rot_dx[3][1])])
      externals1[j].append([(external1_rot_sx[0][0] + i * pad_move, external1_rot_sx[0][1]),(external1_rot_sx[1][0] + i * pad_move, external1_rot_sx[1][1]),
          (external1_rot_sx[2][0] + i * pad_move, external1_rot_sx[2][1]),(external1_rot_sx[3][0] + i * pad_move, external1_rot_sx[3][1])])
      # create new ROIs to the left of the secondary diag, by moving them to the left by 20
      externals2[j].append(
          [(external2[0][0] - i * pad_move, external2[0][1]), (external2[1][0] - i * pad_move, external2[1][1]),
          (external2[2][0] - i * pad_move, external2[2][1]), (external2[3][0] - i * pad_move, external2[3][1])])
      externals2[j].append(
          [(external2_rot_dx[0][0] - i * pad_move, external2_rot_dx[0][1]),
          (external2_rot_dx[1][0] - i * pad_move, external2_rot_dx[1][1]),
          (external2_rot_dx[2][0] - i * pad_move, external2_rot_dx[2][1]),
          (external2_rot_dx[3][0] - i * pad_move, external2_rot_dx[3][1])])    
      externals2[j].append(
          [(external2_rot_sx[0][0] - i * pad_move, external2_rot_sx[0][1]),
          (external2_rot_sx[1][0] - i * pad_move, external2_rot_sx[1][1]),
          (external2_rot_sx[2][0] - i * pad_move, external2_rot_sx[2][1]),
          (external2_rot_sx[3][0] - i * pad_move, external2_rot_sx[3][1])])
  j += 1
  
class Pattern1:
  def __init__(self, img, drawing):
    # photo, scan
    self.img = img   
    # generated drawing from the image 
    self.drawing = drawing
  
  # TODO: EXPLAIN WHAT IT DOES; externals represents an array of ROIs of one of the diagonals (1st or 2nd)
  def get_score_externals(self, externals):
    # initialization
    backgrounds = []
    cnts = []
    rect_or = None
    result = None
    # for each generated ROI
    for external in externals:
      # print(external)
      # extract the drawing found in the current ROI, apply thresholding, remove the unwanted background, identify contours
      # background = extracted and thresholded drawing
      # cnts_ret = the contours that were found
      background, cnt = imgProcess.getBackground(external, self.img)
      backgrounds.append(background)
      cnts.append(cnt)
    best_cover_x = -np.inf
    best_cover_y = - np.inf
    best_back = 0
    # after extracting all the backgrounds from ROIs, we itterate through them
    for background in range(len(backgrounds)):
      # identify the best approximation of the diagonal
      # externals[background] = ROI corresponding to the background 
      # returns cover percentage of the best line detected in the ROI
      cover = best_line(backgrounds, background, externals[background])
      if cover is not None and cover[0] > best_cover_x and cover[1] > best_cover_y:
        # if the line detected has a better cover, then save the cover of X and Y coords
        best_cover_x = cover[0]
        best_cover_y = cover[1]
        best_back = background
    # adds the countours of the lines in the drawing, with green
    self.drawing = imgProcess.draw_contours(self.drawing, cnts[best_back])

    # if the best covers are both over 60
    if best_cover_x and best_cover_y > 60:
        # identify and draw the best line in the corresponding drawing, and return the result as the best like coordinates + the generated drawing from the picture
        result = best_line(backgrounds, best_back, externals[best_back], True, self.drawing)
    
    # TODO: unused line, may need to delete
    pixel_lines = np.sum(np.divide(backgrounds[best_back], 255))
    
    if result is not None:
      # save the result in this format
      (max_left, lefty), (max_right, righty), self.drawing = result

      # if the inclination is greater than 20
      if np.abs(np.rad2deg(np.arctan2(righty - lefty, max_right - max_left))) > 20:
        #print('best inclination: {}'.format(np.abs(np.rad2deg(np.arctan2(righty - lefty, max_right - max_left)))))
        rect_or = np.array([[max_right, righty], [max_left, lefty]])         
    
    # if we have the inclination greater than 20*, assign score 3
    if rect_or is not None:      
      label_diag_line = 3
      return label_diag_line, rect_or       
    else:      
      label_diag_line = 0      
    return label_diag_line, None     

  def get_score(self, ret_fig):
    #initialize diagonals
    diag1_coord = None
    diag2_coord = None

    # FIRST DIAGONAL 
    # detect the label and the best coordinates of each half of the first diag
    label11, diag11_coord = self.get_score_externals(externals1[0])
    label12, diag12_coord = self.get_score_externals(externals1[1])
    # if both of them are correct
    if label11 == 3 and label11 == label12:
        # generate geometrical figures by uniting 2 circles around the extremes of the line, + a line with a buffer around 
        line11_fig = unary_union([Point(tuple(diag11_coord[0])).buffer(15), Point(tuple(diag11_coord[1])).buffer(15), LineString(diag11_coord).buffer(1.5)])
        line12_fig = unary_union([Point(tuple(diag12_coord[0])).buffer(15), Point(tuple(diag12_coord[1])).buffer(15), LineString(diag12_coord).buffer(1.5)])
        #if the shapes of the 2 halves of diagonal intersect 
        if line11_fig.intersects(line12_fig):
            # set the full diagonal extreme points and set the label to 3 (correct) 
            diag1_coord = np.array([diag11_coord[1], diag12_coord[0]])
            label1 = 3
            # draw the line, with blue, and the circles, with red 
            self.drawing = cv2.line(self.drawing, tuple(diag1_coord[0]), tuple(diag1_coord[1]), (0,0,255), 2)
            self.drawing = cv2.circle(self.drawing, tuple(diag1_coord[0]), 15, (255, 0, 0), 2)
            self.drawing = cv2.circle(self.drawing, tuple(diag1_coord[1]), 15, (255, 0, 0), 2)
        else:
            # the halves don't intersect
            print('PATTERN1: diagonale1 non continua')
            # draw the 2 halves separatelly and offer the score 1 (distorted)
            self.drawing = cv2.line(self.drawing, tuple(diag11_coord[0]), tuple(diag11_coord[1]), (0, 0, 255), 2)
            self.drawing = cv2.circle(self.drawing, tuple(diag11_coord[0]), 15, (255, 0, 0), 2)
            self.drawing = cv2.circle(self.drawing, tuple(diag11_coord[1]), 15, (255, 0, 0), 2)
            self.drawing = cv2.line(self.drawing, tuple(diag12_coord[0]), tuple(diag12_coord[1]), (0, 0, 255), 2)
            self.drawing = cv2.circle(self.drawing, tuple(diag12_coord[0]), 15, (255, 0, 0), 2)
            self.drawing = cv2.circle(self.drawing, tuple(diag12_coord[1]), 15, (255, 0, 0), 2)
            label1 = 1
    elif label11 == 0 and label11 == label12:
        # not correct
        label1 = 0
    else:
        # distorted if only 1 of the halves is correct
        label1 = 1
    
    # SECOND DIAGONAL
    # Same process
    label21, diag21_coord = self.get_score_externals(externals2[0])
    label22, diag22_coord = self.get_score_externals(externals2[1])
    if label21 == 3 and label21 == label22:
        line21_fig = unary_union([Point(tuple(diag21_coord[0])).buffer(15), Point(tuple(diag21_coord[1])).buffer(15), LineString(diag21_coord).buffer(1.5)])
        line22_fig = unary_union([Point(tuple(diag22_coord[0])).buffer(15), Point(tuple(diag22_coord[1])).buffer(15), LineString(diag22_coord).buffer(1.5)])
        if line21_fig.intersects(line22_fig):
            diag2_coord = np.array([diag21_coord[0], diag22_coord[1]])
            label2 = 3
            self.drawing = cv2.line(self.drawing, tuple(diag2_coord[0]), tuple(diag2_coord[1]), (0, 0, 255), 2)
            self.drawing = cv2.circle(self.drawing, tuple(diag2_coord[0]), 15, (255, 0, 0), 2)
            self.drawing = cv2.circle(self.drawing, tuple(diag2_coord[1]), 15, (255, 0, 0), 2)
        else:
            print('PATTERN1: diagonale2 non continua')
            self.drawing = cv2.line(self.drawing, tuple(diag21_coord[0]), tuple(diag21_coord[1]), (0, 0, 255), 2)
            self.drawing = cv2.circle(self.drawing, tuple(diag21_coord[0]), 15, (255, 0, 0), 2)
            self.drawing = cv2.circle(self.drawing, tuple(diag21_coord[1]), 15, (255, 0, 0), 2)
            self.drawing = cv2.line(self.drawing, tuple(diag22_coord[0]), tuple(diag22_coord[1]), (0, 0, 255), 2)
            self.drawing = cv2.circle(self.drawing, tuple(diag22_coord[0]), 15, (255, 0, 0), 2)
            self.drawing = cv2.circle(self.drawing, tuple(diag22_coord[1]), 15, (255, 0, 0), 2)
            label2 = 1
    elif label21 == 0 and label21 == label22:
        label2 = 0
    else:
        label2 = 1

    # Generate final score
    if label1 == 3 and label1 == label2: 
        # if both diagonals are correct, create geometrical forms for the points in the extremes    
        lines_point_or1 = [Point(tuple(diag1_coord[0])).buffer(15), Point(tuple(diag1_coord[1])).buffer(15)]
        lines_point_or2 = [Point(tuple(diag2_coord[0])).buffer(15), Point(tuple(diag2_coord[1])).buffer(15)]
        if ret_fig is not None:
          # if I already have a figure, use it
          rect_v = ret_fig[1]
        else:
          # else build it by creating a buffer around the points of the big rectangle that has the diagonals
          rect_v = [Point(x).buffer(15) for x in p_dst]
        # first diag intersects with the top left point of the rectangle
        p1 = lines_point_or1[0].intersects(rect_v[0])
        if not p1:
            print('PATTERN1: angolo alto a sx wrong')
        
        # second diag intersects with the top right point of the rectangle
        p2 = lines_point_or2[0].intersects(rect_v[1])
        if not p2:
            print('PATTERN1: angolo alto a dx wrong')

        # first diag intersects with the bottom right point of the rectangle
        p3 = lines_point_or1[1].intersects(rect_v[2])
        if not p3:
            print('PATTERN1: angolo basso a dx wrong')
          
        # second diag intersects with the bottom left point of the rectangle
        p4 = lines_point_or2[1].intersects(rect_v[3])
        if not p4:
            print('PATTERN1: angolo basso a sx wrong')

        # if all of them intersect with the rectangle, then the figure gains score 3
        if p1 and p2 and p3 and p4:
          label_diag_line = 3
        else:
          #if not, it means it exists, but it's not in the correct place => score 2
          label_diag_line = 2
    
    #if both diagonals are 0 => both are absent
    elif label1 == 0 and label1 == label2:
      print('PATTERN1: diagonali assenti')
      label_diag_line = 0
    else:
      # a diagonal is missing => distorted
      print('PATTERN1: una diagonale mancante')
      label_diag_line = 1     
    # self.drawing = the generated drawing based on the image
    # label_diag_line = score; diag1_coord, diag2_coord = coords of the diagonals
    return self.drawing, label_diag_line, diag1_coord, diag2_coord
    
