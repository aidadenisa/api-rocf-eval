import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from shapely.geometry import Polygon, Point, LinearRing
from scipy.spatial import distance

from preprocessing import homography
from prediction import image_processing as imgProcess
from prediction.image_processing import thick_rect

### INITIALIZATIONS

#original coordinates of the external rectangle
p_dst = [(382, 219), (852, 219), (852, 537), (382, 537)]
#the padding that we consider from the original rectangle
pad_ext = 30
pad_int = 30

# TODO: explain
def diag_eq(diag, x):
    return int(diag[0][1] + ((diag[1][1]-diag[0][1])/(diag[1][0]-diag[0][0]))*(x-diag[0][0]))

#building the external rectangle 
external = [(p_dst[0][0]-pad_ext, diag_eq([p_dst[0], p_dst[2]], p_dst[0][0]-pad_ext)), (p_dst[1][0]+pad_ext, diag_eq([p_dst[1], p_dst[3]], p_dst[1][0]+pad_ext)),
     (p_dst[2][0]+pad_ext, diag_eq([p_dst[0], p_dst[2]], p_dst[2][0]+pad_ext)), (p_dst[3][0]-pad_ext, diag_eq([p_dst[1], p_dst[3]],p_dst[3][0]-pad_ext))]
#building the internal rectangle
internal = [(p_dst[0][0]+pad_int, diag_eq([p_dst[0], p_dst[2]],p_dst[0][0]+pad_int)), (p_dst[1][0]-pad_int, diag_eq([p_dst[1], p_dst[3]], p_dst[1][0]-pad_int)),
          (p_dst[2][0]-pad_int, diag_eq([p_dst[0], p_dst[2]], p_dst[2][0]-pad_int)), (p_dst[3][0]+pad_int, diag_eq([p_dst[1], p_dst[3]], p_dst[3][0]+pad_int))]

def buildROIs(p_dst): 
    pad_ext = 30
    pad_int = 30

    #building the external rectangle 
    external = [
        [p_dst[0][0]-pad_ext, diag_eq([p_dst[0], p_dst[2]], p_dst[0][0]-pad_ext)],
        [p_dst[1][0]+pad_ext, diag_eq([p_dst[1], p_dst[3]], p_dst[1][0]+pad_ext)],
        [p_dst[2][0]+pad_ext, diag_eq([p_dst[0], p_dst[2]], p_dst[2][0]+pad_ext)],
        [p_dst[3][0]-pad_ext, diag_eq([p_dst[1], p_dst[3]],p_dst[3][0]-pad_ext)]
    ]
    #building the internal rectangle
    internal = [
        [p_dst[0][0]+pad_int, diag_eq([p_dst[0], p_dst[2]],p_dst[0][0]+pad_int)], 
        [p_dst[1][0]-pad_int, diag_eq([p_dst[1], p_dst[3]], p_dst[1][0]-pad_int)],
        [p_dst[2][0]-pad_int, diag_eq([p_dst[0], p_dst[2]], p_dst[2][0]-pad_int)], 
        [p_dst[3][0]+pad_int, diag_eq([p_dst[1], p_dst[3]], p_dst[3][0]+pad_int)]
    ]

    return [external, internal]

###FUNCTIONS

# order the points in the order: top-left, top-right, bottom-right, bottom-left
def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    D = distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    return np.array([tl, tr, br, bl])

def get_background_new(background, pad_new, external, side_idx):
    #initialize a new background
    background_new = np.zeros_like(background)
    # check which side are you considering for copying 
    # depending on the side, you would copyparts of the original background into the new background; the pad will change and the method will offer different results, incrementally
    if side_idx == 0:
        background_new[external[0][1] + pad_new[0]:external[3][1] + 1, external[0][0]:external[1][0] + 1] = background[external[0][1] + pad_new[0]:external[3][1] + 1, external[0][0]:external[1][0] + 1]
    elif side_idx == 1:
        background_new[external[0][1] + pad_new[0]:external[3][1] + 1 - pad_new[1], external[0][0]:external[1][0] + 1] = background[external[0][1] + pad_new[0]:external[3][1] + 1 - pad_new[1], external[0][0]:external[1][0] + 1]
    elif side_idx == 2:
        background_new[external[0][1] + pad_new[0]:external[3][1] + 1 - pad_new[1], external[0][0] + pad_new[2]:external[1][0] + 1 ] = background[external[0][1] + pad_new[0]:external[3][1] + 1 - pad_new[1], external[0][0] + pad_new[2]:external[1][0] + 1]
    elif side_idx == 3:
        background_new[external[0][1] + pad_new[0]:external[3][1] + 1 - pad_new[1], external[0][0] + pad_new[2]:external[1][0] + 1 - pad_new[3]] = background[external[0][1] + pad_new[0]:external[3][1] + 1 - pad_new[1],external[0][0] + pad_new[2]:external[1][0] + 1 - pad_new[3]]    
    # plt.imshow(background, cmap='gray')
    # plt.show()
    # plt.imshow(background_new, cmap='gray')
    # plt.show()
    cnts_new, hier_new = cv2.findContours(background_new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return background_new, cnts_new, hier_new

def get_pixel_sum(background, external, pad_new, internal):
  background_new = np.zeros_like(background)
  # copy the external and pad pixels from background to new background
  background_new[external[0][1] + pad_new[0]:external[3][1] + 1 - pad_new[1],
        external[0][0] + pad_new[2]:external[1][0] + 1 - pad_new[3]] = background[external[0][1] + pad_new[0]
        :external[3][1] + 1 - pad_new[1],external[0][0] + pad_new[2]:external[1][0] + 1 - pad_new[3]]    
  # create an internal black rectangle 
  background_new = cv2.rectangle(background_new, internal[0], internal[2], 0, -1)
  #   plt.imshow(background_new, cmap='gray')
  #   plt.show()
  # return the sum of pixels
  return np.sum(background_new)

# Hough Line Transform - Probabilistically detect the presence of a line
def best_line(background, drawing=None, oriz=True):
    # detects the presence of lines using probabilistic Hough Transform
    # returns a vector that will store the parameters (xstart,ystart,xend,yend) of the detected lines
    lines_filtered = cv2.HoughLinesP(background, 1, np.pi / 180, 50, None, 40, 20)
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
            # append in points a tuple formed by the coords of the start point of the line    
            points.append((l[0], l[1]))
            # if we have to detect an horizontal line
            if oriz:
                # if the Xstart coord. of the point identified is smaller than the max left, then max_left becomes the new point
                if l[0] < max_left:
                    max_left = l[0]
                # if the X coord. of the point identified is bigger than the max right, then max_right becomes the new point
                if l[0] > max_right:
                    max_right = l[0]
            # else, if we are detecting a vertical line
            else:
                # if the Ystart is smaller than max_left, then max_left becomes the coord.
                if l[1] < max_left:
                    max_left = l[1]
                # if the Ystart is bigger than max_right, then max_right becomes the coord.
                if l[1] > max_right:
                    max_right = l[1]
            # append in points a tuple formed by the coords of the end point of the line    
            points.append((l[2], l[3]))
            # do the same reasoning here as above
            if oriz:
                if l[2] < max_left:
                    max_left = l[2]
                if l[2] > max_right:
                    max_right = l[2]
            else:
                if l[3] < max_left:
                    max_left = l[3]
                if l[3] > max_right:
                    max_right = l[3]
        #if the length of points array is bigger than 0
        if len(points) > 0:
          # fit a line between all the points in the points array
          # line	=	cv.fitLine(	points, distType, param, reps, aeps[, line]	)
          # (vx, vy, x0, y0), where (vx, vy) is a normalized vector collinear to the line and (x0, y0) is a point on the line.
          [vx, vy, x, y] = cv2.fitLine(np.array(points), cv2.DIST_L2, 0, 0.01, 0.01)
          # if the line is horizontal, 
          if oriz:
              # TODO: ASK
              t0 = (max_left - x) / vx
              t1 = (max_right - x) / vx
              lefty = int(y + t0 * vy)
              righty = int(y + t1 * vy)
              drawing = cv2.line(drawing, (max_left, lefty), (max_right, righty), (0, 0, 255), 2, cv2.LINE_AA)
              return (max_left, lefty), (max_right, righty)
          else:
              # TODO: ASK
              t0 = (max_left - y) / vy
              t1 = (max_right - y) / vy
              lefty = int(x + t0 * vx)
              righty = int(x + t1 * vx)
              drawing = cv2.line(drawing, (lefty, max_left), (righty, max_right), (0, 0, 255), 2, cv2.LINE_AA)
              return (lefty, max_left), (righty, max_right)
    return None

class Pattern2:
  def __init__(self, img, drawing, model, scaler):
    # the image we have to analyze 
    self.img = img    
    # the 0-initialized drawing 3d structure
    self.drawing = drawing
    self.roi = buildROIs(p_dst)
    # SVM model and scaler, pretrained, used for detecting the presence of the shape from the number of non-white pixels; this is used when the polygon was not detected
    self.model = model
    self.scaler = scaler
  
  def get_score(self):    
    rect_ext = None
    # extract the drawing, apply thresholding, remove the unwanted background, identify contours
    # background = extracted and thresholded drawing
    # cnts_ret = the contours that were found
    # hier = the hierarchy of the contours; for each contour, we have the structure [Next, Previous, First Child, Parent]
    background, cnts_ret, hier = imgProcess.getBackground(external, self.img, True, True, internal)

    # for each contour, if it has no parent and if the length of the contour is bigger than 500, extract it.
    ext = [i for i in range(hier[0].shape[0]) if hier[0][i][3] == -1 and cv2.arcLength(cnts_ret[i], True) > 500]
    # initialize with 4 0s
    pad_new = np.zeros(4).astype(np.uint8)
    # if there was only 1 contour that respected the rules of extraction
    if len(ext) == 1 and ext[0]:
        # extract it in variable c
        c = ext[0]

        # calculate the perimeter
        peri = cv2.arcLength(cnts_ret[c], True)   
        # we approximate a polygon for the contour, by ensuring it's a closed figure, and its maximum distance from the original curve
        # the result is a list of vertices of the approximate polygon 
        approx = cv2.approxPolyDP(cnts_ret[c], 0.02*peri, True)
        # print('peri: {}, vert: {}'.format(peri, len(approx)))  

        #if the approximate polygon had 4 vertices, and the contour area of the contour area is bigger than 100000
        if len(approx) == 4 and cv2.contourArea(cnts_ret[c]) > 100000:   
          # if the vertices are 4 => they create 4 sides, that we want to itterate on
          for side_idx in range(4):
            # as long as the length of the vertices is 4
            while len(approx) == 4:
                # create a background copy and generate its new contours and hierarchy
                _, cnts_new, hier_new = get_background_new(background, pad_new, external, side_idx)
                # extract the contours that have no parent and have the arcLength greater than 500
                ext = [i for i in range(hier_new[0].shape[0]) if
                      hier_new[0][i][3] == -1 and cv2.arcLength(cnts_new[i], True) > 500]
                # if there is only 1 contour which satisfies the conditions
                if len(ext) == 1:
                    # calculate the perimeter
                    peri = cv2.arcLength(cnts_new[ext[0]], True)
                    # if the length of the list of vertices that approximates the new polygon is still 4, and the contour area is bigger then 100000
                    if len(cv2.approxPolyDP(cnts_new[ext[0]], 0.02 * peri, True)) == 4 and cv2.contourArea(cnts_new[ext[0]]) > 100000:
                        # set approx as the new polygon
                        approx = cv2.approxPolyDP(cnts_new[ext[0]], 0.02 * peri, True)
                        # set the contours as the new contours
                        cnts_ret = cnts_new
                        # set c as the new cotour hierarchy
                        c = ext[0]
                        # the pad of this specific size increases with 1
                        pad_new[side_idx] += 1
                    else:
                        # otherwise remove 1 from the pad
                        pad_new[side_idx] -= 1
                        break
                else:
                    # if there is no contour that satisfies the conditions, remove 1 from the pad of he side
                    pad_new[side_idx] -= 1
                    break    
          # adds the countours of the lines in the drawing, with green
          self.drawing = imgProcess.draw_contours(self.drawing, cnts_ret[c])       
          # order points from the approximation of the polygon in the order: top-left, top-right, bottom-right, bottom-left
          rect_ext = order_points(np.array([coord[0] for coord in approx]))
    # get the sum of pixels of the drawing
    pixel_rect = get_pixel_sum(background, external, pad_new, internal)
    # using SVM, predict if the drawing is present, based on the number of non-white pixels
    rect_prediction = self.model.predict(self.scaler.transform(np.array([pixel_rect]).reshape(-1,1)))
    # if a polygon was extracted
    if rect_ext is not None:
        thick = 30
        pad_line = 100
        # get a rectangle that is the ROI in which the first (top) line can be found
        line1 = thick_rect([(rect_ext[0][0], rect_ext[0][1]), (rect_ext[1][0]+pad_line, rect_ext[1][1])], thick)
        # get the drawing inside the ROI defined above, from the image, and generate the background and conture for it
        background, cnt = imgProcess.getBackground(line1, self.img)
        # identify the best approximation of the first horizontal line
        # returns (max_left, lefty), (max_right, righty)
        line1_v = best_line(background, self.drawing, True)
        # create a list with 2 circles around the 2 points that define the line above
        # print("line1: " , line1_v)
        l1 = [Point(line1_v[0]).buffer(30), Point(line1_v[1]).buffer(30)]

        # do the same as for the line above to the rest of the lines
        line2 = thick_rect([(rect_ext[1][0], rect_ext[1][1]-pad_line), (rect_ext[2][0], rect_ext[2][1]+pad_line)], thick)
        background, cnt = imgProcess.getBackground(line2, self.img)
        line2_v = best_line(background, self.drawing, False)
        l2 = [Point(line2_v[0]).buffer(30), Point(line2_v[1]).buffer(30)]

        line3 = thick_rect([(rect_ext[2][0]+pad_line, rect_ext[2][1]), (rect_ext[3][0]-pad_line, rect_ext[3][1])], thick)
        background, cnt = imgProcess.getBackground(line3, self.img)
        line3_v = best_line(background, self.drawing, True)
        l3 = [Point(line3_v[0]).buffer(30), Point(line3_v[1]).buffer(30)]

        line4 = thick_rect([(rect_ext[0][0], rect_ext[0][1]-pad_line), (rect_ext[3][0], rect_ext[3][1])], thick)
        background, cnt = imgProcess.getBackground(line4, self.img)
        line4_v = best_line(background, self.drawing, False)
        l4 = [Point(line4_v[0]).buffer(30), Point(line4_v[1]).buffer(30)]

        # add with red the approximated polygon to the drawing
        self.drawing = cv2.polylines(self.drawing, [rect_ext.reshape((-1, 1, 2))], True, (255, 0, 0), 2)
        # for each point in the approxymated polygon 
        for coord in rect_ext:
          # add a circle around each point and color it with red
          self.drawing = cv2.circle(self.drawing, tuple(coord), 20, (255, 0, 0), 2)
        # create a polygon shape from the extracted coordinates, with a buffer of 1.5
        # print('rect_ext' , rect_ext)

        ret = Polygon(rect_ext).buffer(1.5)
        # print('ret' , ret)

        # create a list of points with a buffer of 20px for all the point in the extracted polygon 
        ret_vertices = [Point(point).buffer(20) for point in rect_ext]
        # print('ret' , ret_vertices)

        # update the roi
        self.roi = buildROIs(rect_ext)
           
        # if the circles created by the buffer around the points of the coordinates of each line are intersecting with the correct point from the next line
        if l1[1].intersects(l2[0]) and l2[1].intersects(l3[1]) and l4[1].intersects(l3[0]) and l4[0].intersects(l1[0]):
          # then offer the full score
          label_rect = 3
        else:
          bigDistances = np.array([l1[1].intersects(l2[0]),l2[1].intersects(l3[1]),l4[1].intersects(l3[0]), l4[0].intersects(l1[0])]) > 35
          if len(bigDistances[bigDistances == True]) == 0 :
            label_rect = 3
          else: 
            print('PATTERN2: the lines are not precise')
            # else, the line is not precise, so give 1 
            label_rect = 1
        return self.drawing, label_rect, (ret, ret_vertices)       
    # else, if we were not able to extract a polygon, 
    else:
      # check if, based on the number of non-white pixels, the SVM predicted that there is a polygon in the drawing
      if rect_prediction[0] == 1:
        # if this was predicted, then it means that the shape is distorted, so assign a fewer value
        label_rect = 1
      else:
        # shape is absent
        label_rect = 0
    return self.drawing, label_rect, None

  def get_ROI(self):
    return self.roi




