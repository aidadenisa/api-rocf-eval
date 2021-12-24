import cv2
import numpy as np
from skimage.morphology import skeletonize

from preprocessing import homography

def getBackground(external, img, morph=True, ret_hier=False, internal=None):
    #initialize the background with 0s
    background = np.zeros_like(img)

    #get the points of the external ROI and reshape them for the fillConvexPoly function
    points = np.array([external]).reshape((4, 1, 2))
    #create a fill for the shape that is determined by the points, and fill it with the color white
    background = cv2.fillConvexPoly(background, points, (255, 255, 255))
    # create a mask with true outside the ROI (opposite of current bg)
    not_background = cv2.bitwise_not(background)
    #take from the image the content that is inside the mask formed by the background array, which has white values just in the shape of the external ROI
    background = cv2.bitwise_and(img, background)   
    #create an image that is white on the mask outside the ROI
    if internal is not None:
      #get the points representing the internal ROI and reshape them for the fillConvexPoly function
      int_points = np.array([internal]).reshape((4, 1, 2))
      #fill in the shape created by the interior points with white color
      background = cv2.fillConvexPoly(background, int_points, (255, 255, 255))
    '''overlap = cv2.polylines(cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB), [points], True, (255, 0, 0), 1)    
    plt.imshow(overlap)
    plt.show()'''
    # extract the drawing in a sharper version, by using unimodal thresholding
    # background = homography.sharpenDrawing(background)

    #all the black (0) margins outside of the ROI have to be turned to white (255)
    # equivalent of the (background[background == 0] = 255) from before
    background = cv2.bitwise_or(not_background,background)

    if morph:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        # background = cv2.bitwise_not(background)
        # doing erosion on mostly white background drawing has the opposite effect of erosion => the drawing becomes thicker
        # it is inverted by doing a bitwise_not, which will do a logical not operation
        background = cv2.bitwise_not(cv2.erode(background, kernel))
         # scheletonize the currently thick drawing (Skeletonization reduces binary objects to 1 pixel wide representations. This can be useful for feature extraction, and/or representing an object’s topology.)
        background = skeletonize(background / 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
         # dilate the (white) skeleton drawing (on black background) by a kernel of 3x3px => add a bit of thickness
        background = cv2.dilate(background, kernel)
    else:
         # skipping the thickening of the drawing artificially; this is not recommended because we may have some gaps in the lines after thresholding
        background = cv2.bitwise_not(background)
        background = skeletonize(background / 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        background = cv2.dilate(background, kernel)
        
    #find the contours and their hierarchy using opencv
    cnts, hier = cv2.findContours(background, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if ret_hier:
        return background, cnts, hier
    else:
        return background, cnts

# adds the countours of the lines in the drawing
def draw_contours(drawing, contour):
    # plt.imshow(drawing)
    # plt.show()
    # create a black image in the same shape as drawing
    temp = np.zeros_like(drawing)
    #identify where the drawing image is red and create a mask
    red_mask = np.all(drawing == [255, 0, 0], axis=-1)
    # where the drawing is red, save the same in the temp  
    temp[red_mask] = drawing[red_mask]
    # draw contours with green
    drawing = cv2.drawContours(drawing, contour, -1, (0, 255, 0), 2)
    # plt.imshow(drawing)
    # plt.show()
    drawing[red_mask] = temp[red_mask]
    # plt.imshow(drawing)
    # plt.show()
    return drawing

# construct a ROI around the diagonal, with a buffer of W
def thick_rect(diag, W):
    # distance between diagonal's x-coords
    Dx = diag[1][0] - diag[0][0]
    # distance between diagonal's y-coords
    Dy = diag[1][1] - diag[0][1]
    # calculate the distance between the 2 points of the diagonal ( --- sqrt((x2-x1)^2 + (y2-y1)^2) --- )
    D = np.sqrt(Dx * Dx + Dy * Dy)
    #TODO: why do we calculate it like this?
    Dx = int(0.5 * W * Dx / D)
    Dy = int(0.5 * W * Dy / D)
    #return the rectangle formed around the diagonal line
    return [(diag[0][0] - Dy, diag[0][1] + Dx), (diag[1][0] - Dy, diag[1][1] + Dx), (diag[1][0] + Dy, diag[1][1] - Dx),
            (diag[0][0] + Dy, diag[0][1] - Dx)]