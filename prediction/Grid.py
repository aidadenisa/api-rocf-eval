import numpy as np
import cv2
import os
import math

from preprocessing import homography
from prediction import triplet_loss

root='./'
model_folder=root + 'models'
result_folder=root + 'results'

class Grid():
    def __init__(self, coords):
        self.x = coords[0] 
        self.y = coords[1] 
        self.w = np.abs(coords[0] - coords[2])
        self.h = np.abs(coords[1] - coords[3])
        pad = 50
        step = 10
        row =  np.arange(self.x-pad, self.x+pad+1, step=step, dtype=int)
        column = np.arange(self.y-pad, self.y+pad+1, step=step, dtype=int)
        self.grid = np.transpose([np.tile(row, len(column)), np.repeat(column, len(row))])
        pad_a = 10
        self.actions = [
            lambda x, y, w, h: (x, y, w + pad_a, h),
            lambda x, y, w, h: (x, y, w, h + pad_a),
            lambda x, y, w, h: (x, y - pad_a, w, h),
            lambda x, y, w, h: (x - pad_a, y, w, h),
            lambda x, y, w, h: (x - pad_a, y - pad_a, w + pad_a, h + pad_a),
            lambda x, y, w, h: (x - pad_a, y, w + pad_a, h + pad_a),
            lambda x, y, w, h: (x, y - pad_a, w + pad_a, h + pad_a),
            lambda x, y, w, h: (x, y, w + pad_a, h + pad_a),
        ]
        

    
    def visualize(self, image, model, input_shape, template, idx, name):               
        min_val = np.inf
        min_embeddings = []
        save_min = False

        dataset_images = []

        for i in range(len(self.grid)):                     
            x, y = self.grid[i]
            x = max(0, x)
            y = max(0, y)
            ROI = image[y:y + self.h, x:x + self.w]
            # threshed =  np.array(homography.sharpenDrawing(ROI))
            # ROI_expanded = homography.expandDrawing(ROI)

            input_img = homography.background_thumbnail(ROI, 'L', (input_shape[0], input_shape[1]))
            input_img = input_img.astype('float32')
            input_img /= 255
            input_img =  np.repeat(input_img[..., np.newaxis], 3, -1)            

            dataset_images.append(input_img)
            
        dataset_anchors = [template for i in range(len(dataset_images))]
        dataset_images = np.array(dataset_images)
        dataset_anchors = np.array(dataset_anchors)
        
        result = model.predict([dataset_images, dataset_anchors], batch_size=32)

        for i in range(len(result)):
            distance = triplet_loss._pairwise_distances(np.array([result[i]]), squared=False).numpy()[0, 0]
            if distance < min_val:
                min_val = distance
                min_embeddings = result[i]
                save_min = True
                x, y = self.grid[i]
                min_x = max(0, x)
                min_y = max(0, y)
                
        done = False
        min_w = self.w
        min_h = self.h
        min_x2 = min_x
        min_y2 = min_y
        iteraction = 0

        while not done and iteraction < 5:
          found_min = False
          dataset_images = []
          dataset_anchors = []
          coords_actions = []
          for action in self.actions:
              (x, y, w, h) = action(min_x, min_y, min_w, min_h)
              coords_actions.append([x, y, w, h])
              ROI = image[y:y + h, x:x + w]
              # threshed = np.array(homography.sharpenDrawing(ROI))
              ROI_expanded = homography.expandDrawing(ROI)
            
              input_img = homography.background_thumbnail(ROI_expanded, 'L',
                                          (input_shape[0], input_shape[1]))
              input_img = input_img.astype('float32')
              input_img /= 255
              input_img =  np.repeat(input_img[..., np.newaxis], 3, -1)

              dataset_images.append(input_img)

          dataset_anchors = [template for i in range(len(dataset_images))]
          dataset_images = np.array(dataset_images)
          dataset_anchors = np.array(dataset_anchors)
        
          result = model.predict([dataset_images, dataset_anchors], batch_size=32)

          for i in range(len(result)):
            distance = triplet_loss._pairwise_distances(np.array([result[i]]), squared=False).numpy()[0, 0]
            if distance < min_val:
                min_val = distance
                min_embeddings = result[i]
                # (x, y, w, h)
                min_x2 = coords_actions[i][0]
                min_y2 = coords_actions[i][1]
                min_w = coords_actions[i][2]
                min_h = coords_actions[i][3]
                min_input = input_img
                found_min = True
              
          if found_min:
            min_x = min_x2
            min_y = min_y2
            w = min_w
            h = min_h
          else:
            done = True
          iteraction += 1
          print('done iteration {}'.format(iteraction))
        
        print('done template {}'.format(idx))    
        return min_val, math.hypot(int(min_x2-min_w/2-(self.x-self.w/2)), int(min_y2-min_h/2-(self.y-self.h/2))), (min_x2, min_y2, min_w, min_h), min_embeddings