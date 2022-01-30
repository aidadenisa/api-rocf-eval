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
        save_min = False
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
            #plt.imshow(input_img[:,:,0], cmap='gray')            
            #plt.show()
            #print(input_img.shape)
            #print(template.shape)
            #inp = np.array([[input_img], [template]])
            #print(inp.shape)
            
            # send image and template to model for prediction
            result = model.predict([[input_img.reshape(1,100,100,3)], [template.reshape(1,100,100,3)]])
            embeddings = result
            # calculate the pairwise distance between all embeddings
            result = triplet_loss._pairwise_distances(embeddings, squared=False).numpy()[0, 0]
            # save the minimum distance
            if result < min_val:
                min_val = result
                save_min = True
                min_y = y
                min_x = x
                min_input = input_img
                #min_bbox = bbox
            #if i in values:
              #print('done percent {} of template {}'.format(10*np.where(values == i)[0][0], idx))
        
        done = False
        min_w = self.w
        min_h = self.h
        min_x2 = min_x
        min_y2 = min_y
        iteraction = 0
        while not done and iteraction < 5:
          found_min = False
          for action in self.actions:
              (x, y, w, h) = action(min_x, min_y, min_w, min_h)
              ROI = image[y:y + h, x:x + w]
              # threshed = np.array(homography.sharpenDrawing(ROI))
              ROI_expanded = homography.expandDrawing(ROI)
            
              input_img = homography.background_thumbnail(ROI_expanded, 'L',
                                          (input_shape[0], input_shape[1]))
              input_img = input_img.astype('float32')
              input_img /= 255
              input_img =  np.repeat(input_img[..., np.newaxis], 3, -1)
              #plt.imshow(input_img[:,:,0], cmap='gray')
              #plt.show()
              result = model.predict([[input_img.reshape(1,100,100,3)], [template.reshape(1,100,100,3)]])
              embeddings = result
              result = triplet_loss._pairwise_distances(embeddings, squared=False).numpy()[0, 0]
              if result < min_val:
                  min_val = result
                  min_x2 = x
                  min_y2 = y
                  min_w = w
                  min_h = h
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
        
        #print(min_bbox)
        clone2 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(clone2, (min_x2, min_y2), (min_x2 + min_w, min_y2 + min_h), color=(255, 0, 0))        
        cv2.putText(clone2, str(min_val), (x + 20, y + 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
          color=(0, 0, 255))                
        cv2.rectangle(clone2, (self.x, self.y), (self.x+self.w, self.y+self.h), color=(0,0,255))
        #cv2.rectangle(clone2, (min_x+min_bbox[0], min_y+min_bbox[1]), (min_x+min_bbox[0]+min_bbox[2], min_y+min_bbox[1]+min_bbox[3]), color=(0,255,0))
        # plt.imshow(cv2.cvtColor(clone2, cv2.COLOR_BGR2RGB))
        # plt.show() 
        #plt.close('all')           
        cv2.imwrite(os.path.join(result_folder, name, 'minimum_'+str(idx)+'.png'), clone2)
        # fig, ax = plt.subplots(nrows=1, ncols=2)
        # ax.ravel()[0].imshow(min_input[:,:,0], cmap='gray')
        # ax.ravel()[1].imshow(template[:,:,0], cmap='gray')
        # # plt.savefig(os.path.join(result_folder, name, 'input_'+str(idx)+'.png'))
        #plt.show()
        # plt.close('all')    
        print('done template {}'.format(idx))    
        return min_val, math.hypot(int(min_x2-min_w/2-(self.x-self.w/2)), int(min_y2-min_h/2-(self.y-self.h/2))), (min_x2, min_y2, min_w, min_h)