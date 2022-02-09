import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import backend as T
import tempfile

from prediction import Grid
from prediction.model_storage import getKerasModel

root='./'
model_folder=root + 'models'
result_folder=root + 'results'
template_dic={
    0:'best_model_triplet_cross_transfer.hdf5',
    1:'best_model_triplet_face_transfer.hdf5',
    2:'best_model_triplet_rail_transfer.hdf5',
    3:'best_model_triplet_rombo_transfer.hdf5',
    4:'best_model_triplet_rett_diag_transfer.hdf5',
    5:'best_model_triplet_rect_transfer.hdf5',
    6:'best_model_triplet_cross_vert_transfer.hdf5'
} 

def loadModels(template_dic):
    models = []
    for temp in template_dic:
        m = getKerasModel(template_dic[temp])
        models.append(m)

    return models

# models = loadModels(template_dic)

class Visualization():
    def __init__(self, name, image, points, templates, shape, writer):
        self.img = image
        self.name = name
        self.grids=[]
        for point in points.values():
            self.grids.append(Grid.Grid(point))
        self.templates = templates
        self.input_shape = shape
        self.scores=[]
        self.distances=[]
        self.rects=[]
        self.embeddings=[]
        self.writer = writer
 
    def run(self):
        if not os.path.isdir(os.path.join(result_folder, self.name)):
            os.makedirs(os.path.join(result_folder, self.name))
        for i in range(len(self.templates)):
                   
            #template = np.reshape(self.templates[i], self.input_shape)
            template = self.templates[i]
            template =  np.repeat(template[..., np.newaxis], 3, -1)
            self.model = getKerasModel(template_dic[i])
            # self.model = models[i]

            #plot_model(self.model, show_shapes=True)
            max_val, distance, rect, embedding = self.grids[i].visualize(self.img, self.model, self.input_shape, template, i, self.name)
            self.scores.append(max_val)
            self.distances.append(distance)
            self.rects.append(rect)
            self.embeddings.append(embedding)
            T.clear_session()
            del self.model          
        
        result = {'names':self.name, 'scores': self.scores, 'distances':self.distances, 'rect': self.rects, 'embeddings': self.embeddings}

        return result
