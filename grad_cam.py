import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.python.framework import ops

from skimage import exposure

keras = tf.keras
K = keras.backend

tf.compat.v1.disable_eager_execution()

class Visualization():
    def __init__(self, model, image_array, processed_input_array):
        self.model = model
        self.image_array = image_array
        self.processed_input_array = processed_input_array
        self.hmap_image_holder = None
        self.hmap_holder = None
        self.labels = []
        self.df_indicies = []
        self.type_or_name = []

    def reset_holders(self):
        self.hmap_image_holder = np.empty((0, 224, 224, 3))
        self.hmap_holder = np.empty((0, 224, 224))
        self.labels = []
        self.df_indicies = []
        self.type_or_name = []

    def grad_cam(self, img_nums, c_out_classes, df_indexes, type_or_names):
        # imgs_nums based on image_array
        # df_indices from df.  Get from data_get indices.
        # type_or_names just for information purposes
        model = self.model
        processed_input_array = self.processed_input_array
        image_array = self.image_array

        for img_num, c_out_class, df_index, type_or_name in zip(img_nums, c_out_classes, df_indexes, type_or_names):
            y_c = model.output[..., c_out_class]  # get output for class
            conv_output = model.get_layer('relu').output
            _, hmap_h, hmap_w, n_featuremaps = conv_output.shape
            grads = K.gradients(y_c, conv_output)[0]  # output is initially a list
            hmap_gradient_function = K.function([model.input], [conv_output, grads])

            c_preprocessed_input = processed_input_array[0][img_num]
            output, grads_val = hmap_gradient_function([np.expand_dims(c_preprocessed_input, axis=0)])
            output, grads_val = output[0, :], grads_val[0, ...] # get rid of 0 axis

            weights = np.mean(grads_val, axis=(0, 1)) # mean weight for each featuremap

            heat_map = np.zeros((hmap_h, hmap_w))
            for i in range(weights.shape[0]):
                heat_map += weights[i] * output[..., i]

            # Alternative dot product instead of for-loop
            # heat_map = np.dot(output, weights)

            heat_map = np.maximum(heat_map, 0) # relu

            # heat_map = heat_map / heat_map.max() # optional scaling

            _, img_h, img_w, img_c = image_array.shape

            if self.hmap_holder is None: # initialize based on conv_output
                self.hmap_image_holder = np.empty((0, img_h, img_w, img_c))
                self.hmap_holder = np.empty((0, hmap_h, hmap_w))

            heat_map = np.expand_dims(heat_map, axis=0)
            c_img = np.expand_dims(image_array[img_num], axis=0)

            self.hmap_image_holder = np.concatenate((self.hmap_image_holder, c_img), axis=0)
            self.hmap_holder = np.concatenate((self.hmap_holder, heat_map), axis=0)
            self.df_indicies.append(df_index)
            self.labels.append(c_out_class)
            self.type_or_name.append(type_or_name)

    def save_it(self, cname):
        np.savez(cname, imgs=self.hmap_image_holder, hmaps=self.hmap_holder,
                 labels=self.labels, df_index=self.df_indicies, type_or_name=self.type_or_name)
