import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
import math
import matplotlib.pyplot as plt

from tensorflow.python.framework import ops

from skimage import exposure

keras = tf.keras
K = keras.backend

class Vis_tool():
    def __init__(self, model_path=None, val_path=None, out_class=None, layer_name=None):
        self.graph_one = tf.Graph()
        self.graph_two = tf.Graph()

        if model_path is None:
            self.hmap_model_path = None
            self.sal_model_path = None
        else:
            self.hmap_model_path = model_path
            self.sal_model_path = model_path

        if val_path is None:
            val_path = None
        else:
            val_path = val_path

        if out_class is None:
            self.out_class = 2
        else:
            self.out_class = out_class

        if layer_name is None:
            self.layer_name = 'relu'
        else:
            self.layer_name = layer_name

        self.final_img_size = (224, 224)

        image_array, id_array, label_array, orig_idx_array = read_npz_hotlabel(val_path)

        self.image_array = image_array
        self.label_array = label_array
        self.id_array = id_array

        self.hmap_switch = False
        self.salicency_switch = False

    def chose_images_on_labels(self, start=0, end=20):
        #Only in batches of 20 images

        label_array = self.label_array
        out_class = self.out_class
        image_array = self.image_array

        max_label = np.argmax(label_array, axis=1)
        label_bool = (max_label == out_class)
        choice_images = image_array[label_bool]

        self.choice_images = choice_images[start:end]
        self.choice_ids = self.id_array[label_bool][start:end]

    def create_hmap_model(self):
        if not self.hmap_switch:
            with self.graph_one.as_default():
                self.sess_one = tf.Session()
                with self.sess_one.as_default():
                    self.model = keras.models.load_model(self.hmap_model_path)

                    y_c = self.model.output[..., self.out_class] # get output for class
                    conv_output = self.model.get_layer(self.layer_name).output
                    grads = K.gradients(y_c, conv_output)[0] #output is initially a  list

                    self.hmap_gradient_function = K.function([self.model.input], [conv_output, grads])
                    self.hmap_switch = True

        else:
            print('hmap model already created')


    def make_hmaps(self):
        self.hmap_image_holder = np.empty((0, 224, 224, 3))
        self.hmap_holder = np.empty((0, 224, 224))

        with self.sess_one.as_default():
            for i in range(self.choice_images.shape[0]):
                one_train_img = self.choice_images[i]
                output, grads_val = self.hmap_gradient_function([np.expand_dims(one_train_img, axis=0)])
                output, grads_val = output[0, :], grads_val[0, ...] # get rid of 0 axis

                weights = np.mean(grads_val, axis=(0, 1)) # mean weight for each featuremap

                heat_map = np.zeros((7, 7))
                for i in range(weights.shape[0]):
                    heat_map += weights[i] * output[..., i]

                # Alternative dot product instead of for-loop
                # heat_map = np.dot(output, weights)

                heat_map = np.maximum(heat_map, 0) # relu

                # heat_map = heat_map / heat_map.max() # optional scaling

                hmap = cv2.resize(heat_map, self.final_img_size)
                hmap = np.expand_dims(hmap, axis=0)

                one_train_img = np.expand_dims(one_train_img, axis=0)
                self.hmap_image_holder = np.concatenate((self.hmap_image_holder, one_train_img), axis=0)

                self.hmap_holder = np.concatenate((self.hmap_holder, hmap), axis=0)

    def create_guided_model(self):
        if not self.salicency_switch:
            with self.graph_two.as_default():
                self.sess_two = tf.Session()
                with self.sess_two.as_default():
                    if "GuidedBackProp" not in ops._gradient_registry._registry:
                        @tf.RegisterGradient('GuidedBackProp')
                        def _GuidedBackProp(op, grad):
                            dtype = op.inputs[0].dtype
                            new_grad = grad * tf.cast(grad > 0., dtype) * tf.cast(op.inputs[0] > 0., dtype)
                            return new_grad

                    with self.graph_two.gradient_override_map({"Relu": "GuidedBackProp"}):
                        self.guided_model = keras.models.load_model(self.sal_model_path)

                    def loss_function(y_true, y_pred):
                        c_loss = tf.reduce_mean(y_pred[:, 2])
                        return c_loss

                    # guided_model.compile(optimizer='RMSprop', loss=loss_function)

                    c_input = self.guided_model.input
                    c_output = self.guided_model.output[..., self.out_class]  # for class 2

                    loss = tf.reduce_mean(c_output)
                    gradient = K.gradients(loss, c_input)[0]
                    self.sal_grad_func = K.function([c_input], [gradient])
            self.salicency_switch = True

        else:
            print("Guided/saliency model already created")

    def make_saliency_maps(self, learning_phase=0):
        self.sal_holder = np.empty((0, 224, 224, 3))

        with self.graph_two.as_default():
            with self.sess_two.as_default():
                K.set_learning_phase(learning_phase)
                grad_val = self.sal_grad_func([self.choice_images])

        maxed_grad_val = np.max(grad_val[0], axis=-1)
        self.abs_grad_val = np.abs(maxed_grad_val)
        self.pos_grad_val = np.maximum(maxed_grad_val, 0)
        self.neg_grad_val = np.where(maxed_grad_val < 0, np.abs(maxed_grad_val), 0)

    def guided_grad_cam(self):
        self.chose_images_on_labels()
        self.create_hmap_model()
        self.make_hmaps()
        self.create_guided_model()
        self.make_saliency_maps()

    def show_images(self):

        self.guided_grad_cam = self.hmap_holder * self.abs_grad_val

        # tile_alt_imshow(self.choice_images, heat_maps=self.guided_grad_cam, cmap='jet', alpha=0.7)

def check_model_layers(model, name_oi = 'relu'):
    for layer in model.layers:
        try:
            if layer.activation.__name__ == name_oi:
                print(layer.name)
        except: # for layers without an activation
            pass

def check_gradient():

    graph = tf.get_default_graph()
    sess = tf.Session(graph=graph)

    @tf.RegisterGradient("Custom")
    def _no_modifications(op, grad):
        global x
        global y
        x = op
        y = grad
        return tf.cast((grad > 0.), tf.float32) * tf.cast(op.inputs[0] > 0., tf.float32) * grad

    c_inp = tf.Variable((np.arange(10).reshape(5, 2) - 5), dtype=tf.float32)
    c_other = tf.Variable((np.arange(10).reshape(2, 5) - 3), dtype=tf.float32)

    with graph.gradient_override_map({"Relu": "Custom"}):
        c_output = tf.nn.relu(c_inp)
        c_output = tf.matmul(c_output, c_other)

    grad = tf.gradients(c_output, c_inp)

    sess.run(tf.global_variables_initializer())

    a, b, c = sess.run([c_inp, c_output, grad])

def create_and_save_image_maps():
    df = pd.read_excel('./old_xlsx/new_recolored_df.xlsx')
    df[df['ID'] == 11447]['Report Text'].values #to read the Report text of the ID

    vt = Vis_tool(out_class=0, val_path='./pelvis_only_224_test_hot_nonhardware.npz')

    vt.create_hmap_model()
    vt.create_guided_model()

    n_start = 0
    n_end = 16

    # repeat from here
    vt.chose_images_on_labels(start=n_start, end=n_end)
    vt.make_hmaps()
    vt.make_saliency_maps()
    guided_cam = vt.hmap_holder * vt.abs_grad_val

    c_heatmap = guided_cam
    # c_heatmap = vt.hmap_holder

    # tile_alt_imshow(vt.choice_images, heat_maps=c_heatmap, cmap='jet', titles=vt.choice_ids, alpha=0.7)

    n_start = n_start + 16
    n_end = n_end + 16

    image_holder = np.empty((0, 224, 224, 3))
    hmap_holder = np.empty((0, 224, 224))
    guided_holder = np.empty((0, 224, 224))
    id_holder = []

    # if you want to save the image
    c_choice_n = 7
    c_img = np.expand_dims(vt.choice_images[c_choice_n], axis=0)
    c_hmap = np.expand_dims(vt.hmap_holder[c_choice_n], axis=0)
    c_guided = np.expand_dims(guided_cam[c_choice_n], axis=0)

    image_holder = np.concatenate((image_holder, c_img), axis=0)
    hmap_holder = np.concatenate((hmap_holder, c_hmap), axis=0)
    guided_holder = np.concatenate((guided_holder, c_guided), axis=0)
    id_holder.append(vt.choice_ids[c_choice_n])

    np.savez('sep_label_cam_class_2222', image_array=image_holder, hmap_array=hmap_holder, guided_array=guided_holder, id_array=np.array(id_holder))

    # tile_alt_imshow(image_holder, heat_maps=guided_holder, cmap='jet', titles=id_holder, alpha=0.7)

def heatmap_custom_tile_imshow(img_arrays, heatmap_1, heatmap_2, titles=None, width=24, height=12, save_name=None,
                               cmap='jet', alpha_1=0.5, alpha_2 = 0.7, vmin=None, vmax=None, prob_array=None):


    if len(img_arrays.shape) == 4:
        img_n, img_h, img_w, _ = img_arrays.shape
    else:
        img_n, img_h, img_w = img_arrays.shape

    fig, axes = plt.subplots(3, 6, figsize=(width, height))
    fig.subplots_adjust(hspace=0, wspace=0)

    flat_axes = axes.flatten()
    n_range = np.arange(img_n)

    for ax, i in zip(flat_axes[:6], n_range):
        img = img_arrays[i,...,0]
        img = exposure.equalize_hist(img)

        c_img = ax.imshow(img, cmap='gray')

        if prob_array is not None:
            c_prob = prob_array[i].round(2)
            c_text = 'Normal: {0:.2f}\n' \
                     'Ant pelvis: {1:.2f}\n' \
                     'Post pelvis: {2:.2f}\n' \
                     'Prox femur: {3:.2f}\n' \
                     'Acetabular: {4:.2f}\n' \
                     'Complex: {5:.2f}'.format(c_prob[0], c_prob[1], c_prob[2], c_prob[4], c_prob[5], c_prob[3])
            ax.text(10, 70, c_text, fontsize=10, bbox={'facecolor': 'white', 'pad': 2})

        if titles is not None:
            ax.set_title(titles[i], color='red')

        ax.axis('off')

    for ax, i in zip(flat_axes[6:12], n_range):
        img = img_arrays[i,...,0]
        img = exposure.equalize_hist(img)

        c_img = ax.imshow(img, cmap='gray')

        plot_heat_map = ax.imshow(heatmap_1[i], cmap=cmap, alpha=alpha_1, vmin=vmin, vmax=vmax)

        if titles is not None:
            ax.set_title(titles[i], color='red')

        ax.axis('off')

    for ax, i in zip(flat_axes[12:18], n_range):
        img = img_arrays[i,...,0]
        img = exposure.equalize_hist(img)

        c_img = ax.imshow(img, cmap='gray')

        plot_heat_map = ax.imshow(heatmap_2[i], cmap=cmap, alpha=alpha_2, vmin=vmin, vmax=vmax)

        if titles is not None:
            ax.set_title(titles[i], color='red')

        ax.axis('off')

    if save_name is not None:
        plt.savefig(save_name, dpi=100, format='tiff', bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
