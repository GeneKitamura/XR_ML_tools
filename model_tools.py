import pandas as pd
import math
import tensorflow as tf
import tensorflow_addons as tfa
import time
import sys
import numpy as np
import re
import os
import matplotlib.pyplot as plt

from skimage import io
from glob import glob
from sklearn import metrics

from .fetch_data import data_get, prepare_dataset
from .infer_tools import np_preprocess

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def split_train_val_df(labels_df, label_types=None, train_split=0.8):
    # labels_df with meaningful index to link back to whole df
    train_idx = {}
    if label_types is None:
        label_types = ['pos_label', 'hard_label']

    for label_type in label_types:
        train_idx[label_type] = []
        for label in labels_df[label_type].unique():
            c_df = labels_df[labels_df[label_type] == label]
            tn = int(c_df.shape[0] * train_split)
            tidx = c_df.sample(tn).index.tolist()
            train_idx[label_type] = train_idx[label_type] + tidx
        train_idx[label_type] = sorted(train_idx[label_type])

    return train_idx

def split_tr_val_te(input_df, label=None):
    cdf = input_df.copy()
    traini = split_train_val_df(cdf, [label], train_split=0.7)[label]
    cdf['tr_val_te'] = 2  # 0:train, 1:val, 2:test
    cdf.loc[traini, 'tr_val_te'] = 0

    nontrain_df = cdf[~cdf.index.isin(traini)]
    vali = split_train_val_df(nontrain_df, [label], train_split=0.5)[label]
    cdf.loc[vali, 'tr_val_te'] = 1
    testi = cdf[~cdf.index.isin(traini + vali)].index

    cdf['new_train'] = False
    cdf.loc[traini, 'new_train'] = True

    return cdf

def load_densenet(input_size=224, n_class=3, activation=None):
    base_model = keras.applications.densenet.DenseNet121(include_top=False, input_shape=(input_size, input_size, 3), pooling='avg',  weights='imagenet')
    x = base_model.output  # (None, 1024)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(n_class, activation=activation, kernel_regularizer=keras.regularizers.l2())(x)  # no softmax for stability
    model = Model(inputs=base_model.input, outputs=x)

    return model

def preprocess_densenet(uint16=False):
    if not uint16:
        pixel_max = 255.
    else:
        pixel_max = 65535.

    def inner_fxn(img, label):
        # for densenet, mode='torch'
        img = tf.cast(img, tf.float32)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        c0 = (img[..., 0] / pixel_max - mean[0]) / std[0]
        c1 = (img[..., 1] / pixel_max - mean[0]) / std[1]
        c2 = (img[..., 2] / pixel_max - mean[0]) / std[2]
        new_img = tf.stack([c0, c1, c2], axis=-1)
        return new_img, label

    return inner_fxn

def trunc_name(*datapoint):
    img, label, name = datapoint
    return img, label

def pass_through(x, y):
    return x, y

def position_augment(final_size=224):

    def inner_fxn(img, label):
    # data should be in uint8 or uint16 for brightness/contrast
        img = tf.image.random_brightness(img, 0.3)
        img = tf.image.random_contrast(img, 0.7, 1.3)
        img = tf.image.random_crop(img, (final_size, final_size, 3))
        return img, label

    return inner_fxn

def rotate_augment(deg_rotation=180, final_size=224):

    def inner_fxn(img, label):
        img, label = position_augment(final_size)(img, label) #first while dtype is uint8/uint16
        deg_to_radians = tf.cast(math.pi / 180, tf.float32)
        img = tf.cast(img, tf.float32)
        label = tf.cast(label, tf.float32) #angle of correction

        #new
        rand_rot_val = tf.random.uniform([1], minval=-deg_rotation, maxval=deg_rotation) #[-deg_rotation, deg_rotation)
        angle_change = label + rand_rot_val # can be greater than 180
        # labels are degree of correction, tfa rotates in opposite direction
        rotated_img = tfa.image.rotate(img, -angle_change * deg_to_radians, interpolation='BILINEAR')
        # rotated up to deg_rotation, correct image rotation and change by rand_rot_val

        norm_rot_angle = rand_rot_val / deg_rotation
        # to [-1, 1] #Now in direction of rotation, not degree of correction
        # range of rand_rot_val, so only up to 20, 40, 60, etc...

        #old
        # neut_img = tfa.image.rotate(img, -label * deg_to_radians, interpolation='BILINEAR')
        # # labels are degree of correction, tfa rotates in opposite direction
        # rand_rot_val = tf.random.uniform([1], minval=-179, maxval=180) #[-179, 180)
        # rotated_img = tfa.image.rotate(neut_img, -rand_rot_val * deg_to_radians, interpolation='BILINEAR')
        # norm_rot_angle = rand_rot_val / 180 # to [-1, 1] #In direction of rotation, not degree of correction

        return rotated_img, norm_rot_angle

    return inner_fxn

def val_rot_map(deg_rotation=180):

    def inner_fxn(img, label):
        norm_rot_angle = -label / deg_rotation # in direction of rotation
        # problem when label is greater than augmentation def_rotation value (results in value > 1 when NN output is maxed at 1)
        return img, norm_rot_angle
    return inner_fxn

def train_numpy_keras(get_numpy_ds, batch_size=20, augment=position_augment(), val_map=pass_through,
                      preprocess_map=preprocess_densenet, preprocess_uint16=False, epochs=20,
                      save_path=None, excel_path=None, n_class=None, activation=None,
                      loss=None, metrics=None, monitor='val_loss', net_input=224):

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds, n_train, val_ds, n_val, test_ds, n_test, n_labels = get_numpy_ds
    train_steps = math.ceil(n_train/batch_size)
    val_steps = math.ceil(n_val/batch_size)

    train_ds = train_ds.map(trunc_name) #get rid of name/idx
    val_ds = val_ds.map(trunc_name) #get rid of name/idx

    inner_preprocess = preprocess_map(uint16=preprocess_uint16)

    # to see items
    # iter_ds = iter(train_ds)
    # one_item = next(iter_ds)

    #shuffle MUST be after cache, otherwise data is always fed in the same way
    #batch after map for each tf random function to have different values on each element, rather than same value per batch
    train_ds = train_ds.cache().map(augment).map(inner_preprocess).shuffle(buffer_size=1500).batch(batch_size).repeat().prefetch(AUTOTUNE)
    # no need to repeat val_ds; it will run from top every time.
    val_ds = val_ds.cache().map(val_map).map(inner_preprocess).batch(batch_size).prefetch(AUTOTUNE)

    if n_class is None:
        n_class = n_labels
    model = load_densenet(input_size=net_input, n_class=n_class, activation=activation)

    optimizer = Adam(lr=1e-4)
    if loss is None:
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    if metrics is None:
        metrics = [keras.metrics.SparseCategoricalAccuracy()]
    # can provide logits for SparseCategoricalAccuracy since argmax of logits and probs are the same
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.2, patience=10, min_lr=1e-6, min_delta=1e-3),
        tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=30, min_delta=1e-3),
        tf.keras.callbacks.ModelCheckpoint(monitor=monitor, filepath=save_path, save_best_only=True, save_weights_only=True)
    ]

    train_start = time.time()
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks, steps_per_epoch=train_steps, validation_steps=val_steps)

    print('total time {:.2f}'.format(time.time() - train_start), 3)

    c_df = pd.DataFrame()
    for key, val in history.history.items():
        c_df[key] = val
    c_df.to_excel(excel_path, index=False)

def eval_cat_model(get_numpy_ds, model_weights, net_input=224):
    _, _, val_ds, n_val, test_ds, n_test, n_labels = get_numpy_ds

    model = load_densenet(input_size=net_input, n_class=n_labels)
    model.load_weights(model_weights)

    total_inp_data = next(iter(val_ds.batch(n_val)))
    processed_val = preprocess_densenet()(total_inp_data[0], total_inp_data[1])
    predictions = model.predict(processed_val)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = total_inp_data[1].numpy()
    disc_bool = pred_labels != true_labels

    disc_imgs = total_inp_data[0].numpy()[disc_bool]
    incorrect_names = total_inp_data[2].numpy()[disc_bool]
    incorrect_labels = pred_labels[disc_bool]
    correct_labels = true_labels[disc_bool]

    for label in range(n_labels):
        fpr, tpr, _ = metrics.roc_curve(y_score=predictions[:, label], y_true=true_labels, pos_label=label)
        auc = metrics.auc(fpr, tpr)
        print('auc for {} is {}'.format(label, auc))

    print(metrics.classification_report(y_true=true_labels, y_pred=pred_labels))

    # tile_alt_imshow(disc_imgs, titles=list(zip(correct_labels, incorrect_labels, incorrect_names)))

def eval_scalar_model(params=None, as_unint16=True, scalar_val=120, use_ds=False, flip_pred=-1):
    if params is None:
        params = {
            'label_col': 'angle',
            'train_bool_col': 'new_train',
            'df_excel': './pos_hard_processed/rot0to7000.xlsx',
            'pos_df': './pos_hard_processed/pos_correct0to7000.xlsx',
            'test_set_col': 'tr_val_te',
            'flip_RtoL_col': 'true_label',
            'val_npz': './pos_hard_npz/224_0to7000.npz',
            'model_weight_root': './rot/model_7000_{}deg',
            'out_excel_root': './rot/rot_out_7000_{}deg.xlsx'
        }

    label_col = params['label_col']
    train_bool_col = params['train_bool_col']
    df_excel = params['df_excel']
    pos_df = params['pos_df']
    test_set_col = params['test_set_col']
    flip_RtoL_col = params['flip_RtoL_col']
    val_npz = params['val_npz']
    model_weights = params['model_weight_root'].format(scalar_val)
    out_excel = params['out_excel_root'].format(scalar_val)

    out_metric_final = pd.read_excel(out_excel)
    for i in ['loss', 'val_loss']:
        plt.plot(out_metric_final.index, out_metric_final[i])
        plt.title(i)

    if use_ds:
        val_map = val_rot_map(scalar_val)
        preprocess_map = preprocess_densenet
        inner_preprocess = preprocess_map(uint16=as_unint16)
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        (train_ds, n_train, val_ds, n_val, test_ds, n_test, n_labels) = data_get(
            df_excel, label_col, train_bool_col,
            train_npz=val_npz, val_npz=val_npz, img_as_uint16=as_unint16, flip_RtoL_col=flip_RtoL_col,
            npz_df=pos_df, test_set_col=test_set_col)

        c_ds = val_ds

        orig_ds = c_ds.batch(500)
        processed_ds = c_ds.map(trunc_name).cache().map(val_map).map(inner_preprocess).batch(500)

        processed_iter = iter(processed_ds)
        orig_iter = iter(orig_ds)

        orig_item = next(orig_iter)
        print('\norig')
        orig_imgs, orig_labels, orig_idx = orig_item

        processed_item = next(processed_iter)
        print('\nprocessed_item')
        proc_imgs, proc_labels = processed_item
        scaled_proc_labels = proc_labels * scalar_val

    else:
        (train_images, train_labels, train_idx, train_indices,
         val_images, val_labels, val_idx, val_indices,
         test_images, test_labels, test_idx, test_indices,n_labels) = prepare_dataset(
            df_excel, label_col, train_bool_col,
            train_npz=val_npz, val_npz=val_npz, img_as_uint16=as_unint16,  flip_RtoL_col=flip_RtoL_col,
            npz_df=pos_df, test_set_col=test_set_col)

        orig_imgs = val_images
        orig_labels = val_labels
        orig_idx = val_idx

        proc_imgs, orig_labels = np_preprocess(as_unint16)(orig_imgs, orig_labels)

    n_class = 1
    activation = tf.keras.activations.tanh
    net = load_densenet(n_class=n_class, activation=activation)
    net.load_weights(model_weights).expect_partial()

    pred_angle = net.predict(proc_imgs)
    scaled_pred_angle = flip_pred * pred_angle * scalar_val #pred angle opposite sign/direction of orig_labels
    scaled_pred_angle = scaled_pred_angle.reshape([proc_imgs.shape[0]])

    diff = np.abs(orig_labels - scaled_pred_angle)
    ci = 1.96 * np.std(diff) / np.sqrt(diff.shape[0])
    print('mean absolute error: {:.2f} +- {:.2f}'.format(np.mean(diff), ci))